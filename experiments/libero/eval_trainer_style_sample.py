import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from libero.libero import benchmark

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.eval_libero_single import (
    _denormalize_action,
    _load_eval_checkpoints,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
)
from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
)
from fastwam.trainer import Wan22Trainer
from fastwam.utils.logging_config import setup_logging
from fastwam.utils import misc
from fastwam.utils.video_io import save_mp4
from fastwam.utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim


def _build_eval_dataset(cfg: DictConfig):
    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    work_dir = misc.get_work_dir()
    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    train_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    train_cfg["pretrained_norm_stats"] = str(dataset_stats_path)
    train_cfg["is_training_set"] = True
    dataset = instantiate(train_cfg)
    return dataset, dataset_stats_path


def _select_eval_index(dataset, cfg: DictConfig) -> int:
    explicit_index = cfg.EVALUATION.get("sample_index")
    if explicit_index is not None:
        index = int(explicit_index)
    else:
        eval_seed = int(cfg.EVALUATION.get("sample_seed", cfg.get("seed", 42)))
        rng = torch.Generator(device="cpu").manual_seed(eval_seed)
        index = int(torch.randint(0, len(dataset), (1,), generator=rng).item())
    if index < 0 or index >= len(dataset):
        raise IndexError(f"EVALUATION.sample_index={index} is out of bounds for dataset of size {len(dataset)}.")
    return index


def _resolve_sequence_cfg(dataset, cfg: DictConfig) -> tuple[str, int, int]:
    mode = str(cfg.EVALUATION.get("sequence_sampling_mode", "single_clip")).strip().lower()
    num_clips = int(cfg.EVALUATION.get("sequence_num_clips", 1))
    if num_clips < 1:
        raise ValueError(f"EVALUATION.sequence_num_clips must be >= 1, got {num_clips}")

    if mode not in {"single_clip", "consecutive_same_episode"}:
        raise ValueError(
            f"Unsupported EVALUATION.sequence_sampling_mode={mode}. "
            "Expected one of: ['single_clip', 'consecutive_same_episode']."
        )

    stride = cfg.EVALUATION.get("sequence_stride")
    if stride is None:
        stride = max(int(getattr(dataset, "num_frames", 1)) - 1, 1)
    else:
        stride = int(stride)
    if stride < 1:
        raise ValueError(f"EVALUATION.sequence_stride must be >= 1, got {stride}")
    return mode, num_clips, stride


def _find_episode_bounds(base_dataset, sample_index: int) -> tuple[int, int, int]:
    episode_from = base_dataset.episode_data_index["from"].tolist()
    episode_to = base_dataset.episode_data_index["to"].tolist()
    for episode_id, (ep_start, ep_end) in enumerate(zip(episode_from, episode_to)):
        ep_start = int(ep_start)
        ep_end = int(ep_end)
        if ep_start <= sample_index < ep_end:
            return episode_id, ep_start, ep_end
    raise IndexError(f"Could not locate sample_index={sample_index} inside episode_data_index.")


def _select_eval_indices(dataset, cfg: DictConfig) -> list[int]:
    mode, num_clips, stride = _resolve_sequence_cfg(dataset, cfg)
    first_index = _select_eval_index(dataset, cfg)
    if mode == "single_clip" or num_clips == 1:
        return [first_index]

    base_dataset = dataset.lerobot_dataset
    clip_num_frames = int(getattr(dataset, "num_frames", 1))
    explicit_index = cfg.EVALUATION.get("sample_index")

    if explicit_index is not None:
        episode_id, ep_start, ep_end = _find_episode_bounds(base_dataset, first_index)
        max_first_start = ep_end - clip_num_frames - stride * (num_clips - 1)
        if first_index > max_first_start:
            raise ValueError(
                f"EVALUATION.sample_index={first_index} is too close to the end of episode {episode_id} "
                f"for {num_clips} consecutive clips with stride={stride} and clip_num_frames={clip_num_frames}. "
                f"Max valid starting index is {max_first_start}."
            )
        return [first_index + i * stride for i in range(num_clips)]

    rng = torch.Generator(device="cpu").manual_seed(int(cfg.EVALUATION.get("sample_seed", cfg.get("seed", 42))))
    episode_from = base_dataset.episode_data_index["from"].tolist()
    episode_to = base_dataset.episode_data_index["to"].tolist()
    episode_perm = torch.randperm(len(episode_from), generator=rng).tolist()

    for episode_id in episode_perm:
        ep_start = int(episode_from[episode_id])
        ep_end = int(episode_to[episode_id])
        max_first_start = ep_end - clip_num_frames - stride * (num_clips - 1)
        if max_first_start < ep_start:
            continue
        if max_first_start == ep_start:
            first_start = ep_start
        else:
            first_start = int(torch.randint(ep_start, max_first_start + 1, (1,), generator=rng).item())
        return [first_start + i * stride for i in range(num_clips)]

    raise RuntimeError(
        f"No episode is long enough for {num_clips} consecutive clips with stride={stride} and clip_num_frames={clip_num_frames}."
    )


def _denormalize_action_metrics(sample: dict[str, Any], pred_action: torch.Tensor, gt_action: torch.Tensor, dataset) -> tuple[float, float]:
    if sample["proprio"] is None:
        raise ValueError("Eval sample must contain `proprio` for action denormalization.")
    proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
    processor = dataset.lerobot_dataset.processor

    denorm_actions = {}
    action_meta = processor.shape_meta["action"]
    state_meta = processor.shape_meta["state"]
    for action_name, raw_action in (("pred", pred_action), ("gt", gt_action)):
        if raw_action.ndim == 2:
            action_btd = raw_action.unsqueeze(0)
        elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
            action_btd = raw_action
        else:
            raise ValueError(
                f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
            )
        action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)
        batch = {
            "action": action_btd,
            "state": proprio,
        }
        batch = processor.action_state_merger.backward(batch)
        batch = processor.normalizer.backward(batch)
        merged_batch = {
            "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
            "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
        }
        merged_batch = processor.action_state_merger.forward(merged_batch)
        denorm_actions[action_name] = merged_batch["action"].unsqueeze(0)

    pred_action_denorm = denorm_actions["pred"]
    gt_action_denorm = denorm_actions["gt"]
    action_diff = pred_action_denorm - gt_action_denorm
    return float(action_diff.abs().mean().item()), float(action_diff.pow(2).mean().item())


def _save_metrics_json(metrics: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=True, indent=2)


def _get_action_video_freq_ratio(dataset) -> int:
    ratio = int(getattr(dataset, "action_video_freq_ratio", 1))
    if ratio < 1:
        raise ValueError(f"action_video_freq_ratio must be >= 1, got {ratio}")
    return ratio


def _capture_rollout_frames_for_action_sequence(
    pred_action: torch.Tensor,
    *,
    processor,
    task_suite_name: str,
    task_id: int,
    trial_index: int,
    num_steps_wait: int,
    binarize_gripper: bool,
    expected_num_video_frames: int,
    action_video_freq_ratio: int,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    if len(initial_states) == 0:
        raise ValueError(f"No initial states found for {task_suite_name} task_id={task_id}")
    trial_index = int(trial_index) % len(initial_states)
    initial_state = initial_states[trial_index]

    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed=0)
    try:
        env.reset()
        obs = env.set_init_state(initial_state)
        for _ in range(int(num_steps_wait)):
            obs, _, _, _ = env.step(get_libero_dummy_action())

        action_np = _denormalize_action(pred_action, processor)[0]
        action_np[..., -1] = action_np[..., -1] * 2 - 1
        action_np = invert_gripper_action(action_np)
        if binarize_gripper:
            action_np[..., -1] = np.sign(action_np[..., -1])

        rollout_frames = [get_libero_image(obs)]
        capture_steps = {
            min(step_idx * action_video_freq_ratio, action_np.shape[0])
            for step_idx in range(1, expected_num_video_frames)
        }
        capture_steps.discard(0)

        done = False
        for step_idx, action_step in enumerate(action_np, start=1):
            obs, _, done, _ = env.step(action_step.tolist())
            if step_idx in capture_steps:
                rollout_frames.append(get_libero_image(obs))
            if done:
                break

        while len(rollout_frames) < expected_num_video_frames:
            rollout_frames.append(get_libero_image(obs))

        return rollout_frames[:expected_num_video_frames], {
            "task_description": task_description,
            "trial_index": trial_index,
            "done": bool(done),
            "action_steps_executed": int(min(len(action_np), step_idx if len(action_np) > 0 else 0)),
        }
    finally:
        env.close()


def _save_two_row_video(pred_video_tensor: torch.Tensor, rollout_frames: list[Any], path: Path, fps: int) -> None:
    rollout_frame_tensors = []
    target_h = int(pred_video_tensor.shape[2])
    target_w = int(pred_video_tensor.shape[3])
    for frame in rollout_frames:
        if isinstance(frame, dict):
            image_parts = []
            for value in frame.values():
                value_array = np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
                image_parts.append(value_array)
            rollout_image = np.concatenate(image_parts, axis=1)
        elif isinstance(frame, Image.Image):
            rollout_image = np.array(frame.convert("RGB"))
        else:
            rollout_image = np.array(frame, copy=True)

        if rollout_image.shape[:2] != (target_h, target_w):
            rollout_image = np.array(
                Image.fromarray(rollout_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )
        rollout_frame_tensors.append(
            torch.from_numpy(rollout_image.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        )

    rollout_video_tensor = torch.stack(rollout_frame_tensors, dim=1)
    if rollout_video_tensor.shape[1] != pred_video_tensor.shape[1]:
        raise ValueError(
            f"Rollout/pred frame count mismatch: rollout={tuple(rollout_video_tensor.shape)} "
            f"pred={tuple(pred_video_tensor.shape)}"
        )

    stitched_video_tensor = torch.cat([pred_video_tensor, rollout_video_tensor], dim=2).contiguous()
    stitched_frames = []
    for t in range(stitched_video_tensor.shape[1]):
        frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
        stitched_frames.append(Image.fromarray(frame))
    save_mp4(stitched_frames, str(path), fps=fps)


def _evaluate_clip(model, dataset, sample_index: int, cfg: DictConfig) -> dict[str, Any]:
    raw_sample = dataset[sample_index]
    sample = Wan22Trainer._to_batched_eval_sample(raw_sample)

    with torch.no_grad():
        val_loss, loss_dict = model.training_loss(sample)
        val_loss = float(val_loss.detach().float().item())

    prompt = sample["prompt"][0]
    video0 = sample["video"][0]
    action = sample["action"][0] if sample["action"] is not None else None
    proprio = sample["proprio"][0, 0] if sample["proprio"] is not None else None
    input_image = video0[:, 0].unsqueeze(0)
    _, num_frames, _, _ = video0.shape

    infer_kwargs = {
        "input_image": input_image,
        "num_frames": num_frames,
        "action": action,
        "action_horizon": sample["action_horizon"],
        "proprio": proprio,
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "action_cfg_scale": float(cfg.EVALUATION.get("action_cfg_scale", 1.0)),
        "num_inference_steps": int(cfg.EVALUATION.get("num_inference_steps", cfg.get("eval_num_inference_steps", 10))),
        "seed": int(cfg.EVALUATION.get("sample_seed", cfg.get("seed", 42))),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    if sample["context"] is not None:
        infer_kwargs["prompt"] = None
        infer_kwargs["context"] = sample["context"][0]
        infer_kwargs["context_mask"] = sample["context_mask"][0]
    else:
        infer_kwargs["prompt"] = prompt

    pred = model.infer(**infer_kwargs)
    pred_video = pred["video"]
    pred_action = pred.get("action", None)

    pred_video_tensor = pil_frames_to_video_tensor(pred_video)
    gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
    if pred_video_tensor.shape != gt_video_tensor.shape:
        raise ValueError(
            "Prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

    psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
    ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

    action_l1 = None
    action_l2 = None
    if action is not None and pred_action is not None:
        action_l1, action_l2 = _denormalize_action_metrics(sample, pred_action, action, dataset)

    gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
    vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
    vae_recon_video = model._decode_latents(vae_latents, tiled=False)
    vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)
    if vae_video_tensor.shape != gt_video_tensor.shape:
        raise ValueError(
            "VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

    psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
    ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)
    psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
    ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

    stitched_video_tensor = torch.cat([pred_video_tensor, vae_video_tensor, gt_video_tensor], dim=2).contiguous()
    return {
        "sample_index": int(sample_index),
        "prompt": prompt,
        "num_frames": int(num_frames),
        "action_horizon": int(sample["action_horizon"]) if sample["action_horizon"] is not None else None,
        "val_loss": val_loss,
        "loss_dict": {k: float(v) for k, v in loss_dict.items()},
        "psnr_rg": float(psnr_rollout_vs_gt),
        "ssim_rg": float(ssim_rollout_vs_gt),
        "psnr_rd": float(psnr_rollout_vs_decode),
        "ssim_rd": float(ssim_rollout_vs_decode),
        "psnr_dg": float(psnr_decode_vs_gt),
        "ssim_dg": float(ssim_decode_vs_gt),
        "action_l1": action_l1,
        "action_l2": action_l2,
        "pred_action": pred_action.detach().cpu() if pred_action is not None else None,
        "pred_video_tensor": pred_video_tensor,
        "stitched_video_tensor": stitched_video_tensor,
    }


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero_idm")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")

    output_dir = Path(str(cfg.EVALUATION.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(output_dir))
    Path(misc.get_work_dir()).mkdir(parents=True, exist_ok=True)

    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model_device = str(cfg.EVALUATION.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_eval_checkpoints(model, cfg)
    model = model.to(model_device).eval()

    dataset, dataset_stats_path = _build_eval_dataset(cfg)
    sample_indices = _select_eval_indices(dataset, cfg)
    clip_results = [_evaluate_clip(model, dataset, sample_index, cfg) for sample_index in sample_indices]
    output_video_mode = str(cfg.EVALUATION.get("output_video_mode", "pred_vae_gt")).strip().lower()
    if output_video_mode not in {"pred_vae_gt", "pred_rollout"}:
        raise ValueError(
            f"Unsupported EVALUATION.output_video_mode={output_video_mode}. "
            "Expected one of: ['pred_vae_gt', 'pred_rollout']."
        )

    if len(clip_results) == 1:
        base_name = f"sample_{clip_results[0]['sample_index']:06d}"
    else:
        base_name = f"sequence_{clip_results[0]['sample_index']:06d}_to_{clip_results[-1]['sample_index']:06d}"

    rollout_meta = None
    if output_video_mode == "pred_vae_gt":
        stitched_video_tensor = torch.cat([item["stitched_video_tensor"] for item in clip_results], dim=1).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = output_dir / f"{base_name}--pred-vae-gt.mp4"
        save_mp4(stitched_frames, str(video_path), fps=int(cfg.EVALUATION.get("fps", 8)))
    else:
        if len(clip_results) != 1:
            raise ValueError(
                "EVALUATION.output_video_mode=pred_rollout currently supports a single clip only. "
                "Please set EVALUATION.sequence_num_clips=1."
            )
        pred_action = clip_results[0]["pred_action"]
        pred_video_tensor = clip_results[0]["pred_video_tensor"]
        if pred_action is None:
            raise ValueError("Model did not return `pred_action`, so pred_rollout video cannot be created.")
        rollout_frames, rollout_meta = _capture_rollout_frames_for_action_sequence(
            pred_action=pred_action,
            processor=dataset.lerobot_dataset.processor,
            task_suite_name=str(cfg.EVALUATION.task_suite_name),
            task_id=int(cfg.EVALUATION.task_id),
            trial_index=int(cfg.EVALUATION.get("env_trial_index", 0)),
            num_steps_wait=int(cfg.EVALUATION.get("num_steps_wait", 30)),
            binarize_gripper=bool(cfg.EVALUATION.get("binarize_gripper", True)),
            expected_num_video_frames=int(pred_video_tensor.shape[1]),
            action_video_freq_ratio=_get_action_video_freq_ratio(dataset),
        )
        video_path = output_dir / f"{base_name}--pred-rollout.mp4"
        _save_two_row_video(
            pred_video_tensor=pred_video_tensor,
            rollout_frames=rollout_frames,
            path=video_path,
            fps=int(cfg.EVALUATION.get("fps", 8)),
        )

    metric_keys = ["val_loss", "psnr_rg", "ssim_rg", "psnr_rd", "ssim_rd", "psnr_dg", "ssim_dg"]
    metrics = {
        "sample_index": clip_results[0]["sample_index"],
        "sample_indices": [item["sample_index"] for item in clip_results],
        "dataset_stats_path": str(dataset_stats_path),
        "base_ckpt_path": cfg.EVALUATION.get("base_ckpt_path"),
        "ckpt": str(cfg.ckpt),
        "prompt": clip_results[0]["prompt"],
        "num_frames": int(clip_results[0]["num_frames"]),
        "action_horizon": clip_results[0]["action_horizon"],
        "num_clips": len(clip_results),
        "sequence_sampling_mode": str(cfg.EVALUATION.get("sequence_sampling_mode", "single_clip")),
        "sequence_stride": int(cfg.EVALUATION.get("sequence_stride", max(int(getattr(dataset, "num_frames", 1)) - 1, 1))),
        "output_video_mode": output_video_mode,
        "video_path": str(video_path),
        "clip_metrics": [],
    }
    if rollout_meta is not None:
        metrics["env_trial_index"] = int(cfg.EVALUATION.get("env_trial_index", 0))
        metrics["rollout_task_suite_name"] = str(cfg.EVALUATION.task_suite_name)
        metrics["rollout_task_id"] = int(cfg.EVALUATION.task_id)
        metrics["rollout_done"] = bool(rollout_meta["done"])
        metrics["rollout_action_steps_executed"] = int(rollout_meta["action_steps_executed"])
        metrics["rollout_task_description"] = rollout_meta["task_description"]
    for key in metric_keys:
        metrics[key] = float(np.mean([item[key] for item in clip_results]))

    action_l1_values = [item["action_l1"] for item in clip_results if item["action_l1"] is not None]
    action_l2_values = [item["action_l2"] for item in clip_results if item["action_l2"] is not None]
    if action_l1_values:
        metrics["action_l1"] = float(np.mean(action_l1_values))
    if action_l2_values:
        metrics["action_l2"] = float(np.mean(action_l2_values))

    loss_keys = set()
    for item in clip_results:
        loss_keys.update(item["loss_dict"].keys())
    metrics["loss_dict"] = {
        key: float(np.mean([item["loss_dict"].get(key, 0.0) for item in clip_results]))
        for key in sorted(loss_keys)
    }

    for item in clip_results:
        clip_metric = {
            "sample_index": item["sample_index"],
            "prompt": item["prompt"],
            "val_loss": item["val_loss"],
            "psnr_rg": item["psnr_rg"],
            "ssim_rg": item["ssim_rg"],
            "psnr_rd": item["psnr_rd"],
            "ssim_rd": item["ssim_rd"],
            "psnr_dg": item["psnr_dg"],
            "ssim_dg": item["ssim_dg"],
            "loss_dict": item["loss_dict"],
        }
        if item["action_l1"] is not None:
            clip_metric["action_l1"] = item["action_l1"]
        if item["action_l2"] is not None:
            clip_metric["action_l2"] = item["action_l2"]
        metrics["clip_metrics"].append(clip_metric)

    metrics_path = output_dir / f"{base_name}--metrics.json"
    _save_metrics_json(metrics, metrics_path)

    print(json.dumps(metrics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
