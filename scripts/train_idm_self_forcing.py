import logging
import types
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fastwam.runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    _resolve_train_device,
    build_datasets,
)
from fastwam.trainer import Wan22Trainer
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import setup_logging
from fastwam.models.wan22.wan_video_dit import sinusoidal_embedding_1d


register_default_resolvers()


def _cfg_get(cfg: DictConfig, key: str, default: Any) -> Any:
    if cfg is None or key not in cfg:
        return default
    return cfg[key]


def _patch_video_pre_for_absolute_frames(model, pre_state, timestep, frame_start):
    """Fix token time/RoPE when we call video pre_dit on a temporal chunk.

    WanVideoDiT.pre_dit always treats the first local latent frame as the observed
    first frame. During AR chunk rollout that is only true for global frame 0, so
    later chunks need absolute frame RoPE and nonzero denoising timestep.
    """
    f, h, w = pre_state["meta"]["grid_size"]
    tokens_per_frame = int(pre_state["meta"]["tokens_per_frame"])
    batch_size = int(pre_state["meta"]["batch_size"])
    device = pre_state["tokens"].device
    dtype = timestep.dtype

    token_timesteps = torch.ones(
        (batch_size, f, tokens_per_frame),
        dtype=dtype,
        device=device,
    ) * timestep.to(device=device, dtype=dtype).view(batch_size, 1, 1)
    if frame_start == 0:
        token_timesteps[:, 0, :] = 0
    token_timesteps = token_timesteps.reshape(batch_size, -1)
    token_t_emb = sinusoidal_embedding_1d(model.video_expert.freq_dim, token_timesteps.reshape(-1))
    t = model.video_expert.time_embedding(token_t_emb).reshape(batch_size, -1, model.video_expert.hidden_dim)
    pre_state["t"] = t
    pre_state["t_mod"] = model.video_expert.time_projection(t).unflatten(2, (6, model.video_expert.hidden_dim))

    freqs = torch.cat(
        [
            model.video_expert.freqs[0][frame_start : frame_start + f]
            .view(f, 1, 1, -1)
            .expand(f, h, w, -1),
            model.video_expert.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            model.video_expert.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1)
    pre_state["freqs"] = freqs.to(device)
    return pre_state


def _video_pre_dit_chunk(
    model,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    fuse_flag: bool,
    frame_start: int,
):
    pre = model.video_expert.pre_dit(
        x=latents,
        timestep=timestep,
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=fuse_flag,
    )
    return _patch_video_pre_for_absolute_frames(model, pre, timestep=timestep, frame_start=frame_start)


def _detach_video_cache(video_cache):
    return [{"k": layer["k"].detach(), "v": layer["v"].detach()} for layer in video_cache]


def _append_video_cache(
    old_cache,
    new_cache,
):
    if old_cache is None:
        return _detach_video_cache(new_cache)
    if len(old_cache) != len(new_cache):
        raise ValueError(f"Cache depth mismatch: old={len(old_cache)}, new={len(new_cache)}")
    merged = []
    for old_layer, new_layer in zip(old_cache, new_cache):
        merged.append(
            {
                "k": torch.cat([old_layer["k"], new_layer["k"].detach()], dim=1),
                "v": torch.cat([old_layer["v"], new_layer["v"].detach()], dim=1),
            }
        )
    return merged


def _run_video_tokens_with_cache(
    model,
    video_tokens: torch.Tensor,
    video_freqs: torch.Tensor,
    video_t_mod: torch.Tensor,
    video_context_payload,
    video_kv_cache,
    current_attention_mask,
):
    """Local video-cache forward, kept outside core FastWAM files on purpose."""
    expert = model.mot.mixtures["video"]
    x = video_tokens
    for layer_idx in range(model.mot.num_layers):
        block = expert.blocks[layer_idx]
        (
            q,
            k,
            v,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        ) = model.mot._build_expert_attention_io(
            expert=expert,
            block=block,
            x=x,
            freqs=video_freqs,
            t_mod=video_t_mod,
        )
        if video_kv_cache is not None:
            k = torch.cat([video_kv_cache[layer_idx]["k"], k], dim=1)
            v = torch.cat([video_kv_cache[layer_idx]["v"], v], dim=1)
        mixed = model.mot._mixed_attention(
            q_cat=q,
            k_cat=k,
            v_cat=v,
            attention_mask=current_attention_mask,
        )
        x = model.mot._apply_post_with_optional_checkpoint(
            block=block,
            residual_x=residual_x,
            gate_msa=gate_msa,
            shift_mlp=shift_mlp,
            scale_mlp=scale_mlp,
            gate_mlp=gate_mlp,
            use_gradient_checkpointing=use_gradient_checkpointing,
            mixed_slice=mixed,
            context_payload=video_context_payload,
        )
    return x


def _build_chunk_attention_mask(model, current_seq_len: int, cached_seq_len: int, tokens_per_frame: int, device):
    current_mask = model.video_expert.build_video_to_video_mask(
        video_seq_len=current_seq_len,
        video_tokens_per_frame=tokens_per_frame,
        device=device,
    )
    if cached_seq_len <= 0:
        return current_mask
    cached_mask = torch.ones((current_seq_len, cached_seq_len), dtype=torch.bool, device=device)
    return torch.cat([cached_mask, current_mask], dim=1)


def _prefill_video_cache_from_latents(
    model,
    latents: torch.Tensor,
    frame_start: int,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    fuse_flag: bool,
):
    timestep_zero = torch.zeros((latents.shape[0],), device=latents.device, dtype=latents.dtype)
    pre = _video_pre_dit_chunk(
        model=model,
        latents=latents,
        timestep=timestep_zero,
        context=context,
        context_mask=context_mask,
        fuse_flag=fuse_flag,
        frame_start=frame_start,
    )
    seq_len = int(pre["tokens"].shape[1])
    tokens_per_frame = int(pre["meta"]["tokens_per_frame"])
    mask = model.video_expert.build_video_to_video_mask(
        video_seq_len=seq_len,
        video_tokens_per_frame=tokens_per_frame,
        device=pre["tokens"].device,
    )
    cache = model.mot.prefill_video_cache(
        video_tokens=pre["tokens"],
        video_freqs=pre["freqs"],
        video_t_mod=pre["t_mod"],
        video_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
        video_attention_mask=mask,
    )
    return _detach_video_cache(cache), seq_len, tokens_per_frame


def _predict_video_chunk_noise(
    model,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    frame_start: int,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    fuse_flag: bool,
    video_cache,
    cached_seq_len: int,
) -> torch.Tensor:
    pre = _video_pre_dit_chunk(
        model=model,
        latents=latents,
        timestep=timestep,
        context=context,
        context_mask=context_mask,
        fuse_flag=fuse_flag,
        frame_start=frame_start,
    )
    tokens_per_frame = int(pre["meta"]["tokens_per_frame"])
    current_mask = _build_chunk_attention_mask(
        model=model,
        current_seq_len=int(pre["tokens"].shape[1]),
        cached_seq_len=int(cached_seq_len),
        tokens_per_frame=tokens_per_frame,
        device=pre["tokens"].device,
    )
    tokens = _run_video_tokens_with_cache(
        model=model,
        video_tokens=pre["tokens"],
        video_freqs=pre["freqs"],
        video_t_mod=pre["t_mod"],
        video_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
        video_kv_cache=video_cache,
        current_attention_mask=current_mask,
    )
    return model.video_expert.post_dit(tokens, pre)


def _predict_action_noise_with_cache_train(
    model,
    latents_action: torch.Tensor,
    timestep_action: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    video_kv_cache,
    attention_mask: torch.Tensor,
    video_seq_len: int,
) -> torch.Tensor:
    action_pre = model.action_expert.pre_dit(
        action_tokens=latents_action,
        timestep=timestep_action,
        context=context,
        context_mask=context_mask,
    )
    action_tokens = model.mot.forward_action_with_video_cache(
        action_tokens=action_pre["tokens"],
        action_freqs=action_pre["freqs"],
        action_t_mod=action_pre["t_mod"],
        action_context_payload={
            "context": action_pre["context"],
            "mask": action_pre["context_mask"],
        },
        video_kv_cache=video_kv_cache,
        attention_mask=attention_mask,
        video_seq_len=video_seq_len,
    )
    return model.action_expert.post_dit(action_tokens, action_pre)


def _self_forcing_training_loss(self, sample, tiled: bool = False):
    inputs = self.build_inputs(sample, tiled=tiled)
    input_latents = inputs["input_latents"]
    batch_size = int(input_latents.shape[0])
    context = inputs["context"]
    context_mask = inputs["context_mask"]
    action = inputs["action"]
    action_is_pad = inputs["action_is_pad"]
    fuse_flag = inputs["fuse_vae_embedding_in_latents"]
    enable_proprio_joint = bool(getattr(self, "enable_proprio_joint", False))

    chunk_size = int(getattr(self, "sf_video_chunk_size", 3))
    rollout_steps = int(getattr(self, "sf_video_rollout_steps", 4))
    context_noise = float(getattr(self, "sf_context_noise", 0.0))
    if chunk_size <= 0:
        raise ValueError(f"sf_video_chunk_size must be positive, got {chunk_size}")
    if rollout_steps <= 0:
        raise ValueError(f"sf_video_rollout_steps must be positive, got {rollout_steps}")
    if context_noise < 0:
        raise ValueError(f"sf_context_noise must be non-negative, got {context_noise}")

    timesteps, deltas = self.infer_video_scheduler.build_inference_schedule(
        num_inference_steps=rollout_steps,
        device=input_latents.device,
        dtype=input_latents.dtype,
    )

    video_cache = None
    cached_seq_len = 0
    generated_chunks = []
    video_losses = []
    first_frame_latents = inputs["first_frame_latents"]
    num_latent_frames = int(input_latents.shape[2])

    for frame_start in range(0, num_latent_frames, chunk_size):
        frame_end = min(frame_start + chunk_size, num_latent_frames)
        gt_chunk = input_latents[:, :, frame_start:frame_end]
        current = torch.randn_like(gt_chunk)
        if frame_start == 0 and first_frame_latents is not None:
            current[:, :, 0:1] = first_frame_latents

        exit_idx = int(torch.randint(0, rollout_steps, (1,), device=input_latents.device).item())
        for step_idx in range(exit_idx):
            timestep = timesteps[step_idx].expand(batch_size)
            with torch.no_grad():
                pred = _predict_video_chunk_noise(
                    model=self,
                    latents=current,
                    timestep=timestep,
                    frame_start=frame_start,
                    context=context,
                    context_mask=context_mask,
                    fuse_flag=fuse_flag,
                    video_cache=video_cache,
                    cached_seq_len=cached_seq_len,
                )
                if frame_start == 0 and first_frame_latents is not None:
                    pred[:, :, 0:1] = 0
                current = self.infer_video_scheduler.step(pred, deltas[step_idx], current)
                if frame_start == 0 and first_frame_latents is not None:
                    current[:, :, 0:1] = first_frame_latents

        timestep = timesteps[exit_idx].expand(batch_size)
        pred = _predict_video_chunk_noise(
            model=self,
            latents=current,
            timestep=timestep,
            frame_start=frame_start,
            context=context,
            context_mask=context_mask,
            fuse_flag=fuse_flag,
            video_cache=video_cache,
            cached_seq_len=cached_seq_len,
        )
        sigma = (timestep / float(self.train_video_scheduler.num_train_timesteps)).to(
            current.device, dtype=current.dtype
        ).view(batch_size, 1, 1, 1, 1).clamp(min=1e-4)
        target = (current - gt_chunk) / sigma
        if frame_start == 0 and first_frame_latents is not None:
            pred_loss = pred[:, :, 1:]
            target_loss = target[:, :, 1:]
        else:
            pred_loss = pred
            target_loss = target
        if pred_loss.numel() > 0:
            video_losses.append(F.mse_loss(pred_loss.float(), target_loss.float()))

        clean_chunk = (current - sigma * pred).detach()
        if frame_start == 0 and first_frame_latents is not None:
            clean_chunk[:, :, 0:1] = first_frame_latents
        generated_chunks.append(clean_chunk)

        cache_chunk = clean_chunk
        if context_noise > 0:
            context_timestep = torch.full(
                (batch_size,),
                context_noise,
                device=clean_chunk.device,
                dtype=clean_chunk.dtype,
            )
            cache_chunk = self.train_video_scheduler.add_noise(
                clean_chunk,
                torch.randn_like(clean_chunk),
                context_timestep,
            ).detach()
            if frame_start == 0 and first_frame_latents is not None:
                cache_chunk[:, :, 0:1] = first_frame_latents

        new_cache, new_seq_len, _ = _prefill_video_cache_from_latents(
            model=self,
            latents=cache_chunk,
            frame_start=frame_start,
            context=context,
            context_mask=context_mask,
            fuse_flag=fuse_flag,
        )
        video_cache = _append_video_cache(video_cache, new_cache)
        cached_seq_len += new_seq_len

    generated_video = torch.cat(generated_chunks, dim=2).detach()
    loss_video = torch.stack(video_losses).mean() if video_losses else input_latents.sum() * 0.0

    full_cache, video_seq_len, tokens_per_frame = _prefill_video_cache_from_latents(
        model=self,
        latents=generated_video,
        frame_start=0,
        context=context,
        context_mask=context_mask,
        fuse_flag=fuse_flag,
    )

    noise_action = torch.randn_like(action)
    timestep_action = self.train_action_scheduler.sample_training_t(
        batch_size=batch_size,
        device=self.device,
        dtype=action.dtype,
    )
    noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
    target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

    action_seq_len = int(noisy_action.shape[1])
    attention_mask = self._build_mot_attention_mask(
        video_seq_len=video_seq_len,
        action_seq_len=action_seq_len,
        video_tokens_per_frame=tokens_per_frame,
        device=input_latents.device,
    )
    pred_action = _predict_action_noise_with_cache_train(
        model=self,
        latents_action=noisy_action,
        timestep_action=timestep_action,
        context=context,
        context_mask=context_mask,
        video_kv_cache=full_cache,
        attention_mask=attention_mask,
        video_seq_len=video_seq_len,
    )

    action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
    if action_is_pad is not None:
        valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
    else:
        action_loss_per_sample = action_loss_token.mean(dim=1)
    action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
        action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
    )
    loss_action = (action_loss_per_sample * action_weight).mean()

    loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
    loss_dict = {
        "loss_video_self_forcing": self.loss_lambda_video * float(loss_video.detach().item()),
        "loss_action_generated_video_cond": self.loss_lambda_action * float(loss_action.detach().item()),
    }
    if enable_proprio_joint:
        train_prop_scheduler = getattr(self, "train_proprio_scheduler", None)
        pred_fn = getattr(self, "_predict_proprio_noise", None)
        target_builder = getattr(self, "_build_proprio_target_sequence", None)
        if train_prop_scheduler is None or not callable(pred_fn) or not callable(target_builder):
            raise RuntimeError(
                "Self-forcing proprio-joint is enabled, but required model hooks are missing. "
                "Use `FastWAMIDMProprioJoint` with `setup_proprio_joint_prediction(...)`."
            )

        target_proprio_clean = target_builder(sample=sample, latent_t=int(generated_video.shape[2]))
        if target_proprio_clean is None:
            raise ValueError(
                "Self-forcing proprio-joint requires `sample['future_proprio']` or `sample['proprio']`."
            )
        noise_proprio = torch.randn_like(target_proprio_clean)
        timestep_proprio = train_prop_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=target_proprio_clean.dtype,
        )
        noisy_proprio = train_prop_scheduler.add_noise(target_proprio_clean, noise_proprio, timestep_proprio)
        target_proprio = train_prop_scheduler.training_target(
            target_proprio_clean,
            noise_proprio,
            timestep_proprio,
        )
        pred_proprio = pred_fn(
            latents_proprio=noisy_proprio,
            timestep_proprio=timestep_proprio,
            context=context,
            context_mask=context_mask,
        )

        proprio_loss_token = F.mse_loss(pred_proprio.float(), target_proprio.float(), reduction="none").mean(dim=2)
        proprio_loss_per_sample = proprio_loss_token.mean(dim=1)
        proprio_weight = train_prop_scheduler.training_weight(timestep_proprio).to(
            proprio_loss_per_sample.device, dtype=proprio_loss_per_sample.dtype
        )
        loss_proprio = (proprio_loss_per_sample * proprio_weight).mean()
        lambda_proprio = float(getattr(self, "loss_lambda_proprio", 1.0))
        loss_total = loss_total + lambda_proprio * loss_proprio
        loss_dict["loss_proprio_joint"] = lambda_proprio * float(loss_proprio.detach().item())

    return loss_total, loss_dict


def run_self_forcing_training(cfg: DictConfig):
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
    )
    misc.register_work_dir(cfg.output_dir)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config_payload, f)

    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)

    sf_cfg = cfg.get("self_forcing", {})
    model.sf_video_chunk_size = int(_cfg_get(sf_cfg, "video_chunk_size", 3))
    model.sf_video_rollout_steps = int(_cfg_get(sf_cfg, "video_rollout_steps", 4))
    model.sf_context_noise = float(_cfg_get(sf_cfg, "context_noise", 0.0))
    model.training_loss = types.MethodType(_self_forcing_training_loss, model)

    train_ds, val_ds = build_datasets(cfg.data)
    trainer = Wan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    trainer.train()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_self_forcing_training(cfg)


if __name__ == "__main__":
    main()
