import os

import hydra
import torch
from omegaconf import DictConfig

from fastwam.runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    _resolve_train_device,
    build_datasets,
)
from fastwam.trainer import Wan22Trainer
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import setup_logging
from fastwam.models.wan22.lora import has_lora, lora_enabled


register_default_resolvers()


class LoRAControlTrainer(Wan22Trainer):
    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()

        prompt = sample["prompt"][0]
        video0 = sample["video"][0]
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample["action_horizon"],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(**infer_kwargs)
        pred_action = pred.get("action", None)

        disable_lora_for_action_eval = os.environ.get("EVAL_DISABLE_LORA_FOR_ACTION", "1") == "1"
        if disable_lora_for_action_eval and action is not None and hasattr(model, "video_expert") and has_lora(model.video_expert):
            with lora_enabled(model.video_expert, False):
                pred_no_lora = model.infer(**infer_kwargs)
            pred_action = pred_no_lora.get("action", pred_action)

        # Keep upstream evaluate() for metrics/video serialization, but with our action prediction override.
        # Minimal duplication: patch pred action into current sample then reuse parent logic by monkey patching infer.
        original_infer = model.infer
        try:
            def _infer_override(**kwargs):
                out = pred.copy()
                out["action"] = pred_action
                return out
            model.infer = _infer_override
            result = super().evaluate()
        finally:
            model.infer = original_infer

        if was_dit_training:
            self._set_dit_only_train_mode()
        if result is not None:
            result["action_eval_disable_lora"] = 1.0 if disable_lora_for_action_eval else 0.0
        return result


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(
        log_level="INFO",
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)

    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = hydra.utils.instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    train_ds, val_ds = build_datasets(cfg.data)

    trainer = LoRAControlTrainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    trainer.train()


if __name__ == "__main__":
    main()
