from __future__ import annotations

from typing import Any, Iterable, Optional

import torch

from fastwam.utils.logging_config import get_logger

from .fastwam_idm import FastWAMIDM
from .lora import align_lora_dtype_device, has_lora, inject_lora_linear_layers, iter_lora_modules, lora_state_dict

logger = get_logger(__name__)


class FastWAMVideoLoRA(FastWAMIDM):
    """Video-only LoRA finetuning branch on top of FastWAM IDM."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_lora_config: dict[str, Any] = {}
        self.base_checkpoint_path_hint: Optional[str] = None

    def setup_video_lora(
        self,
        *,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_substrings: Optional[Iterable[str]] = None,
    ) -> None:
        if has_lora(self.video_expert):
            logger.warning("Video expert already has LoRA modules. Skipping reinjection.")
            return

        stats = inject_lora_linear_layers(
            self.video_expert,
            rank=int(rank),
            alpha=float(alpha),
            dropout=float(dropout),
            target_substrings=target_substrings,
        )
        if stats.num_replaced == 0:
            raise RuntimeError("LoRA injection replaced zero linear layers in `video_expert`.")

        self.video_lora_config = {
            "rank": int(rank),
            "alpha": float(alpha),
            "dropout": float(dropout),
            "target_substrings": list(target_substrings) if target_substrings is not None else [],
            "replaced_modules": list(stats.replaced_modules),
        }
        logger.info(
            "Injected video LoRA into %d linear layers.",
            stats.num_replaced,
        )

    def configure_trainable_modules(self) -> None:
        self.eval()
        self.requires_grad_(False)
        self.video_expert.train()
        for module in iter_lora_modules(self.video_expert):
            module.train()
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)

    def trainable_parameters(self):
        for module in iter_lora_modules(self.video_expert):
            yield module.lora_A
            yield module.lora_B

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        pred_video = self.video_expert(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        loss_total = self.loss_lambda_video * loss_video
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": 0.0,
        }
        return loss_total, loss_dict

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "video_expert_lora": lora_state_dict(self.video_expert),
            "video_lora_config": dict(self.video_lora_config),
            "base_checkpoint_path_hint": self.base_checkpoint_path_hint,
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    @staticmethod
    def _remap_linear_keys_for_lora_wrapped_module(
        state_dict: dict[str, torch.Tensor],
        target_state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        remapped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key in target_state_dict:
                remapped[key] = value
                continue

            mapped_key = None
            if key.endswith(".weight"):
                candidate = key[: -len(".weight")] + ".base.weight"
                if candidate in target_state_dict:
                    mapped_key = candidate
            elif key.endswith(".bias"):
                candidate = key[: -len(".bias")] + ".base.bias"
                if candidate in target_state_dict:
                    mapped_key = candidate

            remapped[mapped_key or key] = value
        return remapped

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "video_expert_lora" not in payload:
            if "mot" in payload:
                target_state = self.mot.state_dict()
                remapped = self._remap_linear_keys_for_lora_wrapped_module(payload["mot"], target_state)
                missing, unexpected = self.mot.load_state_dict(remapped, strict=False)
                logger.info(
                    "Loaded base/full checkpoint into LoRA-wrapped MoT with strict=False. Missing=%d Unexpected=%d",
                    len(missing),
                    len(unexpected),
                )
            elif "dit" in payload:
                logger.warning("Loading legacy `dit` checkpoint into video expert only.")
                target_state = self.video_expert.state_dict()
                remapped = self._remap_linear_keys_for_lora_wrapped_module(payload["dit"], target_state)
                missing, unexpected = self.video_expert.load_state_dict(remapped, strict=False)
                logger.info(
                    "Loaded legacy video checkpoint into LoRA-wrapped video expert with strict=False. Missing=%d Unexpected=%d",
                    len(missing),
                    len(unexpected),
                )
            else:
                raise ValueError(f"Checkpoint missing both `mot` and `video_expert_lora` keys: {path}")

            if payload.get("base_checkpoint_path_hint") is not None:
                self.base_checkpoint_path_hint = str(payload["base_checkpoint_path_hint"])
            align_lora_dtype_device(self.video_expert)
            if self.proprio_encoder is not None:
                if "proprio_encoder" in payload:
                    self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
                else:
                    logger.warning(
                        "Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params."
                    )
            elif "proprio_encoder" in payload:
                logger.warning(
                    "Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring."
                )

            if optimizer is not None and "optimizer" in payload:
                optimizer.load_state_dict(payload["optimizer"])
            return payload

        missing, unexpected = self.video_expert.load_state_dict(payload["video_expert_lora"], strict=False)
        logger.info(
            "Loaded video LoRA adapter into video expert with strict=False. Missing=%d Unexpected=%d",
            len(missing),
            len(unexpected),
        )
        if payload.get("base_checkpoint_path_hint") is not None:
            self.base_checkpoint_path_hint = str(payload["base_checkpoint_path_hint"])
        align_lora_dtype_device(self.video_expert)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload
