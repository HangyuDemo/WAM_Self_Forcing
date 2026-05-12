from __future__ import annotations

from typing import Any, Iterable, Optional

import torch

from fastwam.utils.logging_config import get_logger

from .fastwam_idm import FastWAMIDM

logger = get_logger(__name__)

try:
    from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
except ImportError:  # pragma: no cover - handled explicitly at runtime
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    get_peft_model_state_dict = None
    set_peft_model_state_dict = None


class FastWAMVideoPeftLoRA(FastWAMIDM):
    """Video-only PEFT LoRA finetuning branch on top of FastWAM IDM."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_lora_config: dict[str, Any] = {}
        self.base_checkpoint_path_hint: Optional[str] = None
        self.peft_adapter_name: str = "default"

    @staticmethod
    def _ensure_peft_available() -> None:
        if get_peft_model is None:
            raise ImportError(
                "PEFT-based video LoRA requires `peft` to be installed in the FastWAM environment."
            )

    def _resolve_target_modules(self, target_substrings: Optional[Iterable[str]]) -> list[str]:
        linear_names = [name for name, module in self.video_expert.named_modules() if isinstance(module, torch.nn.Linear)]
        if not linear_names:
            raise RuntimeError("Found zero nn.Linear layers in `video_expert`; cannot attach PEFT LoRA.")
        if target_substrings is None:
            return linear_names

        target_list = [str(item) for item in target_substrings]
        if len(target_list) == 0:
            return linear_names

        matched = [name for name in linear_names if any(sub in name for sub in target_list)]
        if not matched:
            raise RuntimeError(
                f"PEFT target_substrings={target_list} matched zero linear layers in `video_expert`."
            )
        return matched

    def setup_video_lora(
        self,
        *,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_substrings: Optional[Iterable[str]] = None,
    ) -> None:
        self._ensure_peft_available()
        if hasattr(self.video_expert, "peft_config"):
            logger.warning("Video expert already has PEFT adapters. Skipping reinjection.")
            return

        target_modules = self._resolve_target_modules(target_substrings)
        peft_cfg = LoraConfig(
            r=int(rank),
            lora_alpha=float(alpha),
            lora_dropout=float(dropout),
            bias="none",
            target_modules=target_modules,
        )
        self.video_expert = get_peft_model(self.video_expert, peft_cfg, adapter_name=self.peft_adapter_name)
        self.video_lora_config = {
            "rank": int(rank),
            "alpha": float(alpha),
            "dropout": float(dropout),
            "target_substrings": list(target_substrings) if target_substrings is not None else [],
            "target_modules": list(target_modules),
        }
        logger.info("Injected PEFT video LoRA into %d linear layers.", len(target_modules))

    def configure_trainable_modules(self) -> None:
        self.eval()
        self.requires_grad_(False)
        self.video_expert.train()
        for name, param in self.video_expert.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)

    def trainable_parameters(self):
        for name, param in self.video_expert.named_parameters():
            if param.requires_grad and "lora_" in name:
                yield param

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
            "loss_total": float(loss_total.detach().item()),
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_video_unweighted": float(loss_video.detach().item()),
            "loss_lambda_video": float(self.loss_lambda_video),
            "loss_action": 0.0,
        }
        return loss_total, loss_dict

    @staticmethod
    def _remap_linear_keys_for_peft_wrapped_module(
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
                candidate = key[: -len(".weight")] + ".base_layer.weight"
                if candidate in target_state_dict:
                    mapped_key = candidate
            elif key.endswith(".bias"):
                candidate = key[: -len(".bias")] + ".base_layer.bias"
                if candidate in target_state_dict:
                    mapped_key = candidate

            remapped[mapped_key or key] = value
        return remapped

    def save_checkpoint(self, path, optimizer=None, step=None):
        self._ensure_peft_available()
        payload = {
            "video_expert_peft_lora": get_peft_model_state_dict(
                self.video_expert, adapter_name=self.peft_adapter_name
            ),
            "video_lora_config": dict(self.video_lora_config),
            "base_checkpoint_path_hint": self.base_checkpoint_path_hint,
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        self._ensure_peft_available()
        payload = torch.load(path, map_location="cpu")
        if "video_expert_peft_lora" not in payload:
            if "mot" in payload:
                target_state = self.mot.state_dict()
                remapped = self._remap_linear_keys_for_peft_wrapped_module(payload["mot"], target_state)
                missing, unexpected = self.mot.load_state_dict(remapped, strict=False)
                logger.info(
                    "Loaded base/full checkpoint into PEFT-wrapped MoT with strict=False. Missing=%d Unexpected=%d",
                    len(missing),
                    len(unexpected),
                )
            elif "dit" in payload:
                logger.warning("Loading legacy `dit` checkpoint into PEFT-wrapped video expert only.")
                target_state = self.video_expert.state_dict()
                remapped = self._remap_linear_keys_for_peft_wrapped_module(payload["dit"], target_state)
                missing, unexpected = self.video_expert.load_state_dict(remapped, strict=False)
                logger.info(
                    "Loaded legacy video checkpoint into PEFT-wrapped video expert with strict=False. Missing=%d Unexpected=%d",
                    len(missing),
                    len(unexpected),
                )
            else:
                raise ValueError(f"Checkpoint missing both `mot` and `video_expert_peft_lora` keys: {path}")

            if payload.get("base_checkpoint_path_hint") is not None:
                self.base_checkpoint_path_hint = str(payload["base_checkpoint_path_hint"])
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

        incompatible = set_peft_model_state_dict(
            self.video_expert,
            payload["video_expert_peft_lora"],
            adapter_name=self.peft_adapter_name,
        )
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        logger.info(
            "Loaded video PEFT LoRA adapter into video expert with strict=False. Missing=%d Unexpected=%d",
            len(missing),
            len(unexpected),
        )
        if payload.get("base_checkpoint_path_hint") is not None:
            self.base_checkpoint_path_hint = str(payload["base_checkpoint_path_hint"])
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def load_base_then_lora(self, *, base_checkpoint_path: str, lora_checkpoint_path: str) -> dict[str, Any]:
        if not base_checkpoint_path:
            raise ValueError("`base_checkpoint_path` must be a non-empty string.")
        if not lora_checkpoint_path:
            raise ValueError("`lora_checkpoint_path` must be a non-empty string.")

        logger.info(
            "Restoring FastWAMVideoPeftLoRA using training-style order: "
            "PEFT-wrapped model -> base checkpoint -> LoRA checkpoint."
        )
        self.base_checkpoint_path_hint = str(base_checkpoint_path)
        self.load_checkpoint(str(base_checkpoint_path), optimizer=None)
        payload = self.load_checkpoint(str(lora_checkpoint_path), optimizer=None)
        logger.info(
            "Finished training-style FastWAMVideoPeftLoRA restore. base=%s lora=%s",
            base_checkpoint_path,
            lora_checkpoint_path,
        )
        return payload
