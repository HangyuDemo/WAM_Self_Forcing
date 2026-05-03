from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from fastwam.utils.logging_config import get_logger

from .fastwam_idm import FastWAMIDM
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_dit import sinusoidal_embedding_1d

logger = get_logger(__name__)


class FastWAMIDMProprioJoint(FastWAMIDM):
    """Separate IDM route: jointly denoise video + proprio in stage-1, keep action stage unchanged."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_proprio_joint: bool = False
        self.loss_lambda_proprio: float = 0.0
        self.proprio_pred_dim: Optional[int] = None
        self.proprio_pred_freq_dim: int = 256
        self.proprio_predictor: Optional[nn.Module] = None
        self.proprio_time_mlp: Optional[nn.Module] = None
        self.proprio_context_mlp: Optional[nn.Module] = None
        self.train_proprio_scheduler: Optional[WanContinuousFlowMatchScheduler] = None
        self.infer_proprio_scheduler: Optional[WanContinuousFlowMatchScheduler] = None

    def setup_proprio_joint_prediction(
        self,
        enabled: bool = True,
        pred_dim: Optional[int] = None,
        hidden_dim: int = 1024,
        freq_dim: int = 256,
        train_shift: float = 5.0,
        infer_shift: float = 5.0,
        num_train_timesteps: int = 1000,
        lambda_proprio: float = 1.0,
    ) -> None:
        self.enable_proprio_joint = bool(enabled)
        if not self.enable_proprio_joint:
            self.loss_lambda_proprio = 0.0
            return

        if self.proprio_dim is None:
            raise ValueError("Proprio-joint prediction requires `proprio_dim` to be enabled.")

        self.proprio_pred_dim = int(self.proprio_dim if pred_dim is None else pred_dim)
        self.proprio_pred_freq_dim = int(freq_dim)
        hidden_dim = int(hidden_dim)
        self.loss_lambda_proprio = float(lambda_proprio)

        self.proprio_time_mlp = nn.Sequential(
            nn.Linear(self.proprio_pred_freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.proprio_context_mlp = nn.Sequential(
            nn.Linear(self.text_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.proprio_predictor = nn.Sequential(
            nn.Linear(self.proprio_pred_dim + hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.proprio_pred_dim),
        )

        self.train_proprio_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(num_train_timesteps),
            shift=float(train_shift),
        )
        self.infer_proprio_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(num_train_timesteps),
            shift=float(infer_shift),
        )

        self.proprio_time_mlp.to(device=self.device, dtype=self.torch_dtype)
        self.proprio_context_mlp.to(device=self.device, dtype=self.torch_dtype)
        self.proprio_predictor.to(device=self.device, dtype=self.torch_dtype)

    def configure_trainable_modules(self):
        super().eval()
        super().requires_grad_(False)
        self.dit.train()
        self.dit.requires_grad_(True)
        if self.proprio_encoder is not None:
            self.proprio_encoder.train()
            self.proprio_encoder.requires_grad_(True)
        if self.enable_proprio_joint:
            for module in (self.proprio_time_mlp, self.proprio_context_mlp, self.proprio_predictor):
                if module is not None:
                    module.train()
                    module.requires_grad_(True)

    def trainable_parameters(self):
        params = [p for p in self.dit.parameters() if p.requires_grad]
        if self.proprio_encoder is not None:
            params.extend([p for p in self.proprio_encoder.parameters() if p.requires_grad])
        if self.enable_proprio_joint:
            for module in (self.proprio_time_mlp, self.proprio_context_mlp, self.proprio_predictor):
                if module is not None:
                    params.extend([p for p in module.parameters() if p.requires_grad])
        return params

    def _masked_context_mean(self, context: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        valid = context_mask.to(device=context.device, dtype=context.dtype).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        return (context * valid).sum(dim=1) / denom

    def _predict_proprio_noise(
        self,
        latents_proprio: torch.Tensor,
        timestep_proprio: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enable_proprio_joint:
            raise RuntimeError("Proprio-joint branch is disabled.")
        if self.proprio_predictor is None or self.proprio_time_mlp is None or self.proprio_context_mlp is None:
            raise RuntimeError("Proprio-joint branch is not initialized.")

        bsz, seq_len, prop_dim = latents_proprio.shape
        if self.proprio_pred_dim is None or prop_dim != self.proprio_pred_dim:
            raise ValueError(f"Proprio latent dim must be {self.proprio_pred_dim}, got {prop_dim}")
        if timestep_proprio.shape[0] == 1 and bsz > 1:
            timestep_proprio = timestep_proprio.expand(bsz)

        timestep_embed = sinusoidal_embedding_1d(self.proprio_pred_freq_dim, timestep_proprio)
        timestep_feat = self.proprio_time_mlp(timestep_embed).unsqueeze(1).expand(-1, seq_len, -1)
        context_feat = self.proprio_context_mlp(
            self._masked_context_mean(context=context, context_mask=context_mask)
        ).unsqueeze(1).expand(-1, seq_len, -1)
        model_in = torch.cat([latents_proprio, timestep_feat, context_feat], dim=-1)
        return self.proprio_predictor(model_in)

    def _downsample_like_video_latents(self, x: torch.Tensor, latent_t: int, name: str) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"`{name}` must be [B,T,D], got {tuple(x.shape)}")
        if x.shape[1] == latent_t:
            return x

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}")
        if (x.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                f"`{name}` length cannot align with video downsample: T={x.shape[1]}, temporal_factor={temporal_factor}."
            )
        tail = x[:, 1:, :]
        tail_grouped = tail.view(x.shape[0], -1, temporal_factor, x.shape[2])
        # Match video latent grouping granularity and pick each group's aligned endpoint.
        down = torch.cat([x[:, :1, :], tail_grouped[:, :, -1, :]], dim=1)
        if down.shape[1] != latent_t:
            raise ValueError(
                f"Downsampled `{name}` length mismatch: got {down.shape[1]}, expected {latent_t}."
            )
        return down

    def _build_proprio_target_sequence(self, sample: dict, latent_t: int) -> Optional[torch.Tensor]:
        if self.proprio_dim is None:
            return None
        source = sample.get("future_proprio", None)
        if source is None:
            source = sample.get("proprio", None)
        if source is None:
            return None
        if not isinstance(source, torch.Tensor):
            raise TypeError(f"Proprio target must be tensor, got {type(source)}")
        if source.ndim != 3:
            raise ValueError(f"Proprio target must be [B,T,D], got {tuple(source.shape)}")
        if source.shape[2] != self.proprio_dim:
            raise ValueError(f"Proprio target dim must be {self.proprio_dim}, got {source.shape[2]}")
        source = source.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        return self._downsample_like_video_latents(source, latent_t=latent_t, name="future_proprio")

    def _expand_proprio_to_video_frames(self, proprio_latent: torch.Tensor, num_video_frames: int) -> torch.Tensor:
        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}")
        head = proprio_latent[:, :1, :]
        tail = proprio_latent[:, 1:, :].repeat_interleave(temporal_factor, dim=1)
        expanded = torch.cat([head, tail], dim=1)
        if expanded.shape[1] < num_video_frames:
            pad_count = num_video_frames - expanded.shape[1]
            expanded = torch.cat([expanded, expanded[:, -1:, :].expand(-1, pad_count, -1)], dim=1)
        return expanded[:, :num_video_frames, :]

    def training_loss(self, sample, tiled: bool = False):
        base_loss_total, base_loss_dict = super().training_loss(sample=sample, tiled=tiled)
        if not self.enable_proprio_joint:
            return base_loss_total, base_loss_dict

        if self.train_proprio_scheduler is None:
            raise RuntimeError("Proprio-joint training enabled but scheduler is not initialized.")

        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        latent_t = int(input_latents.shape[2])
        context = inputs["context"]
        context_mask = inputs["context_mask"]

        target_proprio_clean = self._build_proprio_target_sequence(sample=sample, latent_t=latent_t)
        if target_proprio_clean is None:
            raise ValueError("Proprio-joint training requires `sample['future_proprio']` or `sample['proprio']`.")

        noise_proprio = torch.randn_like(target_proprio_clean)
        timestep_proprio = self.train_proprio_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=target_proprio_clean.dtype,
        )
        noisy_proprio = self.train_proprio_scheduler.add_noise(target_proprio_clean, noise_proprio, timestep_proprio)
        target_proprio = self.train_proprio_scheduler.training_target(
            target_proprio_clean,
            noise_proprio,
            timestep_proprio,
        )
        pred_proprio = self._predict_proprio_noise(
            latents_proprio=noisy_proprio,
            timestep_proprio=timestep_proprio,
            context=context,
            context_mask=context_mask,
        )

        proprio_loss_token = F.mse_loss(pred_proprio.float(), target_proprio.float(), reduction="none").mean(dim=2)
        proprio_loss_per_sample = proprio_loss_token.mean(dim=1)
        proprio_weight = self.train_proprio_scheduler.training_weight(timestep_proprio).to(
            proprio_loss_per_sample.device, dtype=proprio_loss_per_sample.dtype
        )
        loss_proprio = (proprio_loss_per_sample * proprio_weight).mean()

        loss_total = base_loss_total + self.loss_lambda_proprio * loss_proprio
        loss_dict = dict(base_loss_dict)
        loss_dict["loss_proprio"] = self.loss_lambda_proprio * float(loss_proprio.detach().item())
        return loss_total, loss_dict

    @torch.no_grad()
    def infer_joint(self, *args, **kwargs) -> dict[str, Any]:
        out = super().infer_joint(*args, **kwargs)
        if not self.enable_proprio_joint:
            return out
        if self.infer_proprio_scheduler is None:
            raise RuntimeError("Proprio-joint inference enabled but scheduler is not initialized.")

        prompt = kwargs.get("prompt", None)
        input_image = kwargs.get("input_image", None)
        num_video_frames = int(kwargs.get("num_video_frames"))
        proprio = kwargs.get("proprio", None)
        context = kwargs.get("context", None)
        context_mask = kwargs.get("context_mask", None)
        num_inference_steps = int(kwargs.get("num_inference_steps", 20))
        sigma_shift = kwargs.get("sigma_shift", None)
        seed = kwargs.get("seed", None)
        rand_device = kwargs.get("rand_device", "cpu")

        if input_image is None:
            raise ValueError("`input_image` is required.")
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)

        if proprio is not None:
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        _, _, height, width = input_image.shape
        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1

        proprio_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_proprio = torch.randn(
            (1, latent_t, int(self.proprio_pred_dim)),
            generator=proprio_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None and proprio.shape[1] == latents_proprio.shape[2]:
            latents_proprio[:, 0, :] = proprio

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_proprio.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video in zip(infer_timesteps_video, infer_deltas_video):
            timestep_proprio = step_t_video.unsqueeze(0).to(dtype=latents_proprio.dtype, device=self.device)
            pred_proprio = self._predict_proprio_noise(
                latents_proprio=latents_proprio,
                timestep_proprio=timestep_proprio,
                context=context,
                context_mask=context_mask,
            )
            latents_proprio = self.infer_proprio_scheduler.step(pred_proprio, step_delta_video, latents_proprio)
            if proprio is not None and proprio.shape[1] == latents_proprio.shape[2]:
                latents_proprio[:, 0, :] = proprio

        out["proprio_latent"] = latents_proprio[0].detach().to(device="cpu", dtype=torch.float32)
        out["proprio"] = self._expand_proprio_to_video_frames(
            latents_proprio.detach(), num_video_frames=num_video_frames
        )[0].detach().to(device="cpu", dtype=torch.float32)
        return out

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = super().save_checkpoint(path, optimizer=optimizer, step=step)
        if self.enable_proprio_joint:
            payload["proprio_joint_config"] = {
                "enabled": True,
                "pred_dim": self.proprio_pred_dim,
                "freq_dim": self.proprio_pred_freq_dim,
                "lambda_proprio": self.loss_lambda_proprio,
            }
            payload["proprio_predictor"] = None if self.proprio_predictor is None else self.proprio_predictor.state_dict()
            payload["proprio_time_mlp"] = None if self.proprio_time_mlp is None else self.proprio_time_mlp.state_dict()
            payload["proprio_context_mlp"] = None if self.proprio_context_mlp is None else self.proprio_context_mlp.state_dict()
            torch.save(payload, path)
        return payload

    def load_checkpoint(self, path, optimizer=None):
        payload = super().load_checkpoint(path, optimizer=optimizer)
        cfg = payload.get("proprio_joint_config")
        if isinstance(cfg, dict) and bool(cfg.get("enabled", False)):
            if not self.enable_proprio_joint:
                logger.warning(
                    "Checkpoint contains proprio-joint weights but current model has it disabled; ignoring proprio branch."
                )
                return payload
            if self.proprio_predictor is not None and isinstance(payload.get("proprio_predictor"), dict):
                self.proprio_predictor.load_state_dict(payload["proprio_predictor"], strict=True)
            if self.proprio_time_mlp is not None and isinstance(payload.get("proprio_time_mlp"), dict):
                self.proprio_time_mlp.load_state_dict(payload["proprio_time_mlp"], strict=True)
            if self.proprio_context_mlp is not None and isinstance(payload.get("proprio_context_mlp"), dict):
                self.proprio_context_mlp.load_state_dict(payload["proprio_context_mlp"], strict=True)
        return payload


def create_fastwam_idm_proprio_joint(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    proprio_scheduler=None,
    proprio_joint=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if isinstance(proprio_scheduler, DictConfig):
        proprio_scheduler = OmegaConf.to_container(proprio_scheduler, resolve=True)
    if isinstance(proprio_joint, DictConfig):
        proprio_joint = OmegaConf.to_container(proprio_joint, resolve=True)
    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)

    action_dit_config = {} if action_dit_config is None else action_dit_config
    video_scheduler = {} if video_scheduler is None else video_scheduler
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM IDM proprio-joint.")
    proprio_scheduler = dict(action_scheduler) if proprio_scheduler is None else proprio_scheduler
    proprio_joint = {} if proprio_joint is None else proprio_joint
    loss = {} if loss is None else loss

    model = FastWAMIDMProprioJoint.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )

    model.setup_proprio_joint_prediction(
        enabled=bool(proprio_joint.get("enabled", True)),
        pred_dim=(None if proprio_joint.get("pred_dim") is None else int(proprio_joint.get("pred_dim"))),
        hidden_dim=int(proprio_joint.get("hidden_dim", 1024)),
        freq_dim=int(proprio_joint.get("freq_dim", 256)),
        train_shift=float(proprio_scheduler.get("train_shift", action_scheduler["train_shift"])),
        infer_shift=float(proprio_scheduler.get("infer_shift", action_scheduler["infer_shift"])),
        num_train_timesteps=int(proprio_scheduler.get("num_train_timesteps", action_scheduler["num_train_timesteps"])),
        lambda_proprio=float(loss.get("lambda_proprio", 1.0)),
    )
    return model
