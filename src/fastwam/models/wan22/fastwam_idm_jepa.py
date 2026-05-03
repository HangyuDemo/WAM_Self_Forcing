from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .fastwam_idm import FastWAMIDM

logger = get_logger(__name__)


class FastWAMIDMJEPA(FastWAMIDM):
    """IDM variant that predicts JEPA latent trajectory instead of VAE video target."""

    def __init__(self, *args, loss_lambda_jepa: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_lambda_jepa = float(loss_lambda_jepa)

        self.jepa_encoder = None
        self.jepa_processor = None
        self.jepa_model_id: Optional[str] = None
        self.jepa_pool_mode: str = "mean"
        self.jepa_predict_future_only: bool = True

    def setup_jepa_encoder(
        self,
        jepa_model_id: str,
        freeze_encoder: bool = True,
        pool_mode: str = "mean",
        predict_future_only: bool = True,
        feature_dim_multiplier: int = 1,
    ) -> None:
        try:
            from transformers import AutoModel, AutoVideoProcessor
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "JEPA requires `transformers`. Please install it in the FastWAM environment."
            ) from exc

        self.jepa_model_id = str(jepa_model_id)
        self.jepa_pool_mode = str(pool_mode)
        self.jepa_predict_future_only = bool(predict_future_only)

        self.jepa_encoder = AutoModel.from_pretrained(self.jepa_model_id).to(self.device)
        self.jepa_processor = AutoVideoProcessor.from_pretrained(self.jepa_model_id)

        if freeze_encoder:
            self.jepa_encoder.requires_grad_(False)
            self.jepa_encoder.eval()

        jepa_hidden_dim = int(getattr(self.jepa_encoder.config, "hidden_size"))
        feature_dim_multiplier = int(feature_dim_multiplier)
        if feature_dim_multiplier <= 0:
            raise ValueError(f"`feature_dim_multiplier` must be positive, got {feature_dim_multiplier}")
        jepa_target_dim = int(jepa_hidden_dim * feature_dim_multiplier)
        video_hidden_dim = int(getattr(self.video_expert, "hidden_dim"))

        # Keep trainable heads inside self.dit (MoT), so trainer optimizer picks them up.
        self.dit.jepa_pred_head = nn.Sequential(
            nn.LayerNorm(video_hidden_dim),
            nn.Linear(video_hidden_dim, video_hidden_dim),
            nn.GELU(),
            nn.Linear(video_hidden_dim, jepa_target_dim),
        ).to(device=self.device, dtype=self.torch_dtype)

        self.dit.jepa_action_ctx_proj = nn.Sequential(
            nn.LayerNorm(jepa_target_dim),
            nn.Linear(jepa_target_dim, self.text_dim),
        ).to(device=self.device, dtype=self.torch_dtype)

        logger.info(
            "Initialized JEPA branch: model_id=%s, hidden=%d, target_dim=%d, pool_mode=%s, future_only=%s",
            self.jepa_model_id,
            jepa_hidden_dim,
            jepa_target_dim,
            self.jepa_pool_mode,
            self.jepa_predict_future_only,
        )

    def _check_jepa_ready(self) -> None:
        if self.jepa_encoder is None or self.jepa_processor is None:
            raise RuntimeError(
                "JEPA encoder/processor is not initialized. "
                "Use model target `create_fastwam_idm_jepa` or call `setup_jepa_encoder(...)`."
            )
        if not hasattr(self.dit, "jepa_pred_head") or not hasattr(self.dit, "jepa_action_ctx_proj"):
            raise RuntimeError("JEPA projection heads are missing on `self.dit`.")

    @torch.no_grad()
    def _pool_jepa_features(self, features: torch.Tensor, num_input_frames: int) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"JEPA features must be 3D [B,N,D], got {tuple(features.shape)}")

        if self.jepa_pool_mode == "mean":
            tubelet = int(getattr(self.jepa_encoder.config, "tubelet_size", 1) or 1)
            num_temporal = max(num_input_frames // tubelet, 1)
            if num_temporal > 0 and features.shape[1] % num_temporal == 0:
                tokens_per_step = int(features.shape[1] // num_temporal)
                features = features.view(features.shape[0], num_temporal, tokens_per_step, features.shape[2]).mean(dim=2)
        return features

    @torch.no_grad()
    def _encode_jepa_trajectory(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, C, T, H, W], range [-1, 1]
        Returns:
            trajectory: [B, T_jepa, D_jepa]
        """
        if video.ndim != 5:
            raise ValueError(f"`video` must be [B,C,T,H,W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`video` channel dim must be 3, got shape {tuple(video.shape)}")

        self._check_jepa_ready()

        bsz = int(video.shape[0])
        video_uint8 = ((video.detach().float().clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
        # [B, T, C, H, W]
        video_uint8 = video_uint8.permute(0, 2, 1, 3, 4).cpu()

        processed = []
        for i in range(bsz):
            proc_out = self.jepa_processor(videos=video_uint8[i], return_tensors="pt")
            pixel_values = proc_out.get("pixel_values_videos", None)
            if pixel_values is None:
                pixel_values = proc_out.get("pixel_values", None)
            if pixel_values is None:
                raise RuntimeError("AutoVideoProcessor output missing `pixel_values_videos`.")
            processed.append(pixel_values)

        pixel_values_b = torch.cat(processed, dim=0).to(self.device)
        encoder_dtype = next(self.jepa_encoder.parameters()).dtype
        pixel_values_b = pixel_values_b.to(dtype=encoder_dtype)

        try:
            features = self.jepa_encoder.get_vision_features(pixel_values_videos=pixel_values_b)
        except TypeError:
            features = self.jepa_encoder.get_vision_features(pixel_values_b)

        features = self._pool_jepa_features(features, num_input_frames=int(pixel_values_b.shape[1]))
        return features.to(device=self.device, dtype=self.torch_dtype)

    @torch.no_grad()
    def _encode_jepa_trajectory_multiview(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, V, C, T, H, W], range [-1, 1]
        Returns:
            trajectory: [B, T_jepa, V * D_jepa]
        """
        if video.ndim != 6:
            raise ValueError(f"`video` must be [B,V,C,T,H,W], got shape {tuple(video.shape)}")
        if video.shape[2] != 3:
            raise ValueError(f"`video` channel dim must be 3, got shape {tuple(video.shape)}")

        self._check_jepa_ready()

        bsz, num_views, _, num_frames, _, _ = video.shape
        video_uint8 = ((video.detach().float().clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
        # [B*V, T, C, H, W]
        video_uint8 = video_uint8.permute(0, 1, 3, 2, 4, 5).reshape(bsz * num_views, num_frames, 3, video.shape[-2], video.shape[-1]).cpu()

        processed = []
        for i in range(video_uint8.shape[0]):
            proc_out = self.jepa_processor(videos=video_uint8[i], return_tensors="pt")
            pixel_values = proc_out.get("pixel_values_videos", None)
            if pixel_values is None:
                pixel_values = proc_out.get("pixel_values", None)
            if pixel_values is None:
                raise RuntimeError("AutoVideoProcessor output missing `pixel_values_videos`.")
            processed.append(pixel_values)

        pixel_values_b = torch.cat(processed, dim=0).to(self.device)
        encoder_dtype = next(self.jepa_encoder.parameters()).dtype
        pixel_values_b = pixel_values_b.to(dtype=encoder_dtype)

        try:
            features = self.jepa_encoder.get_vision_features(pixel_values_videos=pixel_values_b)
        except TypeError:
            features = self.jepa_encoder.get_vision_features(pixel_values_b)

        features = self._pool_jepa_features(features, num_input_frames=int(pixel_values_b.shape[1]))
        if features.shape[0] != bsz * num_views:
            raise ValueError(
                "Unexpected JEPA feature batch shape for multiview encoding: "
                f"got {tuple(features.shape)}, expected batch={bsz * num_views}"
            )
        # Flatten order is [b0v0, b0v1, b1v0, b1v1, ...], so chunk(dim=0) would
        # mix different samples incorrectly. Reshape back to [B, V, T, D] first,
        # then fuse views on the feature dimension.
        features = features.view(bsz, num_views, features.shape[1], features.shape[2])
        features = features.permute(0, 2, 1, 3).reshape(bsz, features.shape[2], num_views * features.shape[3])
        return features.to(device=self.device, dtype=self.torch_dtype)

    def _predict_jepa_trajectory(
        self,
        pred_video_tokens: torch.Tensor,
        tokens_per_frame: int,
        target_steps: int,
    ) -> torch.Tensor:
        if pred_video_tokens.ndim != 3:
            raise ValueError(f"`pred_video_tokens` must be [B,S,D], got {tuple(pred_video_tokens.shape)}")
        if tokens_per_frame <= 0:
            raise ValueError(f"`tokens_per_frame` must be positive, got {tokens_per_frame}")
        if pred_video_tokens.shape[1] % tokens_per_frame != 0:
            raise ValueError(
                "Video token sequence cannot be reshaped by `tokens_per_frame`: "
                f"S={pred_video_tokens.shape[1]}, tokens_per_frame={tokens_per_frame}"
            )

        bsz, _, hidden_dim = pred_video_tokens.shape
        num_video_steps = int(pred_video_tokens.shape[1] // tokens_per_frame)
        pooled = pred_video_tokens.view(bsz, num_video_steps, tokens_per_frame, hidden_dim).mean(dim=2)
        pred_traj = self.dit.jepa_pred_head(pooled)

        if int(pred_traj.shape[1]) != int(target_steps):
            pred_traj = F.interpolate(
                pred_traj.transpose(1, 2),
                size=int(target_steps),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return pred_traj

    def training_loss(self, sample, tiled: bool = False):
        self._check_jepa_ready()

        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        fuse_flag = inputs["fuse_vae_embedding_in_latents"]

        # JEPA target from real video trajectory (teacher-forcing target).
        if "multi_camera_video" in sample:
            target_jepa_full = self._encode_jepa_trajectory_multiview(sample["multi_camera_video"])
        else:
            if "video" not in sample:
                raise ValueError("JEPA-IDM requires `sample['video']` or `sample['multi_camera_video']`.")
            target_jepa_full = self._encode_jepa_trajectory(sample["video"])
        if target_jepa_full.shape[1] < 2:
            raise ValueError(
                f"JEPA trajectory must have >=2 timesteps (obs+future), got {target_jepa_full.shape[1]}"
            )
        target_jepa_obs = target_jepa_full[:, :1]
        target_jepa_future = target_jepa_full[:, 1:]

        # Branch A: noisy video (used to predict JEPA latent trajectory).
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents_noisy = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents_noisy[:, :, 0:1] = inputs["first_frame_latents"]

        # Branch B: noisy action.
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        # Branch C: teacher-forcing cond-video for action denoising.
        cond_noise_mask = torch.rand((batch_size,), device=self.device) < float(self.video_cond_noise_prob)
        timestep_video_cond = torch.zeros_like(timestep_video, dtype=input_latents.dtype, device=self.device)
        latents_cond = input_latents
        if bool(cond_noise_mask.any()):
            timestep_video_cond_sampled = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=input_latents.dtype,
            )
            timestep_video_cond = torch.where(cond_noise_mask, timestep_video_cond_sampled, timestep_video_cond)
            noise_video_cond = torch.randn_like(input_latents)
            latents_cond_noisy = self.train_video_scheduler.add_noise(
                input_latents, noise_video_cond, timestep_video_cond_sampled
            )
            cond_noise_selector = cond_noise_mask.view(batch_size, 1, 1, 1, 1)
            latents_cond = torch.where(cond_noise_selector, latents_cond_noisy, input_latents)
        if inputs["first_frame_latents"] is not None:
            latents_cond = latents_cond.clone()
            latents_cond[:, :, 0:1] = inputs["first_frame_latents"]

        video_pre_noisy = self.video_expert.pre_dit(
            x=latents_noisy,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_pre_cond = self.video_expert.pre_dit(
            x=latents_cond,
            timestep=timestep_video_cond,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        if video_pre_noisy["t_mod"].ndim != 4 or video_pre_cond["t_mod"].ndim != 4:
            raise ValueError(
                "Teacher-forcing requires token-wise `t_mod`; "
                "ensure `seperated_timestep=true` and `fuse_vae_embedding_in_latents=true`."
            )

        # Pass 1: run the video path and predict JEPA latent trajectory.
        # We intentionally keep the first pass action context text-only so that
        # the JEPA condition in pass 2 is sourced from the model's own predicted
        # JEPA trajectory rather than the ground-truth future trajectory.
        action_pre_stage1 = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        noisy_video_seq_len = int(video_pre_noisy["tokens"].shape[1])
        cond_video_seq_len = int(video_pre_cond["tokens"].shape[1])
        noisy_video_tokens_per_frame = int(video_pre_noisy["meta"]["tokens_per_frame"])
        cond_video_tokens_per_frame = int(video_pre_cond["meta"]["tokens_per_frame"])

        merged_video_tokens = torch.cat([video_pre_noisy["tokens"], video_pre_cond["tokens"]], dim=1)
        merged_video_freqs = torch.cat([video_pre_noisy["freqs"], video_pre_cond["freqs"]], dim=0)
        merged_video_t_mod = torch.cat([video_pre_noisy["t_mod"], video_pre_cond["t_mod"]], dim=1)
        merged_video_context_mask = torch.cat([video_pre_noisy["context_mask"], video_pre_cond["context_mask"]], dim=1)

        attention_mask = self._build_teacher_forcing_attention_mask(
            noisy_video_seq_len=noisy_video_seq_len,
            cond_video_seq_len=cond_video_seq_len,
            action_seq_len=action_pre_stage1["tokens"].shape[1],
            noisy_video_tokens_per_frame=noisy_video_tokens_per_frame,
            cond_video_tokens_per_frame=cond_video_tokens_per_frame,
            device=merged_video_tokens.device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": merged_video_tokens,
                "action": action_pre_stage1["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": merged_video_freqs,
                "action": action_pre_stage1["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre_noisy["context"],
                    "mask": merged_video_context_mask,
                },
                "action": {
                    "context": action_pre_stage1["context"],
                    "mask": action_pre_stage1["context_mask"],
                },
            },
            t_mod_all={
                "video": merged_video_t_mod,
                "action": action_pre_stage1["t_mod"],
            },
        )

        pred_video_tokens = tokens_out["video"][:, :noisy_video_seq_len]
        pred_jepa_full = self._predict_jepa_trajectory(
            pred_video_tokens=pred_video_tokens,
            tokens_per_frame=noisy_video_tokens_per_frame,
            target_steps=int(target_jepa_full.shape[1]),
        )
        pred_jepa_future = pred_jepa_full[:, 1:] if self.jepa_predict_future_only else pred_jepa_full
        jepa_target_for_loss = target_jepa_future if self.jepa_predict_future_only else target_jepa_full

        jepa_loss_per_sample = F.mse_loss(
            pred_jepa_future.float(),
            jepa_target_for_loss.float(),
            reduction="none",
        ).mean(dim=(1, 2))
        jepa_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            jepa_loss_per_sample.device, dtype=jepa_loss_per_sample.dtype
        )
        loss_jepa = (jepa_loss_per_sample * jepa_weight).mean()

        # Pass 2: feed the *predicted* JEPA trajectory into the action branch.
        if self.jepa_predict_future_only:
            pred_jepa_cond = torch.cat([target_jepa_obs, pred_jepa_future], dim=1)
        else:
            pred_jepa_cond = pred_jepa_full.clone()
            pred_jepa_cond[:, :1] = target_jepa_obs
        jepa_cond_ctx = self.dit.jepa_action_ctx_proj(pred_jepa_cond.to(dtype=context.dtype))
        jepa_cond_mask = torch.ones(
            (batch_size, jepa_cond_ctx.shape[1]),
            dtype=torch.bool,
            device=context_mask.device,
        )
        action_context = torch.cat([context, jepa_cond_ctx], dim=1)
        action_context_mask = torch.cat([context_mask, jepa_cond_mask], dim=1)

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=action_context,
            context_mask=action_context_mask,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=merged_video_tokens,
            video_freqs=merged_video_freqs,
            video_t_mod=merged_video_t_mod,
            video_context_payload={
                "context": video_pre_noisy["context"],
                "mask": merged_video_context_mask,
            },
            video_attention_mask=attention_mask[: noisy_video_seq_len + cond_video_seq_len, : noisy_video_seq_len + cond_video_seq_len],
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=noisy_video_seq_len + cond_video_seq_len,
        )
        pred_action = self.action_expert.post_dit(action_tokens, action_pre)
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

        loss_total = self.loss_lambda_jepa * loss_jepa + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_jepa": self.loss_lambda_jepa * float(loss_jepa.detach().item()),
            # Keep `loss_video` key for existing dashboards/scripts compatibility.
            "loss_video": self.loss_lambda_jepa * float(loss_jepa.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def infer_action(self, *args, **kwargs):
        raise NotImplementedError(
            "FastWAM JEPA-IDM currently implements the JEPA-conditioned training path only. "
            "The old IDM inference path was intentionally disabled because it does not consume the "
            "predicted JEPA trajectory and would silently produce inconsistent behavior."
        )

    @torch.no_grad()
    def infer_joint(self, *args, **kwargs):
        raise NotImplementedError(
            "FastWAM JEPA-IDM currently implements the JEPA-conditioned training path only. "
            "A dedicated JEPA-aware inference rollout is still needed."
        )
