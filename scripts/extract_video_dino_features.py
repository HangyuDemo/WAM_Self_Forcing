#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def _read_video_rgb_uint8(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return np.stack(frames, axis=0).astype(np.uint8)


def _pool_temporal_features(features: torch.Tensor, num_input_frames: int, tubelet_size: int) -> torch.Tensor:
    # Match FastWAM JEPA pooling behavior.
    if features.ndim != 3:
        raise ValueError(f"Expected 3D features [B,N,D], got {tuple(features.shape)}")
    num_temporal = max(int(num_input_frames) // max(int(tubelet_size), 1), 1)
    if num_temporal > 0 and features.shape[1] % num_temporal == 0:
        tokens_per_step = int(features.shape[1] // num_temporal)
        features = features.view(features.shape[0], num_temporal, tokens_per_step, features.shape[2]).mean(dim=2)
    return features


def main():
    parser = argparse.ArgumentParser(description="Extract JEPA-style temporal features from one rollout mp4.")
    parser.add_argument("--video", type=str, required=True, help="Input mp4 path.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/vjepa2-vitl-fpc64-256",
        help="HF model id for JEPA/VJEPA-style vision feature extraction.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--output", type=str, default=None, help="Output .pt path. Defaults to <video>.dino.pt")
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if args.output is None:
        out_pt = video_path.with_suffix(".jepa.pt")
    else:
        out_pt = Path(args.output).expanduser().resolve()
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    out_npy = out_pt.with_suffix(".npy")
    out_json = out_pt.with_suffix(".json")

    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    from transformers import AutoModel, AutoVideoProcessor

    frames = _read_video_rgb_uint8(video_path)  # [T, H, W, 3]
    model = AutoModel.from_pretrained(args.model_id).to(args.device).eval()
    processor = AutoVideoProcessor.from_pretrained(args.model_id)

    proc_out = processor(videos=frames, return_tensors="pt")
    pixel_values = proc_out.get("pixel_values_videos", None)
    if pixel_values is None:
        pixel_values = proc_out.get("pixel_values", None)
    if pixel_values is None:
        raise RuntimeError("Processor output missing `pixel_values_videos` and `pixel_values`.")

    pixel_values = pixel_values.to(args.device)
    model_dtype = next(model.parameters()).dtype
    if model_dtype in (torch.float16, torch.bfloat16, torch.float32):
        pixel_values = pixel_values.to(dtype=model_dtype)
    else:
        pixel_values = pixel_values.to(dtype=dtype)

    with torch.no_grad():
        try:
            features = model.get_vision_features(pixel_values_videos=pixel_values)
        except TypeError:
            try:
                features = model.get_vision_features(pixel_values)
            except Exception:
                out = model(pixel_values=pixel_values)
                if hasattr(out, "last_hidden_state"):
                    features = out.last_hidden_state
                else:
                    raise RuntimeError("Cannot obtain vision features from model output.")

    tubelet_size = int(getattr(model.config, "tubelet_size", 1) or 1)
    pooled = _pool_temporal_features(features, num_input_frames=int(pixel_values.shape[1]), tubelet_size=tubelet_size)
    pooled = pooled[0].detach().to(device="cpu", dtype=torch.float32).contiguous()  # [T_feat, D]

    torch.save(
        {
            "model_id": args.model_id,
            "video_path": str(video_path),
            "features": pooled,
            "num_video_frames": int(frames.shape[0]),
            "tubelet_size": int(tubelet_size),
            "pixel_values_shape": tuple(pixel_values.shape),
        },
        out_pt,
    )
    np.save(out_npy, pooled.numpy())
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": args.model_id,
                "video_path": str(video_path),
                "feature_shape": list(pooled.shape),
                "num_video_frames": int(frames.shape[0]),
                "tubelet_size": int(tubelet_size),
                "pixel_values_shape": [int(x) for x in pixel_values.shape],
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    print(f"Saved features: {out_pt}")
    print(f"Saved npy: {out_npy}")
    print(f"Saved meta: {out_json}")
    print(f"Feature shape: {tuple(pooled.shape)}")


if __name__ == "__main__":
    main()
