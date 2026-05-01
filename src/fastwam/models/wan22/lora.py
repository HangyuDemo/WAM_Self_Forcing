from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"`rank` must be positive, got {rank}")

        self.base = base_linear
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.enabled = True

        self.base.requires_grad_(False)

        param_kwargs = {
            "device": base_linear.weight.device,
            "dtype": base_linear.weight.dtype,
        }
        self.lora_A = nn.Parameter(torch.empty(self.rank, base_linear.in_features, **param_kwargs))
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, self.rank, **param_kwargs))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        lora_out = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return out + lora_out * self.scaling

    def enable_adapter(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)


@dataclass
class LoRAInjectionStats:
    replaced_modules: list[str]

    @property
    def num_replaced(self) -> int:
        return len(self.replaced_modules)


def _should_replace_module(
    full_name: str,
    child_name: str,
    target_substrings: Iterable[str] | None,
) -> bool:
    if target_substrings is None:
        return True
    substrings = [s for s in target_substrings if s]
    if not substrings:
        return True
    return any(s in full_name or s == child_name for s in substrings)


def inject_lora_linear_layers(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_substrings: Iterable[str] | None = None,
    prefix: str = "",
) -> LoRAInjectionStats:
    replaced_modules: list[str] = []

    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and _should_replace_module(full_name, child_name, target_substrings):
            setattr(
                module,
                child_name,
                LoRALinear(
                    base_linear=child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                ),
            )
            replaced_modules.append(full_name)
            continue

        child_stats = inject_lora_linear_layers(
            child,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_substrings=target_substrings,
            prefix=full_name,
        )
        replaced_modules.extend(child_stats.replaced_modules)

    return LoRAInjectionStats(replaced_modules=replaced_modules)


def iter_lora_modules(module: nn.Module) -> Iterator[LoRALinear]:
    for submodule in module.modules():
        if isinstance(submodule, LoRALinear):
            yield submodule


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    state = {}
    for name, tensor in module.state_dict().items():
        if ".lora_A" in name or ".lora_B" in name:
            state[name] = tensor.detach().cpu()
    return state


def has_lora(module: nn.Module) -> bool:
    return any(isinstance(submodule, LoRALinear) for submodule in module.modules())


def align_lora_dtype_device(module: nn.Module) -> None:
    for submodule in iter_lora_modules(module):
        ref = submodule.base.weight
        if submodule.lora_A.device != ref.device or submodule.lora_A.dtype != ref.dtype:
            submodule.lora_A.data = submodule.lora_A.data.to(device=ref.device, dtype=ref.dtype)
        if submodule.lora_B.device != ref.device or submodule.lora_B.dtype != ref.dtype:
            submodule.lora_B.data = submodule.lora_B.data.to(device=ref.device, dtype=ref.dtype)


@contextlib.contextmanager
def lora_enabled(module: nn.Module, enabled: bool) -> Iterator[None]:
    lora_modules = list(iter_lora_modules(module))
    old_values = [m.enabled for m in lora_modules]
    try:
        for m in lora_modules:
            m.enable_adapter(enabled)
        yield
    finally:
        for m, old_value in zip(lora_modules, old_values):
            m.enable_adapter(old_value)
