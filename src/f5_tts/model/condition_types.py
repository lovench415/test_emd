from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import torch


@dataclass
class RawConditionBatch:
    speaker_raw: torch.Tensor | None = None
    emotion_global_raw: torch.Tensor | None = None
    emotion_frame_raw: torch.Tensor | None = None
    prosody_raw: torch.Tensor | None = None
    speaker_present: torch.Tensor | None = None
    emotion_global_present: torch.Tensor | None = None
    emotion_frame_mask: torch.Tensor | None = None
    prosody_mask: torch.Tensor | None = None


@dataclass
class ModelConditionBatch:
    speaker: torch.Tensor | None = None
    emotion_global: torch.Tensor | None = None
    emotion_frame: torch.Tensor | None = None
    prosody_frame: torch.Tensor | None = None       # (B, T, prosody_dim) for cross-attn
    prosody_direct: torch.Tensor | None = None       # (B, T, model_dim) for direct addition
    prosody_global: torch.Tensor | None = None       # (B, 11) global prosody stats for AdaLN
    speaker_present: torch.Tensor | None = None
    emotion_global_present: torch.Tensor | None = None
    emotion_frame_mask: torch.Tensor | None = None
    prosody_mask: torch.Tensor | None = None


@dataclass
class ConditioningOutputs:
    fused_global: torch.Tensor | None = None
    fused_present_mask: torch.Tensor | None = None
    adaln: list[torch.Tensor] | None = None
    frame_cond: torch.Tensor | None = None
    frame_mask: torch.Tensor | None = None
    prosody_cond: torch.Tensor | None = None        # (B, T, prosody_dim) for cross-attn
    prosody_direct: torch.Tensor | None = None       # (B, T, model_dim) for direct addition
    prosody_frame_mask: torch.Tensor | None = None


@dataclass
class ConditionPairBatch:
    cond: ModelConditionBatch
    uncond: ModelConditionBatch | None = None


class ConditioningRuntime:
    def __init__(
        self,
        *,
        apply_input: Callable[[torch.Tensor], torch.Tensor],
        adaln_for_block: Callable[[int], torch.Tensor | None],
        apply_block: Callable[[torch.Tensor, int, torch.Tensor | None, bool], torch.Tensor],
    ):
        self._apply_input = apply_input
        self._adaln_for_block = adaln_for_block
        self._apply_block = apply_block

    def apply_input(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_input(x)

    def adaln_for_block(self, block_idx: int):
        return self._adaln_for_block(block_idx)

    def apply_block(self, x: torch.Tensor, block_idx: int, x_mask: torch.Tensor | None = None, checkpoint_activations: bool = False) -> torch.Tensor:
        return self._apply_block(x, block_idx, x_mask, checkpoint_activations)
