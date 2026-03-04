"""
Shared utilities for speaker/emotion encoders.

Provides:
- BaseEncoder: abstract base with resampling, wav normalization, projection
- interpolate_temporal: (B,T,D) → (B,T',D) via linear interpolation
- BACKEND_REGISTRY: raw_dim lookup for all supported backends
"""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Backend raw dimensions ────────────────────────────────────────────

SPEAKER_RAW_DIMS: dict[str, int] = {
    "wavlm_sv": 512,
    "ecapa_tdnn": 192,
    "resemblyzer": 256,
}

EMOTION_RAW_DIMS: dict[str, int] = {
    "emotion2vec_base": 768,
    "emotion2vec_plus": 768,
    "wav2vec2_ser": 1024,
    "hubert_ser": 1024,
}

ENCODER_SAMPLE_RATE = 16_000  # all encoders operate at 16 kHz


# ── Temporal interpolation helper ─────────────────────────────────────

def interpolate_temporal(
    x: torch.Tensor, target_len: int,
) -> torch.Tensor:
    """
    Interpolate (B, T, D) → (B, target_len, D) via linear interpolation.
    No-op if T == target_len.
    """
    if x.shape[1] == target_len:
        return x
    # (B, T, D) → (B, D, T) → interpolate → (B, D, T') → (B, T', D)
    return F.interpolate(
        x.permute(0, 2, 1), size=target_len, mode="linear", align_corners=False,
    ).permute(0, 2, 1)


# ── Base Encoder ──────────────────────────────────────────────────────

class BaseEncoder(nn.Module):
    """
    Abstract base for frozen pretrained encoders with a trainable projection.

    Subclasses implement:
        _load_backend(backend, device) → sets self.encoder, self.raw_dim
        _extract(wav) → raw embedding tensor(s)
    """

    raw_dim: int = 0
    sample_rate: int = ENCODER_SAMPLE_RATE

    def __init__(self, output_dim: int = 512, normalize: bool = True):
        super().__init__()
        self.output_dim = output_dim
        self.normalize = normalize
        self._resamplers: dict[tuple[int, int], object] = {}

    # ── Shared wav preprocessing ──────────────────────────────────────

    def _prepare_wav(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """Ensure (B, T) shape at self.sample_rate."""
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.ndim == 3:
            wav = wav.squeeze(1)

        if sr != self.sample_rate:
            import torchaudio

            key = (sr, self.sample_rate)
            if key not in self._resamplers:
                self._resamplers[key] = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav = self._resamplers[key].to(wav.device)(wav)

        return wav

    @staticmethod
    def _wav_to_list(wav: torch.Tensor) -> list:
        """Convert (B, T) tensor to list of 1D numpy arrays for HF extractors."""
        return [wav[i].cpu().numpy() for i in range(wav.shape[0])]

    # ── Projection utilities ──────────────────────────────────────────

    @staticmethod
    def make_projection(in_dim: int, out_dim: int) -> nn.Sequential:
        """Standard 2-layer MLP projection."""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def _project_and_normalize(
        self, raw: torch.Tensor, proj: nn.Sequential,
    ) -> torch.Tensor:
        """Project to output_dim and optionally L2-normalize."""
        raw = raw.to(proj[0].weight.device, dtype=proj[0].weight.dtype)
        out = proj(raw)
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out

    # ── Abstract interface ────────────────────────────────────────────

    @abstractmethod
    def extract_raw(self, wav: torch.Tensor, sr: int = 24000):
        """Extract raw (un-projected) embeddings. Return type depends on subclass."""
        ...

    @abstractmethod
    def forward(self, wav: torch.Tensor, sr: int = 24000, **kwargs):
        ...

    @abstractmethod
    def project_cached(self, raw, **kwargs):
        """Project precomputed raw embeddings through trainable heads."""
        ...
