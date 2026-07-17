"""
Learnable timbre encoder (MiniMax-Speech style).

Motivation
----------
A pretrained speaker-verification model (WavLM-SV) is trained to DISCRIMINATE
speakers (binary same/different), not to fully DESCRIBE timbre for synthesis. It
compresses a reference into one vector that is good for verification but can lose
timbre nuance, which limits clone fidelity.

MiniMax-Speech and similar SOTA systems instead use a *learnable* speaker encoder
that extracts timbre directly from the reference mel-spectrogram and is trained
JOINTLY with the TTS model, so it learns exactly the timbre features the synthesizer
needs — not the features a verification loss happens to reward.

This module is that encoder. It maps a reference mel (B, T, n_mels) to:
  - a GLOBAL timbre vector (B, timbre_dim)            — slow, utterance-level identity
  - optionally FRAME-level timbre (B, T, timbre_dim)  — fine-grained, time-varying

Design notes
------------
- Timbre changes slowly within an utterance, so the global vector is the primary
  output (attentive statistics pooling over frames). Frame-level output is optional
  (off by default) for a future multi-level path.
- Works on the SAME mel the model already computes (n_mels, hop, sr), so no extra
  feature pipeline is needed; at train time the target mel doubles as the reference.
- Kept small (a few conv blocks + attentive pooling) so it adds little compute and
  trains fast alongside the DiT.
- Output is L2-normalizable (optional) for stable fusion, mirroring normalize_speaker.

This is opt-in: the model only uses it when a timbre encoder is constructed and a
reference mel is provided. It does NOT replace the WavLM-SV path unless you choose
to; the two can coexist (WavLM-SV global identity + learnable timbre detail).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    """1D conv block over time: Conv -> GN -> SiLU, with residual when shapes match."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5, dilation: int = 1):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        # GroupNorm is batch-size independent (safe for tiny batches / variable T)
        self.norm = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.act = nn.SiLU()
        self.res = (in_ch == out_ch)

    def forward(self, x):  # x: (B, C, T)
        y = self.act(self.norm(self.conv(x)))
        return x + y if self.res else y


class AttentiveStatsPool(nn.Module):
    """Attentive statistics pooling (ECAPA-style): masked, returns mean+std concat.

    Produces a fixed-dim utterance vector that weights informative frames more than
    silence/padding — better than plain mean for timbre.
    """

    def __init__(self, channels: int, attn_dim: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, attn_dim, 1), nn.Tanh(),
            nn.Conv1d(attn_dim, channels, 1),
        )

    def forward(self, x, mask=None):  # x: (B, C, T), mask: (B, T) True=valid
        w = self.attn(x)  # (B, C, T)
        if mask is not None:
            w = w.masked_fill(~mask.unsqueeze(1), float("-inf"))
        w = torch.softmax(w, dim=-1)
        mean = (w * x).sum(dim=-1)                                  # (B, C)
        var = (w * (x - mean.unsqueeze(-1)) ** 2).sum(dim=-1)       # (B, C)
        std = var.clamp_min(1e-8).sqrt()
        return torch.cat([mean, std], dim=-1)                       # (B, 2C)


class TimbreEncoder(nn.Module):
    """Learnable timbre encoder from a reference mel-spectrogram.

    Args:
        n_mels: mel channels of the input reference (matches model mel, e.g. 100).
        timbre_dim: output global timbre dimension (e.g. 512 to match speaker_dim).
        hidden: internal channel width.
        n_blocks: number of conv blocks (receptive field over time).
        return_frame: if True, also return frame-level timbre (B, T, timbre_dim).
        l2_normalize: L2-normalize the global vector (stable fusion).
    """

    def __init__(
        self,
        n_mels: int = 100,
        timbre_dim: int = 512,
        hidden: int = 256,
        n_blocks: int = 5,
        return_frame: bool = False,
        l2_normalize: bool = False,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.timbre_dim = timbre_dim
        self.return_frame = return_frame
        self.l2_normalize = l2_normalize

        self.in_proj = nn.Conv1d(n_mels, hidden, 1)
        # dilations grow the temporal receptive field cheaply
        dilations = [1, 2, 3, 4, 1][:n_blocks] + [1] * max(0, n_blocks - 5)
        self.blocks = nn.ModuleList(
            [_ConvBlock(hidden, hidden, kernel=5, dilation=dilations[i]) for i in range(n_blocks)]
        )
        self.pool = AttentiveStatsPool(hidden)
        self.global_proj = nn.Sequential(
            nn.Linear(2 * hidden, timbre_dim), nn.SiLU(),
            nn.Linear(timbre_dim, timbre_dim),
        )
        if return_frame:
            self.frame_proj = nn.Conv1d(hidden, timbre_dim, 1)

    def forward(self, mel: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            mel: (B, T, n_mels) reference mel-spectrogram.
            mask: (B, T) bool, True for valid frames (optional).
        Returns:
            global_timbre: (B, timbre_dim)
            frame_timbre:  (B, T, timbre_dim) if return_frame else None
        """
        x = mel.transpose(1, 2)            # (B, n_mels, T)
        h = self.in_proj(x)               # (B, hidden, T)
        for blk in self.blocks:
            h = blk(h)                    # (B, hidden, T)

        pooled = self.pool(h, mask=mask)  # (B, 2*hidden)
        g = self.global_proj(pooled)      # (B, timbre_dim)
        if self.l2_normalize:
            g = F.normalize(g, dim=-1)

        frame = None
        if self.return_frame:
            frame = self.frame_proj(h).transpose(1, 2)  # (B, T, timbre_dim)
        return g, frame
