"""
Duration Predictor — predicts mel frame count for target text given prosody.

Architecture:
    prosody_global (7: per-channel masked mean of raw prosody) + speaker_emb (512) + text_len (1)
    NOTE: text length unit is CHARACTERS (Unicode codepoints), not bytes.
    Byte units biased duration across scripts (Cyrillic=2 bytes/char →
    a model trained on Cyrillic halves the duration of Latin text).
    Models trained before this change used bytes — retrain to use chars.
    → MLP → speaking_rate (frames/byte)
    → gen_duration = text_byte_len × speaking_rate

Training:
    Target: actual mel_frames / text_byte_len = ground truth rate
    Loss: L1(predicted_rate, gt_rate) weighted by log(text_byte_len)

Inference:
    rate = predictor(ref_prosody_global, ref_speaker_emb)
    gen_duration = gen_text_bytes × rate

The predictor learns per-speaker, per-style speaking rates:
    fast angry speech → low rate (fewer frames/byte)
    slow contemplative speech → high rate (more frames/byte)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DurationPredictor(nn.Module):
    """Predict speaking rate from prosody + speaker features.

    Input:
        prosody_global: (B, 7) — pooled prosody features
        speaker_emb:    (B, speaker_dim) — speaker embedding (optional)
        text_byte_len:  (B,) — text length in bytes (for rate → frames conversion)

    Output:
        predicted_frames: (B,) — predicted mel frame count
        predicted_rate:   (B,) — predicted frames/byte speaking rate
    """

    def __init__(
        self,
        prosody_global_dim: int = 7,
        speaker_dim: int = 512,
        hidden_dim: int = 256,
        use_speaker: bool = True,
        speaker_dropout: float = 0.0,    # dropout on the raw speaker vector before
                                         # it enters the predictor. The speaker
                                         # embedding is ~98% of the input width, so
                                         # the predictor easily memorises per-speaker
                                         # rates instead of learning rate-from-prosody
                                         # → it extrapolates badly to unseen (e.g.
                                         # cross-language) speakers and over-predicts
                                         # duration. 0.2-0.5 recommended when
                                         # use_speaker=True.
        speaker_bottleneck: int = 0,     # if >0, compress speaker_dim → this width
                                         # before concatenation, shrinking the
                                         # speaker share of the input and limiting
                                         # memorisation capacity (e.g. 32).
        input_dropout: float = 0.0,      # dropout on the full input vector.
    ):
        super().__init__()
        self.use_speaker = use_speaker
        self.speaker_dim = speaker_dim
        self.speaker_dropout = nn.Dropout(speaker_dropout) if speaker_dropout > 0 else None
        self.speaker_proj = (
            nn.Linear(speaker_dim, speaker_bottleneck)
            if (use_speaker and speaker_bottleneck > 0) else None
        )
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else None

        input_dim = prosody_global_dim + 1  # +1 for log(text_byte_len)
        if use_speaker:
            input_dim += speaker_bottleneck if speaker_bottleneck > 0 else speaker_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # rate must be positive
        )

        # Initialize to output ~4.0 frames/byte (reasonable default for 24kHz/256hop)
        # Softplus(x) ≈ x for x >> 0, so bias the last linear to ~4.0
        with torch.no_grad():
            nn.init.zeros_(self.net[-2].weight)
            self.net[-2].bias.fill_(1.4)  # softplus(1.4) ≈ 1.74, but we'll scale

        # Learnable base rate (log-scale, initialized to ~4 frames/byte)
        self.log_base_rate = nn.Parameter(torch.tensor(1.386))  # exp(1.386) ≈ 4.0

    def compute_prosody_global(
        self,
        prosody_raw: torch.Tensor | None,
        prosody_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pool frame-level prosody → global (B, prosody_dim).

        Args:
            prosody_raw: (B, T, 6) — frame features (5 original + absolute pitch)
            prosody_mask: (B, T) — valid frame mask

        Returns:
            (B, 7) — mean of each channel over valid frames
        """
        EXPECTED_DIM = 7  # must match prosody_global_dim passed to __init__

        if prosody_raw is None:
            return torch.zeros(1, EXPECTED_DIM, device=self._device())

        if prosody_mask is not None:
            mask = prosody_mask.to(prosody_raw.dtype).unsqueeze(-1)  # (B, T, 1)
            denom = mask.sum(dim=1).clamp_min(1)                     # (B, 1)
            result = (prosody_raw * mask).sum(dim=1) / denom          # (B, D)
        else:
            result = prosody_raw.mean(dim=1)                          # (B, D)

        # Old 5-dim caches lack channel 5 (absolute pitch) → pad to expected dim
        if result.shape[-1] < EXPECTED_DIM:
            result = F.pad(result, (0, EXPECTED_DIM - result.shape[-1]))
        return result

    def _device(self):
        return self.log_base_rate.device

    def forward(
        self,
        prosody_global: torch.Tensor,
        text_byte_len: torch.Tensor,
        speaker_emb: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            prosody_global: (B, 7)
            text_byte_len:  (B,) int or float
            speaker_emb:    (B, speaker_dim) optional

        Returns:
            dict with:
                "frames": (B,) predicted mel frame count
                "rate":   (B,) predicted frames/byte
        """
        B = prosody_global.shape[0]
        device = prosody_global.device
        dtype = prosody_global.dtype

        # Normalize text length
        text_len_f = text_byte_len.to(dtype=dtype, device=device)
        log_text_len = torch.log(text_len_f.clamp_min(1.0)).unsqueeze(-1)  # (B, 1)

        # Build input
        parts = [prosody_global, log_text_len]
        if self.use_speaker and speaker_emb is not None:
            spk = speaker_emb.to(dtype=dtype)
            if self.speaker_dropout is not None:
                spk = self.speaker_dropout(spk)
            if self.speaker_proj is not None:
                spk = self.speaker_proj(spk.to(next(self.speaker_proj.parameters()).dtype)).to(dtype)
            parts.append(spk)
        elif self.use_speaker:
            width = self.speaker_proj.out_features if self.speaker_proj is not None else self.speaker_dim
            parts.append(torch.zeros(B, width, device=device, dtype=dtype))

        x = torch.cat(parts, dim=-1)  # (B, input_dim)
        if self.input_dropout is not None:
            x = self.input_dropout(x)

        # AMP guard: prosody_global/speaker_emb may arrive as fp16 (from cached
        # half-precision embeddings or autocast), but this head's Linear weights
        # are fp32. F.linear requires matching dtypes. Run the whole head in the
        # net's weight dtype and keep the result in that dtype (compute_loss takes
        # log + L1 vs an fp32 target; predict_duration only calls .item()).
        # (autocast does not cast module weights, only ops, so this needs an
        # explicit cast.)
        net_dtype = next(self.net.parameters()).dtype
        x = x.to(net_dtype)
        text_len_net = text_len_f.to(net_dtype)

        # Predict rate adjustment (multiplicative correction on base rate)
        adjustment = self.net(x).squeeze(-1)  # (B,) positive via Softplus

        # Final rate = base_rate × adjustment
        base_rate = torch.exp(self.log_base_rate.to(net_dtype))  # scalar, ~4.0
        rate = base_rate * adjustment               # (B,)

        # Duration = rate × text_byte_len
        frames = rate * text_len_net

        # Keep rate/frames in fp32 (net_dtype): they are tiny scalars, and
        # compute_loss takes log() + L1 against an fp32 target — returning fp16
        # here would reintroduce a dtype mismatch in the loss. predict_duration
        # only calls .item(), so fp32 is fine there too.
        return {"frames": frames, "rate": rate}

    def compute_loss(
        self,
        prosody_raw: torch.Tensor | None,
        prosody_mask: torch.Tensor | None,
        text_byte_lens: torch.Tensor,
        target_mel_frames: torch.Tensor,
        speaker_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute duration prediction loss.

        Args:
            prosody_raw:      (B, T, 5) or None
            prosody_mask:     (B, T) or None
            text_byte_lens:   (B,) text lengths in bytes
            target_mel_frames: (B,) actual mel frame counts
            speaker_emb:      (B, speaker_dim) or None

        Returns:
            Scalar loss (L1 on log-rate, robust to length variation)
        """
        prosody_global = self.compute_prosody_global(prosody_raw, prosody_mask)
        pred = self.forward(prosody_global, text_byte_lens, speaker_emb)

        # Target rate
        target_rate = target_mel_frames.float() / text_byte_lens.float().clamp_min(1.0)

        # L1 on log-rates (scale-invariant, robust to outliers)
        log_pred = torch.log(pred["rate"].clamp_min(0.1))
        log_target = torch.log(target_rate.clamp_min(0.1))
        return F.l1_loss(log_pred, log_target)

    @torch.no_grad()
    def predict_duration(
        self,
        prosody_raw: torch.Tensor | None,
        prosody_mask: torch.Tensor | None,
        text_byte_len: int,
        speaker_emb: torch.Tensor | None = None,
        min_frames: int = 10,
        max_frames: int = 65536,
    ) -> int:
        """Convenience method for inference — returns integer frame count."""
        prosody_global = self.compute_prosody_global(prosody_raw, prosody_mask)
        text_len_t = torch.tensor([text_byte_len], device=self._device(), dtype=torch.float)
        pred = self.forward(prosody_global, text_len_t, speaker_emb)
        frames = int(pred["frames"].item())
        return max(min_frames, min(frames, max_frames))
