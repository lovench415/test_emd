from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from f5_tts.model.condition_types import ModelConditionBatch, ConditioningOutputs, ConditioningRuntime


def _expand_bool_mask(mask: torch.Tensor | bool | None, batch: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.zeros(batch, dtype=torch.bool, device=device)
    if isinstance(mask, bool):
        return torch.full((batch,), mask, dtype=torch.bool, device=device)
    return mask.to(device=device, dtype=torch.bool)


class ConditioningAdaLN(nn.Module):
    def __init__(self, cond_dim: int, model_dim: int, n_blocks: int, bottleneck_dim: int = 256):
        super().__init__()
        self.pre_proj = nn.Sequential(nn.Linear(cond_dim, model_dim), nn.SiLU())
        self.block_projs = nn.ModuleList([])
        for _ in range(n_blocks):
            down = nn.Linear(model_dim, bottleneck_dim)
            up = nn.Linear(bottleneck_dim, model_dim * 6)
            nn.init.zeros_(up.weight)
            nn.init.zeros_(up.bias)
            self.block_projs.append(nn.Sequential(down, nn.SiLU(), up))

    def forward(self, cond: torch.Tensor, present_mask: torch.Tensor | None = None) -> list[torch.Tensor]:
        if present_mask is None:
            present_mask = torch.ones(cond.shape[0], dtype=torch.bool, device=cond.device)
        present = present_mask.to(dtype=cond.dtype)
        h = self.pre_proj(cond) * present[:, None]
        return [proj(h) * present[:, None] for proj in self.block_projs]


class ConditioningCrossAttention(nn.Module):
    def __init__(self, model_dim: int, cond_dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads, self.dim_head = heads, dim_head
        self.norm = nn.LayerNorm(model_dim)
        self.cond_norm = nn.LayerNorm(cond_dim)
        self.to_q = nn.Linear(model_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cond_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cond_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, model_dim), nn.Dropout(dropout))
        # Gate=0.02 (not 0): upstream projections (frame_raw_proj, prosody_raw_proj)
        # are zero-init, so gate=0 blocks ALL gradient to them → they never learn →
        # emotion/prosody cross-attention stays dead forever.
        # 0.02 passes 2% of gradient → projections learn → cross-attn activates.
        self.gate = nn.Parameter(torch.tensor(0.02))

    def forward(self, x: torch.Tensor, cond: torch.Tensor, cond_mask: torch.Tensor | None = None, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        B = x.shape[0]
        if cond_mask is None:
            cond_mask = torch.ones(cond.shape[:2], dtype=torch.bool, device=cond.device)
        else:
            cond_mask = cond_mask.to(device=cond.device, dtype=torch.bool)
        row_has_cond = cond_mask.any(dim=1)
        if not row_has_cond.any():
            return x

        q = self.to_q(self.norm(x)).view(B, -1, self.heads, self.dim_head).transpose(1, 2)
        empty = ~row_has_cond
        if empty.any():
            cond_mask = cond_mask.clone()
            cond_mask[empty, 0] = True

        c = self.cond_norm(cond)
        k = self.to_k(c).view(B, -1, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(c).view(B, -1, self.heads, self.dim_head).transpose(1, 2)

        bias = torch.zeros((B, 1, 1, cond.shape[1]), device=cond.device, dtype=q.dtype)
        attn_mask = bias.masked_fill(~cond_mask[:, None, None, :], float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, -1, self.heads * self.dim_head)
        residual = torch.tanh(self.gate) * self.to_out(out)
        residual = residual * row_has_cond[:, None, None].to(residual.dtype)
        if x_mask is not None:
            residual = residual * x_mask[..., None].to(residual.dtype)
        return x + residual


class ConditioningEmbeddingAdd(nn.Module):
    def __init__(self, cond_dim: int, model_dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(cond_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, present_mask: torch.Tensor | None = None) -> torch.Tensor:
        if present_mask is None:
            present_mask = torch.ones(cond.shape[0], dtype=torch.bool, device=cond.device)
        out = self.proj(cond) * present_mask.to(cond.dtype)[:, None]
        return x + out.unsqueeze(1)


class _FusionCrossAttn(nn.Module):
    """Lightweight cross-attention for fusing two conditioning streams.

    Unlike ConditioningCrossAttention (designed for DiT blocks with model_dim queries),
    this handles arbitrary query/key dims for inter-conditioning fusion.
    """
    def __init__(self, q_dim: int, kv_dim: int, heads: int = 4, dim_head: int = 64):
        super().__init__()
        inner = heads * dim_head
        self.heads, self.dim_head = heads, dim_head
        self.norm_q = nn.LayerNorm(q_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.to_q = nn.Linear(q_dim, inner, bias=False)
        self.to_k = nn.Linear(kv_dim, inner, bias=False)
        self.to_v = nn.Linear(kv_dim, inner, bias=False)
        self.to_out = nn.Linear(inner, q_dim)
        # proj=zero so no random noise at start; gate=0.02 provides gradient for proj.
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)
        self.gate = nn.Parameter(torch.tensor(0.02))

    def forward(self, x, cond, cond_mask=None, x_mask=None):
        B = x.shape[0]
        if cond_mask is not None:
            cond_mask = cond_mask.to(device=x.device, dtype=torch.bool)
            row_has_cond = cond_mask.any(dim=1)  # (B,) per-row
            if not row_has_cond.any():
                return x
            # For rows with all-masked cond: set one token True to avoid
            # softmax(all -inf) = NaN. Residual is zeroed for these rows below.
            empty = ~row_has_cond
            if empty.any():
                cond_mask = cond_mask.clone()
                cond_mask[empty, 0] = True
        else:
            row_has_cond = None

        q = self.to_q(self.norm_q(x)).view(B, -1, self.heads, self.dim_head).transpose(1, 2)
        c = self.norm_kv(cond)
        k = self.to_k(c).view(B, -1, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(c).view(B, -1, self.heads, self.dim_head).transpose(1, 2)
        # Attention mask
        if cond_mask is not None:
            bias = torch.zeros((B, 1, 1, cond.shape[1]), device=x.device, dtype=q.dtype)
            bias = bias.masked_fill(~cond_mask[:, None, None, :], float("-inf"))
        else:
            bias = None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = out.transpose(1, 2).reshape(B, -1, self.heads * self.dim_head)
        residual = torch.tanh(self.gate) * self.to_out(out)
        # Zero residual for rows with no valid conditioning (prevents NaN leak)
        if row_has_cond is not None:
            residual = residual * row_has_cond[:, None, None].to(residual.dtype)
        if x_mask is not None:
            residual = residual * x_mask[..., None].to(residual.dtype)
        return x + residual


class EmoProsodyFusion(nn.Module):
    """Bidirectional cross-attention between emotion and prosody frame features.

    Learns the correlation between emotion (happy/sad/angry) and prosody
    (pitch contour, energy, speaking rate) before they enter the DiT blocks.

    Architecture:
        emotion_frame → attends to prosody_frame → enriched emotion
        prosody_frame → attends to emotion_frame → enriched prosody

    Both directions use zero-init gated residual → starts as identity,
    gradually learns to fuse information as training progresses.
    """

    def __init__(
        self, emotion_dim: int = 512, prosody_dim: int = 256,
        heads: int = 4, dim_head: int = 64,
    ):
        super().__init__()
        # Emotion queries prosody: "what pitch/energy/rhythm matches this emotion?"
        self.emo_attends_pros = _FusionCrossAttn(emotion_dim, prosody_dim, heads, dim_head)
        # Prosody queries emotion: "what emotion context enriches this pitch contour?"
        self.pros_attends_emo = _FusionCrossAttn(prosody_dim, emotion_dim, heads, dim_head)

    def forward(
        self,
        emotion_frame: torch.Tensor,     # (B, T_emo, emotion_dim)
        prosody_frame: torch.Tensor,      # (B, T_pros, prosody_dim)
        emotion_mask: torch.Tensor | None = None,  # (B, T_emo)
        prosody_mask: torch.Tensor | None = None,   # (B, T_pros)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (enriched_emotion, enriched_prosody)."""
        enriched_emo = self.emo_attends_pros(
            emotion_frame, prosody_frame,
            cond_mask=prosody_mask, x_mask=emotion_mask,
        )
        enriched_pros = self.pros_attends_emo(
            prosody_frame, emotion_frame,
            cond_mask=emotion_mask, x_mask=prosody_mask,
        )
        return enriched_emo, enriched_pros


class ConditioningAggregator(nn.Module):
    def __init__(
        self,
        speaker_dim: int = 512,
        emotion_dim: int = 512,
        model_dim: int = 1024,
        n_blocks: int = 22,
        cross_attn_layers: list[int] | None = None,
        cross_attn_heads: int = 8,
        cross_attn_dim_head: int = 64,
        use_adaln: bool = True,
        use_input_add: bool = True,
        use_cross_attn: bool = True,
        dropout: float = 0.0,
        speaker_raw_dim: int | None = None,
        emotion_raw_dim: int | None = None,
        adaln_bottleneck_dim: int = 256,
        # Prosody
        prosody_dim: int = 256,
        prosody_raw_dim: int | None = None,
        use_prosody_cross_attn: bool = True,
        prosody_cross_attn_layers: list[int] | None = None,
        prosody_cross_attn_heads: int = 8,
        prosody_cross_attn_dim_head: int = 64,
    ):
        super().__init__()
        self.speaker_dim = speaker_dim
        self.emotion_dim = emotion_dim
        self.prosody_dim = prosody_dim
        self.model_dim = model_dim
        self.use_adaln = use_adaln
        self.use_input_add = use_input_add
        self.use_cross_attn = use_cross_attn
        self.use_prosody_cross_attn = use_prosody_cross_attn

        self.speaker_raw_proj = nn.Linear(speaker_raw_dim, speaker_dim) if speaker_raw_dim else None
        self.emotion_raw_proj = nn.Linear(emotion_raw_dim, emotion_dim) if emotion_raw_dim else None
        self.frame_raw_proj = (
            nn.Sequential(nn.Linear(emotion_raw_dim, emotion_dim), nn.SiLU(), nn.Linear(emotion_dim, emotion_dim))
            if emotion_raw_dim else None
        )
        if self.frame_raw_proj is not None:
            nn.init.zeros_(self.frame_raw_proj[-1].weight)
            nn.init.zeros_(self.frame_raw_proj[-1].bias)

        # Prosody cross-attention projection: raw (5-dim) → prosody_dim for K/V
        self.prosody_raw_proj = (
            nn.Sequential(nn.Linear(prosody_raw_dim, prosody_dim), nn.SiLU(), nn.Linear(prosody_dim, prosody_dim))
            if prosody_raw_dim else None
        )
        if self.prosody_raw_proj is not None:
            nn.init.zeros_(self.prosody_raw_proj[-1].weight)
            nn.init.zeros_(self.prosody_raw_proj[-1].bias)

        # Prosody direct addition: raw → model_dim, added to hidden state every block.
        # Init: proj=zero (no noise), gate=0.02 (provides gradient for proj).
        # Kaiming proj + zero gate caused garbled speech: gate gets huge gradient
        # from random proj output → noise injected at 22 blocks before proj learns.
        self.prosody_direct_proj = (
            nn.Sequential(
                nn.Linear(prosody_raw_dim, model_dim),
                nn.SiLU(),
                nn.Linear(model_dim, model_dim),
            )
            if prosody_raw_dim else None
        )
        if self.prosody_direct_proj is not None:
            nn.init.zeros_(self.prosody_direct_proj[-1].weight)
            nn.init.zeros_(self.prosody_direct_proj[-1].bias)

        # Temporal smoothing for direct prosody path (depthwise conv, shared)
        self.prosody_temporal_smooth = (
            nn.Conv1d(model_dim, model_dim, kernel_size=5, padding=2, groups=model_dim)
            if prosody_raw_dim else None
        )

        # Per-block gate: small positive (0.02) so proj receives gradient via gate.
        if prosody_raw_dim:
            self.prosody_block_gates = nn.Parameter(torch.full((n_blocks,), 0.02))

        # Prosody global → AdaLN: 9-dim stats (mean/std pitch, voicing ratio,
        # energy stats, delta stats, utterance length, absolute pitch) projected
        # to model_dim and added to the fused speaker+emotion vector before AdaLN.
        PROSODY_GLOBAL_DIM = 9
        self.prosody_global_proj = (
            nn.Sequential(
                nn.Linear(PROSODY_GLOBAL_DIM, model_dim),
                nn.SiLU(),
                nn.Linear(model_dim, model_dim),
            )
            if prosody_raw_dim else None
        )
        if self.prosody_global_proj is not None:
            nn.init.zeros_(self.prosody_global_proj[-1].weight)
            nn.init.zeros_(self.prosody_global_proj[-1].bias)
            self.prosody_global_gate = nn.Parameter(torch.tensor(0.02))

        self.fusion = nn.Sequential(nn.Linear(speaker_dim + emotion_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim))
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

        if use_adaln:
            self.adaln_cond = ConditioningAdaLN(model_dim, model_dim, n_blocks, bottleneck_dim=adaln_bottleneck_dim)
        if use_input_add:
            self.input_add = ConditioningEmbeddingAdd(model_dim, model_dim)
        if use_cross_attn:
            if cross_attn_layers is None:
                cross_attn_layers = list(range(0, n_blocks, 4))
            self.cross_attn_layers = cross_attn_layers
            self.cross_attns = nn.ModuleDict({
                str(i): ConditioningCrossAttention(model_dim, emotion_dim, cross_attn_heads, cross_attn_dim_head, dropout)
                for i in cross_attn_layers
            })

        # Prosody cross-attention: interleaved with emotion (offset by 2 blocks)
        if use_prosody_cross_attn:
            if prosody_cross_attn_layers is None:
                prosody_cross_attn_layers = list(range(2, n_blocks, 4))  # [2,6,10,14,18]
            self.prosody_cross_attn_layers = prosody_cross_attn_layers
            self.prosody_cross_attns = nn.ModuleDict({
                str(i): ConditioningCrossAttention(
                    model_dim, prosody_dim, prosody_cross_attn_heads,
                    prosody_cross_attn_dim_head, dropout,
                )
                for i in prosody_cross_attn_layers
            })

        # Emo-prosody fusion: bidirectional cross-attention between emotion and prosody
        # frames before they enter the DiT. Only created when both streams exist.
        self.emo_prosody_fusion = (
            EmoProsodyFusion(emotion_dim, prosody_dim, heads=4, dim_head=64)
            if use_cross_attn and use_prosody_cross_attn else None
        )

    def _to_model_dtype(self, t: torch.Tensor) -> torch.Tensor:
        ref = self.fusion[0].weight
        return t.to(device=ref.device, dtype=ref.dtype)

    def project_speaker(self, emb: torch.Tensor | None) -> torch.Tensor | None:
        if emb is None:
            return None
        emb = self._to_model_dtype(emb)
        return self.speaker_raw_proj(emb) if self.speaker_raw_proj is not None else emb

    def project_emotion(self, emb: torch.Tensor | None) -> torch.Tensor | None:
        if emb is None:
            return None
        emb = self._to_model_dtype(emb)
        return self.emotion_raw_proj(emb) if self.emotion_raw_proj is not None else emb

    def project_frame(self, frame: torch.Tensor | None) -> torch.Tensor | None:
        if frame is None:
            return None
        frame = self._to_model_dtype(frame)
        return self.frame_raw_proj(frame) if self.frame_raw_proj is not None else frame

    def _align_prosody_raw_dim(self, frame: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        """Align prosody raw features to match projection layer's expected input dim.

        Handles mismatch between encoder output dim (e.g. 6 for new encoder with
        absolute pitch) and checkpoint's projection input dim (e.g. 5 for old checkpoint).
        """
        expected = proj[0].in_features if isinstance(proj, nn.Sequential) else proj.in_features
        actual = frame.shape[-1]
        if actual == expected:
            return frame
        if actual > expected:
            return frame[..., :expected]  # trim extra channels
        # actual < expected: pad with zeros (old 5-dim cache with new 6-dim model)
        return F.pad(frame, (0, expected - actual))

    def project_prosody(self, frame: torch.Tensor | None) -> torch.Tensor | None:
        """Project raw prosody → prosody_dim for cross-attention K/V."""
        if frame is None:
            return None
        frame = self._to_model_dtype(frame)
        if self.prosody_raw_proj is not None:
            frame = self._align_prosody_raw_dim(frame, self.prosody_raw_proj)
            return self.prosody_raw_proj(frame)
        return frame

    def project_prosody_direct(self, frame: torch.Tensor | None, mask: torch.Tensor | None = None) -> torch.Tensor | None:
        """Project raw prosody → model_dim for direct frame addition.

        Includes temporal smoothing (depthwise Conv1d) to ensure the pitch
        contour influence is smooth and doesn't introduce frame-level jitter.
        """
        if frame is None or self.prosody_direct_proj is None:
            return None
        frame = self._to_model_dtype(frame)
        frame = self._align_prosody_raw_dim(frame, self.prosody_direct_proj)
        out = self.prosody_direct_proj(frame)               # (B, T, model_dim)
        # Temporal smoothing
        out = out.permute(0, 2, 1)                           # (B, D, T)
        out = self.prosody_temporal_smooth(out)
        out = out.permute(0, 2, 1)                           # (B, T, D)
        # L2 normalize: bound magnitude to 1.0 per frame.
        # Without this, outlier frames have large norms that accumulate
        # across 22 gated additions → audible noise.
        # Cross-attn path already uses F.normalize; direct path must too.
        out = F.normalize(out, p=2, dim=-1)
        # Mask out invalid frames
        if mask is not None:
            out = out * mask[..., None].to(out.dtype)
        return out

    @staticmethod
    def compute_prosody_global(
        prosody_raw: torch.Tensor,
        prosody_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute 9-dim global prosody statistics from frame-level raw features.

        Features (all z-scored or ratio-scaled for stable MLP input):
            0: mean_log_f0        — average pitch (voiced, normalized)
            1: std_log_f0         — pitch variability / range
            2: voicing_ratio      — fraction of voiced frames (speaking density)
            3: mean_log_energy    — average loudness
            4: std_log_energy     — loudness variability
            5: mean_abs_delta_f0  — pitch movement speed (intonation complexity)
            6: mean_abs_delta_e   — energy movement speed
            7: log_total_frames   — utterance length context (log-scaled)
            8: mean_abs_f0        — absolute pitch level (preserves speaker register)

        Stat 8 uses channel 5 (log_f0_absolute) — the un-normalized absolute
        log pitch.  This is critical for voice cloning: without it, the model
        cannot distinguish a 100Hz male from a 250Hz female, because stats 0-1
        are computed from the z-normalized contour.
        """
        mask_f = prosody_mask.float()
        n_valid = mask_f.sum(dim=1).clamp(min=1)

        log_f0 = prosody_raw[..., 0]         # normalized
        voicing = prosody_raw[..., 1]
        log_energy = prosody_raw[..., 2]
        delta_f0 = prosody_raw[..., 3]
        delta_energy = prosody_raw[..., 4]

        # Channel 5 may not exist in old 5-dim caches → safe fallback
        if prosody_raw.shape[-1] > 5:
            log_f0_abs = prosody_raw[..., 5]  # absolute (un-normalized)
        else:
            log_f0_abs = log_f0               # fallback: use normalized

        voiced_f = ((voicing > 0.5) & prosody_mask).float()
        n_voiced = voiced_f.sum(dim=1).clamp(min=1)

        mean_log_f0 = (log_f0 * voiced_f).sum(dim=1) / n_voiced
        std_log_f0 = (((log_f0 - mean_log_f0.unsqueeze(1)) * voiced_f).pow(2).sum(dim=1) / n_voiced).sqrt()
        voicing_ratio = voiced_f.sum(dim=1) / n_valid
        mean_log_energy = (log_energy * mask_f).sum(dim=1) / n_valid
        std_log_energy = (((log_energy - mean_log_energy.unsqueeze(1)) * mask_f).pow(2).sum(dim=1) / n_valid).sqrt()
        mean_abs_delta_f0 = (delta_f0.abs() * mask_f).sum(dim=1) / n_valid
        mean_abs_delta_e = (delta_energy.abs() * mask_f).sum(dim=1) / n_valid
        log_total_frames = n_valid.log()

        # Absolute pitch: mean of un-normalized log_f0 over voiced frames
        # Typical values: ~4.6 (100Hz male) to ~5.5 (250Hz female)
        mean_abs_f0 = (log_f0_abs * voiced_f).sum(dim=1) / n_voiced

        return torch.stack([
            mean_log_f0, std_log_f0, voicing_ratio, mean_log_energy,
            std_log_energy, mean_abs_delta_f0, mean_abs_delta_e,
            log_total_frames, mean_abs_f0,
        ], dim=1)  # (B, 9)

    def project_model_conditions(
        self,
        *,
        speaker_raw: torch.Tensor | None = None,
        emotion_global_raw: torch.Tensor | None = None,
        emotion_frame_raw: torch.Tensor | None = None,
        prosody_raw: torch.Tensor | None = None,
        speaker_present: torch.Tensor | None = None,
        emotion_global_present: torch.Tensor | None = None,
        emotion_frame_mask: torch.Tensor | None = None,
        prosody_mask: torch.Tensor | None = None,
    ) -> ModelConditionBatch:
        # Compute prosody global stats from raw 5-dim frame features
        prosody_global = None
        if prosody_raw is not None and prosody_mask is not None and self.prosody_global_proj is not None:
            prosody_global = self.compute_prosody_global(prosody_raw, prosody_mask)

        return ModelConditionBatch(
            speaker=self.project_speaker(speaker_raw),
            emotion_global=self.project_emotion(emotion_global_raw),
            emotion_frame=self.project_frame(emotion_frame_raw),
            prosody_frame=self.project_prosody(prosody_raw),
            prosody_direct=self.project_prosody_direct(prosody_raw, mask=prosody_mask),
            prosody_global=prosody_global,
            speaker_present=speaker_present,
            emotion_global_present=emotion_global_present,
            emotion_frame_mask=emotion_frame_mask,
            prosody_mask=prosody_mask,
        )

    def _zeros(self, batch: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch, dim, device=device, dtype=dtype)

    def fuse(self, speaker_emb: torch.Tensor | None, emotion_emb: torch.Tensor | None, speaker_present: torch.Tensor | None = None, emotion_present: torch.Tensor | None = None, drop_speaker: torch.Tensor | bool | None = None, drop_emotion: torch.Tensor | bool | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        ref = speaker_emb if speaker_emb is not None else emotion_emb
        if ref is None:
            raise ValueError("fuse() requires at least one conditioning tensor")
        ref = self._to_model_dtype(ref)
        batch, device, dtype = ref.shape[0], ref.device, ref.dtype

        if speaker_emb is None:
            speaker_emb = self._zeros(batch, self.speaker_dim, device, dtype)
        else:
            speaker_emb = self._to_model_dtype(speaker_emb)
        if emotion_emb is None:
            emotion_emb = self._zeros(batch, self.emotion_dim, device, dtype)
        else:
            emotion_emb = self._to_model_dtype(emotion_emb)

        speaker_present = _expand_bool_mask(speaker_present, batch, device) if speaker_present is not None else torch.ones(batch, dtype=torch.bool, device=device)
        emotion_present = _expand_bool_mask(emotion_present, batch, device) if emotion_present is not None else torch.ones(batch, dtype=torch.bool, device=device)
        drop_speaker = _expand_bool_mask(drop_speaker, batch, device)
        drop_emotion = _expand_bool_mask(drop_emotion, batch, device)

        active_speaker = speaker_present & ~drop_speaker
        active_emotion = emotion_present & ~drop_emotion

        speaker_emb = speaker_emb * active_speaker[:, None].to(speaker_emb.dtype)
        emotion_emb = emotion_emb * active_emotion[:, None].to(emotion_emb.dtype)
        fused = self.fusion(torch.cat([speaker_emb, emotion_emb], dim=-1))
        fused_present = active_speaker | active_emotion
        fused = fused * fused_present[:, None].to(fused.dtype)
        return fused, fused_present

    def forward(self, conditions: ModelConditionBatch | None = None, drop_speaker: torch.Tensor | bool | None = None, drop_emotion: torch.Tensor | bool | None = None, drop_prosody: torch.Tensor | bool | None = None) -> ConditioningOutputs:
        if conditions is None:
            return ConditioningOutputs()

        speaker_emb = conditions.speaker
        emotion_global = conditions.emotion_global
        emotion_frame = conditions.emotion_frame
        prosody_frame = conditions.prosody_frame
        speaker_present = conditions.speaker_present
        emotion_present = conditions.emotion_global_present
        emotion_frame_mask = conditions.emotion_frame_mask
        prosody_mask = conditions.prosody_mask

        if (speaker_emb is None and emotion_global is None and emotion_frame is None
                and prosody_frame is None and conditions.prosody_direct is None
                and conditions.prosody_global is None):
            return ConditioningOutputs()

        ref = speaker_emb if speaker_emb is not None else emotion_global
        if ref is None:
            if emotion_frame is not None:
                ref = emotion_frame[:, 0]
            elif prosody_frame is not None:
                ref = prosody_frame[:, 0]
            elif conditions.prosody_direct is not None:
                ref = conditions.prosody_direct[:, 0]
            else:
                return ConditioningOutputs()  # nothing to condition on
        batch = ref.shape[0]

        fused = adaln = frame_cond = frame_mask = None
        prosody_cond = prosody_frame_mask = None
        fused_present = None
        if (speaker_emb is not None) or (emotion_global is not None):
            fused, fused_present = self.fuse(
                speaker_emb, emotion_global,
                speaker_present=speaker_present,
                emotion_present=emotion_present,
                drop_speaker=drop_speaker,
                drop_emotion=drop_emotion,
            )
            # Enrich fused vector with prosody global stats (pitch range,
            # speaking rate, voicing density) — tells AdaLN about overall
            # speaking style that frame-level signals cannot capture.
            if (self.prosody_global_proj is not None
                    and conditions.prosody_global is not None):
                pg = self._to_model_dtype(conditions.prosody_global)
                pg_emb = self.prosody_global_proj(pg)  # (B, model_dim)
                # Mask for dropped rows: zero both input AND output to prevent
                # Linear bias from leaking through: proj(zeros) = bias ≠ 0.
                if drop_prosody is not None:
                    dp_mask = _expand_bool_mask(drop_prosody, batch, pg_emb.device)
                    pg_emb = pg_emb * (~dp_mask[:, None]).to(pg_emb.dtype)
                fused = fused + torch.tanh(self.prosody_global_gate) * pg_emb
            if self.use_adaln:
                adaln = self.adaln_cond(fused, present_mask=fused_present)

        if self.use_cross_attn and emotion_frame is not None:
            if emotion_frame_mask is None:
                frame_mask = torch.ones(emotion_frame.shape[:2], dtype=torch.bool, device=emotion_frame.device)
            else:
                frame_mask = emotion_frame_mask.to(device=emotion_frame.device, dtype=torch.bool)
            if drop_emotion is not None:
                drop_emotion_mask = _expand_bool_mask(drop_emotion, batch, emotion_frame.device)
                frame_mask = frame_mask & (~drop_emotion_mask[:, None])

        # Prosody cross-attention conditioning (soft influence on specific blocks)
        if self.use_prosody_cross_attn and prosody_frame is not None:
            if prosody_mask is None:
                prosody_frame_mask = torch.ones(prosody_frame.shape[:2], dtype=torch.bool, device=prosody_frame.device)
            else:
                prosody_frame_mask = prosody_mask.to(device=prosody_frame.device, dtype=torch.bool)
            if drop_prosody is not None:
                drop_prosody_mask = _expand_bool_mask(drop_prosody, batch, prosody_frame.device)
                prosody_frame_mask = prosody_frame_mask & (~drop_prosody_mask[:, None])

        # ── Emo-prosody fusion: enrich both streams before they enter DiT ──
        # Applied AFTER drop masks so dropped signals don't leak through fusion.
        if (self.emo_prosody_fusion is not None
                and emotion_frame is not None and prosody_frame is not None
                and frame_mask is not None and prosody_frame_mask is not None
                and frame_mask.any() and prosody_frame_mask.any()):
            emotion_frame, prosody_frame = self.emo_prosody_fusion(
                emotion_frame, prosody_frame,
                emotion_mask=frame_mask, prosody_mask=prosody_frame_mask,
            )

        # L2-normalize and apply masks
        if frame_mask is not None and emotion_frame is not None:
            frame_cond = F.normalize(emotion_frame, p=2, dim=-1)
            frame_cond = frame_cond * frame_mask[..., None].to(frame_cond.dtype)

        if prosody_frame_mask is not None and prosody_frame is not None:
            prosody_cond = F.normalize(prosody_frame, p=2, dim=-1)
            prosody_cond = prosody_cond * prosody_frame_mask[..., None].to(prosody_cond.dtype)

        # Prosody direct addition (hard frame-to-frame influence on ALL blocks)
        # Already projected raw→model_dim + temporal-smoothed in project_model_conditions()
        prosody_direct = conditions.prosody_direct
        if prosody_direct is not None and drop_prosody is not None:
            drop_prosody_mask = _expand_bool_mask(drop_prosody, batch, prosody_direct.device)
            prosody_direct = prosody_direct * (~drop_prosody_mask[:, None, None]).to(prosody_direct.dtype)

        return ConditioningOutputs(
            fused_global=fused, fused_present_mask=fused_present,
            adaln=adaln,
            frame_cond=frame_cond, frame_mask=frame_mask,
            prosody_cond=prosody_cond, prosody_direct=prosody_direct,
            prosody_frame_mask=prosody_frame_mask,
        )

    def apply_input_conditioning(self, x: torch.Tensor, fused: torch.Tensor, present_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.input_add(x, fused, present_mask=present_mask) if self.use_input_add and fused is not None else x

    def apply_block_conditioning(self, x: torch.Tensor, block_idx: int, frame_cond: torch.Tensor | None, frame_mask: torch.Tensor | None = None, prosody_cond: torch.Tensor | None = None, prosody_mask: torch.Tensor | None = None, prosody_direct: torch.Tensor | None = None, x_mask: torch.Tensor | None = None, checkpoint_activations: bool = False) -> torch.Tensor:
        # Emotion cross-attention (blocks [0,4,8,12,16,20])
        key = str(block_idx)
        if self.use_cross_attn and key in self.cross_attns and frame_cond is not None:
            if checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(self.cross_attns[key], x, frame_cond, frame_mask, x_mask, use_reentrant=False)
            else:
                x = self.cross_attns[key](x, frame_cond, cond_mask=frame_mask, x_mask=x_mask)
        # Prosody cross-attention (blocks [2,6,10,14,18])
        if self.use_prosody_cross_attn and key in getattr(self, 'prosody_cross_attns', {}) and prosody_cond is not None:
            if checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(self.prosody_cross_attns[key], x, prosody_cond, prosody_mask, x_mask, use_reentrant=False)
            else:
                x = self.prosody_cross_attns[key](x, prosody_cond, cond_mask=prosody_mask, x_mask=x_mask)
        # Prosody direct addition — gated per block, scaled by 1/n_blocks.
        # Without scaling, the SAME prosody_direct tensor is added at all 22 blocks,
        # so total contribution = sum(gates) × ||pd|| ≈ 22 × gate_mean.
        # Cross-attention adds DIFFERENT vectors (via attention) at only 5-6 blocks.
        # This 22/5 asymmetry causes prosody_direct to dominate and suppress
        # other conditioning paths. Dividing by n_blocks bounds the total
        # contribution to ~gate_mean (comparable to cross-attn).
        if prosody_direct is not None and hasattr(self, 'prosody_block_gates'):
            n_blocks = self.prosody_block_gates.shape[0]
            gate = torch.tanh(self.prosody_block_gates[block_idx]) / n_blocks
            # Align lengths (prosody_direct may differ by 1-2 frames from x)
            pd = prosody_direct
            if pd.shape[1] < x.shape[1]:
                pd = F.pad(pd, (0, 0, 0, x.shape[1] - pd.shape[1]))
            elif pd.shape[1] > x.shape[1]:
                pd = pd[:, :x.shape[1], :]
            x = x + gate * pd
        return x

    def get_adaln_residual(self, block_idx: int, adaln_params: list[torch.Tensor] | None):
        if self.use_adaln and adaln_params is not None:
            return adaln_params[block_idx]
        return None

    def build_runtime(self, outputs: ConditioningOutputs | None = None) -> ConditioningRuntime:
        cond_out = outputs or ConditioningOutputs()

        if cond_out.fused_global is not None and cond_out.fused_present_mask is None:
            raise ValueError("conditioning_outputs.fused_present_mask must be provided when fused_global is set")

        def _apply_input(x: torch.Tensor) -> torch.Tensor:
            if cond_out.fused_global is None:
                return x
            return self.apply_input_conditioning(x, cond_out.fused_global, present_mask=cond_out.fused_present_mask)

        def _adaln_for_block(block_idx: int):
            return self.get_adaln_residual(block_idx, cond_out.adaln)

        def _apply_block(x: torch.Tensor, block_idx: int, x_mask: torch.Tensor | None = None, checkpoint_activations: bool = False) -> torch.Tensor:
            has_emotion = cond_out.frame_cond is not None
            has_prosody_ca = cond_out.prosody_cond is not None
            has_prosody_direct = cond_out.prosody_direct is not None
            if not has_emotion and not has_prosody_ca and not has_prosody_direct:
                return x
            return self.apply_block_conditioning(
                x, block_idx,
                cond_out.frame_cond, frame_mask=cond_out.frame_mask,
                prosody_cond=cond_out.prosody_cond, prosody_mask=cond_out.prosody_frame_mask,
                prosody_direct=cond_out.prosody_direct,
                x_mask=x_mask, checkpoint_activations=checkpoint_activations,
            )

        return ConditioningRuntime(
            apply_input=_apply_input,
            adaln_for_block=_adaln_for_block,
            apply_block=_apply_block,
        )
