"""
Conditioning Module for F5-TTS Enhanced
=========================================

Injects speaker and emotion embeddings into the DiT backbone via:

1. **AdaLN Modulation** — global embeddings modulate AdaLayerNorm params
   alongside timestep.  Low overhead, strong for global style.
2. **Cross-Attention** — frame-level emotion features attend to DiT hidden
   states.  Best for time-varying, fine-grained emotion control.
3. **Embedding Addition** — projected embeddings added to input sequence.
   Simplest method; works well for speaker identity transfer.
4. **CFG-Based Emotion Guidance** — inference-time only (no modules here).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# 1. AdaLN Modulation
# ══════════════════════════════════════════════════════════════════════


class ConditioningAdaLN(nn.Module):
    """
    Produces per-block AdaLN residuals from the fused conditioning vector.
    Output is *added* to the timestep-based modulation parameters.
    Zero-initialized so conditioning starts as identity.
    """

    def __init__(self, cond_dim: int, model_dim: int, n_blocks: int, bottleneck_dim: int = 256,):
        super().__init__()
        self.pre_proj = nn.Sequential(nn.Linear(cond_dim, model_dim), nn.SiLU())

        # Per-block bottleneck:  model_dim -> bottleneck -> (6 * model_dim)
        self.block_projs = nn.ModuleList([])

        for _ in range(n_blocks):
            down = nn.Linear(model_dim, bottleneck_dim)
            up = nn.Linear(bottleneck_dim, model_dim * 6)
      # zero-init LAST layer to preserve identity at start (warm-start safe)
            nn.init.zeros_(up.weight)
            nn.init.zeros_(up.bias)
            self.block_projs.append(nn.Sequential(down, nn.SiLU(), up))

    def forward(self, cond: torch.Tensor) -> list[torch.Tensor]:
        """(B, cond_dim) → list of n_blocks × (B, model_dim*6).

        IMPORTANT (CFG correctness):
        When cond is all zeros (unconditional branch), the AdaLN residual must be EXACTLY zero.
        We therefore gate the outputs by a presence mask computed from the *input* cond, so
        biases inside pre_proj / block MLPs cannot leak a non-zero signal into the uncond path.
        """
        # True where there is any conditioning signal.
        present = (cond.abs().sum(dim=-1) > 1e-8).to(dtype=cond.dtype)  # (B,)
        h = self.pre_proj(cond)
        h = h * present[:, None]  # kill bias leak from pre_proj for uncond rows

        outs = [proj(h) for proj in self.block_projs]
        # Also gate final outputs to prevent bias leakage from trained block_projs.
        outs = [o * present[:, None] for o in outs]
        return outs


# ══════════════════════════════════════════════════════════════════════
# 2. Cross-Attention (frame-level emotion)
# ══════════════════════════════════════════════════════════════════════


class ConditioningCrossAttention(nn.Module):
    """
    Cross-attention: Q from DiT hidden states, K/V from frame-level emotion.
    Gated residual (gate=0 at init) for stable warm-start.
    """

    def __init__(self, model_dim: int, cond_dim: int, heads: int = 8,
                 dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads, self.dim_head = heads, dim_head

        self.norm = nn.LayerNorm(model_dim)
        self.cond_norm = nn.LayerNorm(cond_dim)
        self.to_q = nn.Linear(model_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cond_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cond_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, model_dim), nn.Dropout(dropout))
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        B = x.shape[0]

        # Identify rows where ALL conditioning is zero (CFG uncond branch).
        # LayerNorm(zeros) produces non-zero K/V (bias + normalization),
        # so we must gate the output to avoid leaking into the uncond path.
        cond_mask = (cond.abs().sum(dim=-1) > 1e-8)      # (B, T) True=valid
        row_has_cond = cond_mask.any(dim=1)               # (B,)

        # Fast path: if NO row has conditioning, skip entirely
        if not row_has_cond.any():
            return x

        q = self.to_q(self.norm(x)).view(B, -1, self.heads, self.dim_head).transpose(1, 2)

        # Guard against all-masked rows: force position 0 valid to prevent
        # SDPA from producing NaN (all -inf attention weights).
        empty = ~row_has_cond
        if empty.any():
            cond_mask = cond_mask.clone()
            cond_mask[empty, 0] = True

        # Compute K/V AFTER mask correction
        c = self.cond_norm(cond)
        k = self.to_k(c).view(B, -1, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(c).view(B, -1, self.heads, self.dim_head).transpose(1, 2)

        # Robust SDPA masking: use float bias (0 for valid, -inf for pad)
        bias = torch.zeros((B, 1, 1, cond.shape[1]), device=cond.device, dtype=q.dtype)
        attn_mask = bias.masked_fill(~cond_mask[:, None, None, :], float("-inf"))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, -1, self.heads * self.dim_head)
        residual = torch.tanh(self.gate) * self.to_out(out)

        # Zero out residual for rows that had no valid conditioning frames,
        # preventing LayerNorm bias from leaking into the CFG uncond branch.
        if empty.any():
            residual = residual * row_has_cond[:, None, None].to(residual.dtype)

        return x + residual


# ══════════════════════════════════════════════════════════════════════
# 3. Embedding Addition (input-level)
# ══════════════════════════════════════════════════════════════════════


class ConditioningEmbeddingAdd(nn.Module):
    """Broadcast-adds a projected global embedding to every frame."""

    def __init__(self, cond_dim: int, model_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim),
        )
        # CRITICAL: must be exactly zero so output is identity at init.
        # std=0.01 injects ~0.32 magnitude noise per frame from step 0.
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Gate by input presence: proj(zeros) ≠ 0 due to first Linear's bias
        # leaking through SiLU → trained second Linear.  Without this gate,
        # the CFG unconditional branch receives a ghost conditioning signal.
        present = (cond.abs().sum(dim=-1) > 1e-8).to(dtype=cond.dtype)  # (B,)
        out = self.proj(cond) * present[:, None]
        return x + out.unsqueeze(1)


# ══════════════════════════════════════════════════════════════════════
# 4. Aggregator — combines all conditioning strategies
# ══════════════════════════════════════════════════════════════════════


class ConditioningAggregator(nn.Module):
    """
    Aggregates speaker + emotion into a unified conditioning signal
    and provides interfaces for AdaLN, cross-attention, and input-add.

    Handles raw→projected dimension adaptation via optional linear layers,
    so callers can pass either projected or raw embeddings.
    """

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
    ):
        super().__init__()
        self.speaker_dim = speaker_dim
        self.emotion_dim = emotion_dim
        self.model_dim = model_dim
        self.use_adaln = use_adaln
        self.use_input_add = use_input_add
        self.use_cross_attn = use_cross_attn
        self._speaker_raw_dim = speaker_raw_dim   # for _maybe_project dispatch
        self._emotion_raw_dim = emotion_raw_dim

        # Raw→target projections (ALWAYS created to ensure consistent
        # representation whether embeddings come from cached raw features
        # or from online encoder extraction with a separate MLP)
        self.speaker_raw_proj = (
            nn.Linear(speaker_raw_dim, speaker_dim)
            if speaker_raw_dim else None
        )
        self.emotion_raw_proj = (
            nn.Linear(emotion_raw_dim, emotion_dim)
            if emotion_raw_dim else None
        )
        self.frame_raw_proj = (
            nn.Sequential(nn.Linear(emotion_raw_dim, emotion_dim), nn.SiLU(),
                          nn.Linear(emotion_dim, emotion_dim))
            if emotion_raw_dim else None
        )
        # Zero-init frame projection output for identity start
        if self.frame_raw_proj is not None:
            nn.init.zeros_(self.frame_raw_proj[-1].weight)
            nn.init.zeros_(self.frame_raw_proj[-1].bias)

        # Fusion: [speaker ‖ emotion] → model_dim
        self.fusion = nn.Sequential(
            nn.Linear(speaker_dim + emotion_dim, model_dim), nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        # Zero-init: no conditioning signal before training
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

        # Injection modules
        if use_adaln:
            self.adaln_cond = ConditioningAdaLN(model_dim, model_dim, n_blocks, bottleneck_dim = adaln_bottleneck_dim,)
        if use_input_add:
            self.input_add = ConditioningEmbeddingAdd(model_dim, model_dim)
        if use_cross_attn:
            if cross_attn_layers is None:
                cross_attn_layers = list(range(0, n_blocks, 4))
            self.cross_attn_layers = cross_attn_layers
            self.cross_attns = nn.ModuleDict({
                str(i): ConditioningCrossAttention(
                    model_dim, emotion_dim, cross_attn_heads, cross_attn_dim_head, dropout,
                )
                for i in cross_attn_layers
            })

    # ── Helpers ───────────────────────────────────────────────────────

    def _to_model_dtype(self, t: torch.Tensor) -> torch.Tensor:
        """Cast tensor to the fusion layer's dtype and device."""
        ref = self.fusion[0].weight
        return t.to(device=ref.device, dtype=ref.dtype)

    def _maybe_project_speaker(self, emb: torch.Tensor) -> torch.Tensor:
        # Project raw→target when raw_proj exists AND input matches raw dimension.
        # This correctly handles WavLM (raw=512, target=512) by checking raw_dim
        # instead of target_dim, so the projection is never bypassed.
        if self.speaker_raw_proj is not None and emb.shape[-1] == self._speaker_raw_dim:
            return self.speaker_raw_proj(emb)
        return emb

    def _maybe_project_emotion(self, emb: torch.Tensor) -> torch.Tensor:
        if self.emotion_raw_proj is not None and emb.shape[-1] == self._emotion_raw_dim:
            return self.emotion_raw_proj(emb)
        return emb


    def _maybe_project_frame(self, frame: torch.Tensor) -> torch.Tensor:
        if self.frame_raw_proj is not None and self._emotion_raw_dim and frame.shape[-1] == self._emotion_raw_dim:
            return self.frame_raw_proj(frame)
        return frame

    # ── Public API ────────────────────────────────────────────────────

    def fuse(
        self,
        speaker_emb: torch.Tensor,
        emotion_emb: torch.Tensor,
        drop_speaker: bool = False,
        drop_emotion: bool = False,
    ) -> torch.Tensor:
        """Fuse speaker + emotion → (B, model_dim)."""
        # Drop BOTH conditions → true unconditional (zero fused vector)
        if drop_speaker and drop_emotion:
            speaker_emb = self._to_model_dtype(speaker_emb)
            ref_shape = (speaker_emb.shape[0], self.model_dim)
            return torch.zeros(ref_shape, device=speaker_emb.device, dtype=speaker_emb.dtype)

        # Cast FIRST, then project — avoids float32 matmul through bfloat16 weights.
        speaker_emb = self._maybe_project_speaker(self._to_model_dtype(speaker_emb))
        emotion_emb = self._maybe_project_emotion(self._to_model_dtype(emotion_emb))
        if drop_speaker:
            speaker_emb = torch.zeros_like(speaker_emb)
        if drop_emotion:
            emotion_emb = torch.zeros_like(emotion_emb)
        return self.fusion(torch.cat([speaker_emb, emotion_emb], dim=-1))

    def forward(
        self,
        speaker_emb: torch.Tensor,
        emotion_global: torch.Tensor,
        emotion_frame: torch.Tensor | None = None,
        drop_speaker: bool = False,
        drop_emotion: bool = False,
    ) -> dict:
        """
        Returns:
            fused:      (B, model_dim)
            adaln:      list[22 × (B, model_dim*6)]  (if use_adaln)
            frame_cond: (B, T, emotion_dim)           (if use_cross_attn)
        """
        fused = self.fuse(speaker_emb, emotion_global, drop_speaker, drop_emotion)
        result: dict = {"fused": fused}

        if self.use_adaln:
            result["adaln"] = self.adaln_cond(fused)

        if self.use_cross_attn and emotion_frame is not None:
            emotion_frame = self._to_model_dtype(emotion_frame)

            # Build padding mask BEFORE any projection (projection may introduce bias on zero padded frames).
            frame_pad_mask = (emotion_frame.abs().sum(dim=-1) > 1e-8)  # (B, T), True=valid

            result["frame_cond"] = self._maybe_project_frame(emotion_frame)

            # Enforce exact zeros on padded positions AFTER projection (prevents mask breakage downstream).
            result["frame_cond"] = result["frame_cond"] * frame_pad_mask[..., None].to(result["frame_cond"].dtype)

            # Zero out AFTER projection to avoid bias leak through Linear layers
            if drop_emotion:
                result["frame_cond"] = torch.zeros_like(result["frame_cond"])

        return result

    def apply_input_conditioning(self, x: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
        return self.input_add(x, fused) if self.use_input_add else x

    def apply_block_conditioning(
        self, x: torch.Tensor, block_idx: int,
        frame_cond: torch.Tensor | None, mask: torch.Tensor | None = None,
        checkpoint_activations: bool = False,
    ) -> torch.Tensor:
        key = str(block_idx)
        if self.use_cross_attn and key in self.cross_attns and frame_cond is not None:
            if checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(
                    self.cross_attns[key], x, frame_cond, mask, use_reentrant=False,
                )
            else:
                x = self.cross_attns[key](x, frame_cond, mask)
        return x

    def get_adaln_residual(self, block_idx: int, adaln_params: list[torch.Tensor]):
        if self.use_adaln and adaln_params is not None:
            return adaln_params[block_idx]
        return None
