"""
Enhanced DiT backbone for F5-TTS — adds speaker/emotion conditioning.

Changes from original DiT:
  1. EnhancedDiTBlock with AdaLN conditioning residuals
  2. Cross-attention with frame-level emotion at selected layers
  3. Embedding addition at input level
  4. ConditioningAggregator manages all conditioning paths

All new modules are zero-initialized → loads original weights via strict=False.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    AdaLayerNorm,
    AdaLayerNorm_Final,
    Attention,
    AttnProcessor,
    ConvPositionEmbedding,
    FeedForward,
    TimestepEmbedding,
)

# Import TextEmbedding from the original DiT — no need to duplicate 90 lines
from f5_tts.model.backbones.dit import TextEmbedding

from f5_tts.model.conditioning import ConditioningAggregator


# ── Input Embedding ───────────────────────────────────────────────────

class InputEmbedding(nn.Module):
    """Projects [noised_mel ‖ cond_mel ‖ text] → model dim."""

    def __init__(self, mel_dim: int, text_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x, cond, text_embed, drop_audio_cond=False, audio_mask=None):
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        return self.conv_pos_embed(x, mask=audio_mask) + x


# ── Enhanced DiTBlock ─────────────────────────────────────────────────

class EnhancedDiTBlock(nn.Module):
    """
    DiTBlock + optional AdaLN conditioning residual.

    When cond_adaln_residual is provided, it applies additional affine
    transforms on top of the timestep-based modulation (zero-init at start).
    """

    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1,
                 qk_norm=None, pe_attn_head=None, attn_backend="torch",
                 attn_mask_enabled=True):
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=AttnProcessor(
                pe_attn_head=pe_attn_head, attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
            ),
            dim=dim, heads=heads, dim_head=dim_head,
            dropout=dropout, qk_norm=qk_norm,
        )
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None, cond_adaln_residual=None):
        # AdaLayerNorm: shift/scale already applied to norm
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # Conditioning residual — additional affine on norm + gate adjustments
        if cond_adaln_residual is not None:
            (c_shift_msa, c_scale_msa, c_gate_msa,
             c_shift_mlp, c_scale_mlp, c_gate_mlp) = torch.chunk(cond_adaln_residual, 6, dim=1)
            # Clamp scale residuals: prevents signal destruction (→ -1 zeros out)
            # or explosion (→ +1 doubles). Range [-0.5, 0.5] keeps signal in [0.5×, 1.5×].
            c_scale_msa = c_scale_msa.clamp(-0.5, 0.5)
            c_scale_mlp = c_scale_mlp.clamp(-0.5, 0.5)
            # Clamp shift/gate: prevents runaway values after many training steps.
            # Shifts offset normed signal (~unit scale); gates modulate residual magnitude.
            c_shift_msa = c_shift_msa.clamp(-10.0, 10.0)
            c_shift_mlp = c_shift_mlp.clamp(-10.0, 10.0)
            c_gate_msa = c_gate_msa.clamp(-5.0, 5.0)
            c_gate_mlp = c_gate_mlp.clamp(-5.0, 5.0)
            norm = norm * (1 + c_scale_msa[:, None]) + c_shift_msa[:, None]
            gate_msa = gate_msa + c_gate_msa
            shift_mlp = shift_mlp + c_shift_mlp
            scale_mlp = scale_mlp + c_scale_mlp
            gate_mlp = gate_mlp + c_gate_mlp

        # Self-attention
        attn_out = self.attn(x=norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # Feed-forward
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        x = x + gate_mlp.unsqueeze(1) * self.ff(norm)
        return x


# ── Enhanced DiT ──────────────────────────────────────────────────────

class EnhancedDiT(nn.Module):
    """
    F5-TTS DiT backbone with speaker/emotion conditioning.
    Backward-compatible: loads original weights via strict=False.
    """

    def __init__(
        self, *, dim, depth=8, heads=8, dim_head=64, dropout=0.1, ff_mult=4,
        mel_dim=100, text_num_embeds=256, text_dim=None, text_mask_padding=True,
        text_embedding_average_upsampling=False, qk_norm=None, conv_layers=0,
        pe_attn_head=None, attn_backend="torch", attn_mask_enabled=False,
        long_skip_connection=False, checkpoint_activations=False,
        # Conditioning
        speaker_emb_dim: int = 512, emotion_emb_dim: int = 512,
        speaker_raw_dim: int | None = None, emotion_raw_dim: int | None = None,
        cross_attn_layers: list[int] | None = None,
        cross_attn_heads: int = 8, cross_attn_dim_head: int = 64,
        use_adaln_cond: bool = True, use_input_add_cond: bool = True,
        use_cross_attn_cond: bool = True,
        adaln_bottleneck_dim: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.checkpoint_activations = checkpoint_activations

        if text_dim is None:
            text_dim = mel_dim

        self.time_embed = TimestepEmbedding(dim)
        self.text_embed = TextEmbedding(
            text_num_embeds, text_dim, mask_padding=text_mask_padding,
            average_upsampling=text_embedding_average_upsampling,
            conv_layers=conv_layers,
        )
        self.text_cond, self.text_uncond = None, None
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.transformer_blocks = nn.ModuleList([
            EnhancedDiTBlock(
                dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult,
                dropout=dropout, qk_norm=qk_norm, pe_attn_head=pe_attn_head,
                attn_backend=attn_backend, attn_mask_enabled=attn_mask_enabled,
            )
            for _ in range(depth)
        ])

        self.long_skip_connection = (
            nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        )
        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

        self.cond_aggregator = ConditioningAggregator(
            speaker_dim=speaker_emb_dim, emotion_dim=emotion_emb_dim,
            model_dim=dim, n_blocks=depth,
            cross_attn_layers=cross_attn_layers,
            cross_attn_heads=cross_attn_heads,
            cross_attn_dim_head=cross_attn_dim_head,
            use_adaln=use_adaln_cond, use_input_add=use_input_add_cond,
            use_cross_attn=use_cross_attn_cond, dropout=dropout,
            speaker_raw_dim=speaker_raw_dim, emotion_raw_dim=emotion_raw_dim,
            adaln_bottleneck_dim=adaln_bottleneck_dim,
        )
        self._init_weights()

    def _init_weights(self):
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    # ── Text & input embedding (with caching for CFG) ─────────────────

    def _get_input_embed(self, x, cond, text, drop_audio_cond=False,
                         drop_text=False, cache=True, audio_mask=None):
        if self.text_uncond is None or self.text_cond is None or not cache:
            seq_len = audio_mask.sum(dim=1) if audio_mask is not None else x.shape[1]
            text_embed = self.text_embed(text, seq_len=seq_len, drop_text=drop_text)
            # CFG-correct: ensure unconditional branch has truly zero text embedding.
            if drop_text:
                text_embed = torch.zeros_like(text_embed)
            if cache:
                if drop_text:
                    self.text_uncond = text_embed
                else:
                    self.text_cond = text_embed

        if cache:
            text_embed = self.text_uncond if drop_text else self.text_cond

        return self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond, audio_mask=audio_mask)

    def clear_cache(self):
        self.text_cond = self.text_uncond = None

    # ── Conditioning helpers ──────────────────────────────────────────

    def _compute_conditioning(self, batch, x, speaker_emb, emotion_global,
                              emotion_frame, drop_speaker, drop_emotion):
        """Compute all conditioning outputs or return Nones."""
        has_cond = (speaker_emb is not None) or (emotion_global is not None) or (emotion_frame is not None)
        if not has_cond:
            return None, None, None, False

        # Fill missing with zeros
        if speaker_emb is None:
            speaker_emb = torch.zeros(batch, self.cond_aggregator.speaker_dim,
                                      device=x.device, dtype=x.dtype)
        if emotion_global is None:
            emotion_global = torch.zeros(batch, self.cond_aggregator.emotion_dim,
                                         device=x.device, dtype=x.dtype)

        out = self.cond_aggregator(
            speaker_emb, emotion_global, emotion_frame, drop_speaker, drop_emotion,
        )
        return out.get("fused"), out.get("adaln"), out.get("frame_cond"), True

    # ── Forward ───────────────────────────────────────────────────────

    def forward(
        self, x, cond, text, time, mask=None,
        drop_audio_cond=False, drop_text=False,
        cfg_infer=False, cache=False,
        speaker_emb=None, emotion_global=None, emotion_frame=None,
        drop_speaker=False, drop_emotion=False,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)

        # Conditioning
        fused, adaln_params, frame_cond, has_cond = self._compute_conditioning(
            batch, x, speaker_emb, emotion_global, emotion_frame,
            drop_speaker, drop_emotion,
        )

        # CFG inference: pack cond + uncond
        if cfg_infer:
            x_c = self._get_input_embed(x, cond, text, cache=cache, audio_mask=mask)
            x_u = self._get_input_embed(x, cond, text, drop_audio_cond=True,
                                        drop_text=True, cache=cache, audio_mask=mask)
            x = torch.cat((x_c, x_u), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None

            if has_cond:
                fused_u = self.cond_aggregator.fuse(
                    speaker_emb, emotion_global, drop_speaker=True, drop_emotion=True,
                )
                fused = torch.cat((fused, fused_u), dim=0)
                if adaln_params is not None:
                    adaln_u = self.cond_aggregator.adaln_cond(fused_u)
                    adaln_params = [
                        torch.cat((pc, pu), dim=0)
                        for pc, pu in zip(adaln_params, adaln_u)
                    ]
                if frame_cond is not None:
                    frame_cond = torch.cat((frame_cond, torch.zeros_like(frame_cond)), dim=0)
        else:
            x = self._get_input_embed(
                x, cond, text, drop_audio_cond=drop_audio_cond,
                drop_text=drop_text, cache=cache, audio_mask=mask,
            )

        # Input-level conditioning
        if has_cond and fused is not None:
            x = self.cond_aggregator.apply_input_conditioning(x, fused)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        residual = x if self.long_skip_connection is not None else None

        # Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            adaln_res = (
                self.cond_aggregator.get_adaln_residual(i, adaln_params)
                if adaln_params is not None else None
            )

            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, t, mask, rope, adaln_res, use_reentrant=False,
                )
            else:
                x = block(x, t, mask=mask, rope=rope, cond_adaln_residual=adaln_res)

            if has_cond:
                x = self.cond_aggregator.apply_block_conditioning(
                    x, i, frame_cond, mask,
                    checkpoint_activations=self.checkpoint_activations,
                )

        if residual is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        return self.proj_out(self.norm_out(x, t))
