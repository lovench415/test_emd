from __future__ import annotations

import torch
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    AdaLayerNorm, AdaLayerNorm_Final, Attention, AttnProcessor,
    ConvPositionEmbedding, FeedForward, TimestepEmbedding,
)
from f5_tts.model.backbones.dit import TextEmbedding
from f5_tts.model.condition_types import ConditioningRuntime
from f5_tts.model.conditioning import ConditioningAggregator


def _as_batch_bool(mask, batch: int, device: torch.device):
    if mask is None:
        return torch.zeros(batch, dtype=torch.bool, device=device)
    if isinstance(mask, bool):
        return torch.full((batch,), mask, dtype=torch.bool, device=device)
    return mask.to(device=device, dtype=torch.bool)


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim: int, text_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x, cond, text_embed, drop_audio_cond=False, audio_mask=None):
        if drop_audio_cond is not None:
            if isinstance(drop_audio_cond, bool):
                if drop_audio_cond:
                    cond = torch.zeros_like(cond)
            else:
                cond = torch.where(drop_audio_cond[:, None, None], torch.zeros_like(cond), cond)
        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        return self.conv_pos_embed(x, mask=audio_mask) + x


class EnhancedDiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1, qk_norm=None, pe_attn_head=None, attn_backend="torch", attn_mask_enabled=True):
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=AttnProcessor(pe_attn_head=pe_attn_head, attn_backend=attn_backend, attn_mask_enabled=attn_mask_enabled),
            dim=dim, heads=heads, dim_head=dim_head, dropout=dropout, qk_norm=qk_norm,
        )
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None, cond_adaln_residual=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        if cond_adaln_residual is not None:
            c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = torch.chunk(cond_adaln_residual, 6, dim=1)
            c_scale_msa = c_scale_msa.clamp(-0.5, 0.5)
            c_scale_mlp = c_scale_mlp.clamp(-0.5, 0.5)
            c_shift_msa = c_shift_msa.clamp(-10.0, 10.0)
            c_shift_mlp = c_shift_mlp.clamp(-10.0, 10.0)
            c_gate_msa = c_gate_msa.clamp(-5.0, 5.0)
            c_gate_mlp = c_gate_mlp.clamp(-5.0, 5.0)
            norm = norm * (1 + c_scale_msa[:, None]) + c_shift_msa[:, None]
            gate_msa = gate_msa + c_gate_msa
            shift_mlp = shift_mlp + c_shift_mlp
            scale_mlp = scale_mlp + c_scale_mlp
            gate_mlp = gate_mlp + c_gate_mlp
        attn_out = self.attn(x=norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_out
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        x = x + gate_mlp.unsqueeze(1) * self.ff(norm)
        return x


class EnhancedDiT(nn.Module):
    def __init__(
        self, *, dim, depth=8, heads=8, dim_head=64, dropout=0.1, ff_mult=4,
        mel_dim=100, text_num_embeds=256, text_dim=None, text_mask_padding=True,
        text_embedding_average_upsampling=False, qk_norm=None, conv_layers=0,
        pe_attn_head=None, attn_backend="torch", attn_mask_enabled=False,
        long_skip_connection=False, checkpoint_activations=False,
        speaker_emb_dim: int = 512, emotion_emb_dim: int = 512,
        speaker_raw_dim: int | None = None, emotion_raw_dim: int | None = None,
        cross_attn_layers: list[int] | None = None,
        cross_attn_heads: int = 8, cross_attn_dim_head: int = 64,
        use_adaln_cond: bool = True, use_input_add_cond: bool = True,
        use_cross_attn_cond: bool = True, adaln_bottleneck_dim: int = 256,
        # Prosody
        prosody_dim: int = 256,
        prosody_raw_dim: int | None = None,
        use_prosody_cross_attn: bool = True,
        prosody_cross_attn_layers: list[int] | None = None,
        prosody_cross_attn_heads: int = 8,
        prosody_cross_attn_dim_head: int = 64,
        prosody_direct_layers: list[int] | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.checkpoint_activations = checkpoint_activations
        if text_dim is None:
            text_dim = mel_dim
        self.time_embed = TimestepEmbedding(dim)
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, mask_padding=text_mask_padding, average_upsampling=text_embedding_average_upsampling, conv_layers=conv_layers)
        self.text_cond, self.text_uncond = None, None
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.transformer_blocks = nn.ModuleList([
            EnhancedDiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout, qk_norm=qk_norm, pe_attn_head=pe_attn_head, attn_backend=attn_backend, attn_mask_enabled=attn_mask_enabled)
            for _ in range(depth)
        ])
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)
        self.cond_aggregator = ConditioningAggregator(
            speaker_dim=speaker_emb_dim, emotion_dim=emotion_emb_dim, model_dim=dim, n_blocks=depth,
            cross_attn_layers=cross_attn_layers, cross_attn_heads=cross_attn_heads, cross_attn_dim_head=cross_attn_dim_head,
            use_adaln=use_adaln_cond, use_input_add=use_input_add_cond, use_cross_attn=use_cross_attn_cond, dropout=dropout,
            speaker_raw_dim=speaker_raw_dim, emotion_raw_dim=emotion_raw_dim, adaln_bottleneck_dim=adaln_bottleneck_dim,
            prosody_dim=prosody_dim, prosody_raw_dim=prosody_raw_dim,
            use_prosody_cross_attn=use_prosody_cross_attn,
            prosody_cross_attn_layers=prosody_cross_attn_layers,
            prosody_cross_attn_heads=prosody_cross_attn_heads,
            prosody_cross_attn_dim_head=prosody_cross_attn_dim_head,
            prosody_direct_layers=prosody_direct_layers,
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

    def _get_input_embed(self, x, cond, text, drop_audio_cond=False, drop_text=False, cache=True, audio_mask=None):
        batch = x.shape[0]
        drop_audio_cond = _as_batch_bool(drop_audio_cond, batch, x.device)
        drop_text = _as_batch_bool(drop_text, batch, x.device)
        use_cache = bool(cache and (drop_text.all() or (~drop_text).all()))
        if self.text_uncond is None or self.text_cond is None or not use_cache:
            seq_len = audio_mask.sum(dim=1) if audio_mask is not None else x.shape[1]
            text_embed = self.text_embed(text, seq_len=seq_len, drop_text=False)
            text_embed = torch.where(drop_text[:, None, None], torch.zeros_like(text_embed), text_embed)
            if use_cache:
                if drop_text.all():
                    self.text_uncond = text_embed
                elif (~drop_text).all():
                    self.text_cond = text_embed
        if use_cache and drop_text.all() and self.text_uncond is not None:
            text_embed = self.text_uncond
        elif use_cache and (~drop_text).all() and self.text_cond is not None:
            text_embed = self.text_cond
        return self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond, audio_mask=audio_mask)

    def clear_cache(self):
        self.text_cond = self.text_uncond = None

    def forward(self, x, cond, text, time, mask=None, drop_audio_cond=False, drop_text=False, cache=False, conditioning_runtime: ConditioningRuntime | None = None):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)
        t = self.time_embed(time)

        drop_audio_mask = _as_batch_bool(drop_audio_cond, batch, x.device)
        drop_text_mask = _as_batch_bool(drop_text, batch, x.device)

        runtime = conditioning_runtime

        x = self._get_input_embed(x, cond, text, drop_audio_cond=drop_audio_mask, drop_text=drop_text_mask, cache=cache, audio_mask=mask)

        if runtime is not None:
            x = runtime.apply_input(x)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        residual = x if self.long_skip_connection is not None else None
        for i, block in enumerate(self.transformer_blocks):
            adaln_res = runtime.adaln_for_block(i) if runtime is not None else None
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(block, x, t, mask, rope, adaln_res, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope, cond_adaln_residual=adaln_res)
            if runtime is not None:
                x = runtime.apply_block(x, i, x_mask=mask, checkpoint_activations=self.checkpoint_activations)

        if residual is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))
        return self.proj_out(self.norm_out(x, t))
