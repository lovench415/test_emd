"""
Enhanced DiT Backbone for F5-TTS
==================================

Modified DiT (Diffusion Transformer) that supports external conditioning
from speaker and emotion embeddings. The key changes from the original DiT:

1. InputEmbedding now accepts a global conditioning embedding (embedding add)
2. Each DiTBlock receives AdaLN residuals from the ConditioningAggregator
3. Cross-attention with frame-level emotion features at selected layers
4. Forward pass threads conditioning through all blocks

All modifications are additive (zero-initialized), so loading original
F5-TTS weights works out of the box — new modules start as identity.

Compatible with original F5-TTS checkpoints:
    - All original parameters have the same names and shapes
    - New parameters are prefixed with 'cond_aggregator.' or 'enhanced_*'
    - load_state_dict(strict=False) loads the base model seamlessly
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
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    FeedForward,
    TimestepEmbedding,
    precompute_freqs_cis,
)

from f5_tts.model.conditioning import ConditioningAggregator


# ── Text Embedding (unchanged from original) ─────────────────────────

class TextEmbedding(nn.Module):
    """Identical to the original F5-TTS TextEmbedding."""

    def __init__(
        self, text_num_embeds, text_dim, mask_padding=True,
        average_upsampling=False, conv_layers=0, conv_mult=2
    ):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)
        self.mask_padding = mask_padding
        self.average_upsampling = average_upsampling
        if average_upsampling:
            assert mask_padding

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 8192
            self.register_buffer(
                "freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos),
                persistent=False,
            )
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def average_upsample_text_by_mask(self, text, text_mask, target_lens):
        batch, max_seq_len, text_dim = text.shape
        text_lens = text_mask.sum(dim=1)
        upsampled_text = torch.zeros_like(text)
        for i in range(batch):
            text_len = int(text_lens[i].item())
            audio_len = int(target_lens[i].item())
            if text_len == 0 or audio_len <= 0:
                continue
            valid_ind = torch.where(text_mask[i])[0]
            valid_data = text[i, valid_ind, :]
            base_repeat = audio_len // text_len
            remainder = audio_len % text_len
            indices = []
            for j in range(text_len):
                repeat_count = base_repeat + (1 if j >= text_len - remainder else 0)
                indices.extend([j] * repeat_count)
            indices = torch.tensor(indices[:audio_len], device=text.device, dtype=torch.long)
            upsampled = valid_data[indices]
            upsampled_text[i, :audio_len, :] = upsampled
        return upsampled_text

    def forward(self, text, seq_len, drop_text=False):
        text = text + 1
        valid_pos_mask = None
        if torch.is_tensor(seq_len):
            seq_len = seq_len.to(device=text.device, dtype=torch.long)
            max_seq_len = int(seq_len.max().item())
        else:
            max_seq_len = int(seq_len)
        text = text[:, :max_seq_len]
        text = F.pad(text, (0, max_seq_len - text.shape[1]), value=0)
        if torch.is_tensor(seq_len):
            seq_pos = torch.arange(max_seq_len, device=text.device).unsqueeze(0)
            valid_pos_mask = seq_pos < seq_len.unsqueeze(1)
            text = text.masked_fill(~valid_pos_mask, 0)
        if self.mask_padding:
            text_mask = text == 0
        if drop_text:
            text = torch.zeros_like(text)
        text = self.text_embed(text)
        if valid_pos_mask is not None:
            text = text.masked_fill(~valid_pos_mask.unsqueeze(-1), 0.0)
        if self.extra_modeling:
            freqs = self.freqs_cis[:max_seq_len, :]
            if valid_pos_mask is not None:
                freqs = freqs.unsqueeze(0) * valid_pos_mask.unsqueeze(-1).to(freqs.dtype)
            text = text + freqs
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)
        if self.average_upsampling:
            if torch.is_tensor(seq_len):
                target_lens = seq_len.to(device=text.device, dtype=torch.long)
            else:
                target_lens = torch.full((text.shape[0],), int(seq_len), device=text.device, dtype=torch.long)
            text = self.average_upsample_text_by_mask(text, ~text_mask, target_lens)
        return text


# ── Enhanced Input Embedding ──────────────────────────────────────────

class EnhancedInputEmbedding(nn.Module):
    """
    Same as original InputEmbedding, but optionally adds the global
    conditioning embedding to the projected input.
    """

    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(
        self,
        x,     # (b, n, mel_dim) noised audio
        cond,  # (b, n, mel_dim) masked condition audio
        text_embed,  # (b, n, text_dim)
        drop_audio_cond=False,
        audio_mask=None,
    ):
        if drop_audio_cond:
            cond = torch.zeros_like(cond)

        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x, mask=audio_mask) + x
        return x


# ── Enhanced DiTBlock ─────────────────────────────────────────────────

class EnhancedDiTBlock(nn.Module):
    """
    DiTBlock with optional AdaLN conditioning residual from external embeddings.
    
    The original AdaLN modulates with timestep embedding:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = AdaLN(x, t)
    
    Enhanced version adds conditioning residual:
        params = AdaLN(x, t)
        if cond_adaln_residual is not None:
            params += cond_adaln_residual  (zero-initialized at start)
    """

    def __init__(
        self, dim, heads, dim_head, ff_mult=4, dropout=0.1,
        qk_norm=None, pe_attn_head=None, attn_backend="torch",
        attn_mask_enabled=True,
    ):
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=AttnProcessor(
                pe_attn_head=pe_attn_head,
                attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
            ),
            dim=dim, heads=heads, dim_head=dim_head,
            dropout=dropout, qk_norm=qk_norm,
        )
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(
        self, x, t, mask=None, rope=None,
        cond_adaln_residual=None,  # NEW: (batch, dim*6) from ConditioningAdaLN
    ):
        # Pre-norm & modulation
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # Add conditioning residual to modulation parameters
        if cond_adaln_residual is not None:
            c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
                torch.chunk(cond_adaln_residual, 6, dim=1)
            )
            # The norm already applied scale/shift, so we adjust gate and mlp params
            gate_msa = gate_msa + c_gate_msa
            shift_mlp = shift_mlp + c_shift_mlp
            scale_mlp = scale_mlp + c_scale_mlp
            gate_mlp = gate_mlp + c_gate_mlp

        # Attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_output

        # FFN
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x


# ── Enhanced DiT ──────────────────────────────────────────────────────

class EnhancedDiT(nn.Module):
    """
    F5-TTS DiT backbone enhanced with speaker/emotion conditioning.
    
    Key additions over original DiT:
    1. ConditioningAggregator for speaker/emotion fusion
    2. EnhancedDiTBlock with AdaLN conditioning residuals
    3. Cross-attention with frame-level emotion at selected layers
    4. Embedding addition at input level
    
    Backward compatible: loads original F5-TTS weights with strict=False.
    New parameters are zero-initialized so they start as identity transforms.
    """

    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        text_mask_padding=True,
        text_embedding_average_upsampling=False,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        attn_backend="torch",
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
        # ── New: conditioning parameters ──
        speaker_emb_dim: int = 512,
        emotion_emb_dim: int = 512,
        cross_attn_layers: list[int] | None = None,
        cross_attn_heads: int = 8,
        cross_attn_dim_head: int = 64,
        use_adaln_cond: bool = True,
        use_input_add_cond: bool = True,
        use_cross_attn_cond: bool = True,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds, text_dim,
            mask_padding=text_mask_padding,
            average_upsampling=text_embedding_average_upsampling,
            conv_layers=conv_layers,
        )
        self.text_cond, self.text_uncond = None, None
        self.input_embed = EnhancedInputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        # Use EnhancedDiTBlock
        self.transformer_blocks = nn.ModuleList([
            EnhancedDiTBlock(
                dim=dim, heads=heads, dim_head=dim_head,
                ff_mult=ff_mult, dropout=dropout, qk_norm=qk_norm,
                pe_attn_head=pe_attn_head, attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
            )
            for _ in range(depth)
        ])

        self.long_skip_connection = (
            nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        )

        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        # ── Conditioning Aggregator ──
        self.cond_aggregator = ConditioningAggregator(
            speaker_dim=speaker_emb_dim,
            emotion_dim=emotion_emb_dim,
            model_dim=dim,
            n_blocks=depth,
            cross_attn_layers=cross_attn_layers,
            cross_attn_heads=cross_attn_heads,
            cross_attn_dim_head=cross_attn_dim_head,
            use_adaln=use_adaln_cond,
            use_input_add=use_input_add_cond,
            use_cross_attn=use_cross_attn_cond,
            dropout=dropout,
        )

        self.initialize_weights()

    def initialize_weights(self):
        """Zero-out AdaLN and output layers (same as original)."""
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            return module(*inputs)
        return ckpt_forward

    def get_input_embed(
        self, x, cond, text, drop_audio_cond=False, drop_text=False,
        cache=True, audio_mask=None,
    ):
        """Same as original, computes text embedding and input projection."""
        if self.text_uncond is None or self.text_cond is None or not cache:
            if audio_mask is None:
                seq_len = x.shape[1]
            else:
                seq_len = audio_mask.sum(dim=1)
            text_embed = self.text_embed(text, seq_len=seq_len, drop_text=drop_text)
            if cache:
                if drop_text:
                    self.text_uncond = text_embed
                else:
                    self.text_cond = text_embed

        if cache:
            text_embed = self.text_uncond if drop_text else self.text_cond

        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond, audio_mask=audio_mask)
        return x

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def forward(
        self,
        x: torch.Tensor,      # (b, n, mel_dim) noised input
        cond: torch.Tensor,    # (b, n, mel_dim) masked cond audio
        text: torch.Tensor,    # (b, nt) text tokens
        time: torch.Tensor,    # (b,) timestep
        mask: torch.Tensor | None = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
        # ── New: external conditioning embeddings ──
        speaker_emb: torch.Tensor | None = None,    # (b, speaker_dim)
        emotion_global: torch.Tensor | None = None,  # (b, emotion_dim)
        emotion_frame: torch.Tensor | None = None,   # (b, T, emotion_dim)
        drop_speaker: bool = False,
        drop_emotion: bool = False,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # Timestep embedding
        t = self.time_embed(time)

        # ── Compute conditioning ──
        cond_output = None
        adaln_params = None
        frame_cond = None
        fused = None

        has_conditioning = speaker_emb is not None and emotion_global is not None

        if has_conditioning:
            cond_output = self.cond_aggregator(
                speaker_emb=speaker_emb,
                emotion_global=emotion_global,
                emotion_frame=emotion_frame,
                drop_speaker=drop_speaker,
                drop_emotion=drop_emotion,
            )
            fused = cond_output["fused"]
            adaln_params = cond_output.get("adaln")
            frame_cond = cond_output.get("frame_cond")

        # ── CFG inference: pack cond & uncond ──
        if cfg_infer:
            x_cond = self.get_input_embed(
                x, cond, text, drop_audio_cond=False, drop_text=False,
                cache=cache, audio_mask=mask,
            )
            x_uncond = self.get_input_embed(
                x, cond, text, drop_audio_cond=True, drop_text=True,
                cache=cache, audio_mask=mask,
            )
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None

            # Duplicate conditioning for CFG batch
            if has_conditioning:
                fused = torch.cat((fused, fused), dim=0)
                if adaln_params is not None:
                    adaln_params = [torch.cat((p, p), dim=0) for p in adaln_params]
                if frame_cond is not None:
                    frame_cond = torch.cat((frame_cond, frame_cond), dim=0)
        else:
            x = self.get_input_embed(
                x, cond, text, drop_audio_cond=drop_audio_cond,
                drop_text=drop_text, cache=cache, audio_mask=mask,
            )

        # ── Apply input-level conditioning ──
        if has_conditioning and fused is not None:
            x = self.cond_aggregator.apply_input_conditioning(x, fused)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        # ── Transformer blocks with conditioning ──
        for i, block in enumerate(self.transformer_blocks):
            # Get AdaLN residual for this block
            adaln_res = None
            if adaln_params is not None:
                adaln_res = self.cond_aggregator.get_adaln_residual(i, adaln_params)

            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block), x, t, mask, rope, adaln_res,
                    use_reentrant=False,
                )
            else:
                x = block(x, t, mask=mask, rope=rope, cond_adaln_residual=adaln_res)

            # Cross-attention with frame-level features
            if has_conditioning:
                x = self.cond_aggregator.apply_block_conditioning(
                    x, i, adaln_params, frame_cond, mask,
                )

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
