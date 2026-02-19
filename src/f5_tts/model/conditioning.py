"""
Conditioning Module for F5-TTS Enhanced
=========================================

Implements multiple strategies for injecting speaker and emotion embeddings
into the DiT denoising backbone. This is the core architectural innovation.

Three conditioning strategies (can be combined):

1. **AdaLN Modulation** (EmoSteer-TTS inspired):
   Global embeddings modulate AdaLayerNorm parameters alongside timestep.
   Low overhead, strong for global style (speaker identity, overall emotion).

2. **Cross-Attention Injection** (TTS-CtrlNet inspired):
   Frame-level emotion features attend to DiT hidden states via cross-attention.
   Best for time-varying, fine-grained emotion control.

3. **Embedding Addition** (ece-tts inspired):
   Projected embeddings are added directly to the input sequence.
   Simplest method; works well for speaker identity transfer.

4. **CFG-Based Emotion Guidance** (F5-TTS-Emotional-CFG):
   At inference time, emotion embeddings steer the classifier-free guidance
   vector. Zero training cost for this component; purely inference-time.

Design principle: We inject conditioning at multiple scales:
    - Input level:  embedding addition to input projection
    - Block level:  AdaLN modulation in each DiT block
    - Attention:    cross-attention for frame-level features
    - Output level: CFG guidance (inference only)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from f5_tts.model.modules import FeedForward


# ==================================================================
# 1. AdaLN-based Conditioning (global embeddings -> scale/shift/gate)
# ==================================================================


class ConditioningAdaLN(nn.Module):
    """
    Converts global embeddings (speaker + emotion) into AdaLN modulation
    parameters that are added to the timestep-based modulation.
    
    For each DiT block, the original AdaLN produces 6 params from timestep:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    
    This module produces an additive residual to those 6 params from the
    speaker/emotion embeddings, so that:
        final_params = timestep_params + cond_params
    
    This is analogous to how EmoSteer-TTS steers generation with emotion
    vectors, but generalized to both speaker and emotion.
    """

    def __init__(self, cond_dim: int, model_dim: int, n_blocks: int):
        super().__init__()
        self.n_blocks = n_blocks
        
        # Shared initial projection
        self.pre_proj = nn.Sequential(
            nn.Linear(cond_dim, model_dim),
            nn.SiLU(),
        )
        
        # Per-block projections: cond -> 6 modulation params
        self.block_projs = nn.ModuleList([
            nn.Linear(model_dim, model_dim * 6)
            for _ in range(n_blocks)
        ])
        
        # Initialize to zero so conditioning starts as identity
        for proj in self.block_projs:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, cond: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            cond: (batch, cond_dim) global conditioning embedding
            
        Returns:
            list of (batch, model_dim * 6) tensors, one per DiT block
        """
        h = self.pre_proj(cond)
        return [proj(h) for proj in self.block_projs]


# ==================================================================
# 2. Cross-Attention for Frame-Level Emotion Features
# ==================================================================


class ConditioningCrossAttention(nn.Module):
    """
    Injects frame-level emotion features via cross-attention.
    
    At each selected DiT block, we add a cross-attention layer where:
        - Query: DiT hidden states (noised audio features)
        - Key/Value: frame-level emotion features
    
    This allows the model to attend to different parts of the emotion
    trajectory at each position, enabling nuanced time-varying control.
    
    Inspired by TTS-CtrlNet and Time-Varying Emotion Control.
    """

    def __init__(
        self,
        model_dim: int,
        cond_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        self.norm = nn.LayerNorm(model_dim)
        self.cond_norm = nn.LayerNorm(cond_dim)

        self.to_q = nn.Linear(model_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cond_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cond_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, model_dim),
            nn.Dropout(dropout),
        )

        # Gating: learnable scalar, initialized to 0 for stable warm-start
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self, 
        x: torch.Tensor,        # (batch, n, model_dim) - DiT hidden states
        cond: torch.Tensor,      # (batch, T_cond, cond_dim) - frame-level features
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Cross-attention from x (query) to cond (key/value).
        Returns x + gate * cross_attn_output.
        """
        batch = x.shape[0]
        
        x_norm = self.norm(x)
        cond_norm = self.cond_norm(cond)

        q = self.to_q(x_norm)
        k = self.to_k(cond_norm)
        v = self.to_v(cond_norm)

        # Reshape for multi-head
        q = q.view(batch, -1, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(batch, -1, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(batch, -1, self.heads, self.dim_head).transpose(1, 2)

        # Scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn_out = attn_out.transpose(1, 2).reshape(batch, -1, self.heads * self.dim_head)

        out = self.to_out(attn_out)

        # Gated residual
        return x + torch.tanh(self.gate) * out


# ==================================================================
# 3. Embedding Addition (input-level injection)
# ==================================================================


class ConditioningEmbeddingAdd(nn.Module):
    """
    Projects global speaker/emotion embeddings and adds them to the
    input embedding sequence. Simple but effective for identity transfer.
    
    The embedding is broadcast across the time dimension and added
    to every frame, giving the model a constant "style context".
    
    Inspired by ece-tts's emotion condition embedding approach.
    """

    def __init__(self, cond_dim: int, model_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        # Initialize near-zero for stability
        nn.init.normal_(self.proj[-1].weight, std=0.01)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (batch, n, model_dim) input sequence
            cond: (batch, cond_dim) global embedding
            
        Returns:
            x + projected_cond: (batch, n, model_dim)
        """
        cond_proj = self.proj(cond)  # (batch, model_dim)
        return x + cond_proj.unsqueeze(1)  # broadcast over time


# ==================================================================
# 4. Combined Conditioning Aggregator
# ==================================================================


class ConditioningAggregator(nn.Module):
    """
    Aggregates speaker and emotion embeddings into a unified conditioning
    signal. Supports multiple fusion strategies.
    
    The aggregator:
    1. Concatenates speaker_emb and emotion_global_emb
    2. Projects to a unified conditioning vector
    3. Provides interfaces for all injection methods
    
    It also manages emotion_frame features for cross-attention.
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
    ):
        super().__init__()

        self.speaker_dim = speaker_dim
        self.emotion_dim = emotion_dim
        self.model_dim = model_dim
        self.use_adaln = use_adaln
        self.use_input_add = use_input_add
        self.use_cross_attn = use_cross_attn

        # Unified conditioning dimension
        self.cond_dim = speaker_dim + emotion_dim

        # Fusion MLP: [speaker; emotion] -> unified
        self.fusion = nn.Sequential(
            nn.Linear(self.cond_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        # Drop conditions for CFG (independent speaker/emotion dropout)
        self.speaker_drop_prob = 0.1
        self.emotion_drop_prob = 0.1

        # --- Injection modules ---

        if use_adaln:
            self.adaln_cond = ConditioningAdaLN(
                cond_dim=model_dim, model_dim=model_dim, n_blocks=n_blocks
            )

        if use_input_add:
            self.input_add_cond = ConditioningEmbeddingAdd(
                cond_dim=model_dim, model_dim=model_dim
            )

        if use_cross_attn:
            # Cross-attention at selected layers (default: every 4th block)
            if cross_attn_layers is None:
                cross_attn_layers = list(range(0, n_blocks, 4))
            self.cross_attn_layers = cross_attn_layers
            
            self.cross_attns = nn.ModuleDict({
                str(i): ConditioningCrossAttention(
                    model_dim=model_dim,
                    cond_dim=emotion_dim,
                    heads=cross_attn_heads,
                    dim_head=cross_attn_dim_head,
                    dropout=dropout,
                )
                for i in cross_attn_layers
            })

    def get_fused_embedding(
        self,
        speaker_emb: torch.Tensor,   # (batch, speaker_dim)
        emotion_emb: torch.Tensor,   # (batch, emotion_dim)
        drop_speaker: bool = False,
        drop_emotion: bool = False,
    ) -> torch.Tensor:
        """
        Fuse speaker and emotion embeddings into a single conditioning vector.
        Supports independent dropout for each modality (multi-condition CFG).
        """
        if drop_speaker:
            speaker_emb = torch.zeros_like(speaker_emb)
        if drop_emotion:
            emotion_emb = torch.zeros_like(emotion_emb)
        
        combined = torch.cat([speaker_emb, emotion_emb], dim=-1)
        return self.fusion(combined)

    def forward(
        self,
        speaker_emb: torch.Tensor,           # (batch, speaker_dim)
        emotion_global: torch.Tensor,         # (batch, emotion_dim)
        emotion_frame: torch.Tensor | None = None,  # (batch, T, emotion_dim)
        drop_speaker: bool = False,
        drop_emotion: bool = False,
    ) -> dict:
        """
        Compute all conditioning outputs.
        
        Returns dict with:
            "fused":      (batch, model_dim) - fused global conditioning
            "adaln":      list of (batch, model_dim*6) per block - AdaLN params
            "frame_cond": (batch, T, emotion_dim) - for cross-attention
        """
        fused = self.get_fused_embedding(
            speaker_emb, emotion_global, drop_speaker, drop_emotion
        )

        result = {"fused": fused}

        if self.use_adaln:
            result["adaln"] = self.adaln_cond(fused)

        if self.use_cross_attn and emotion_frame is not None:
            if drop_emotion:
                emotion_frame = torch.zeros_like(emotion_frame)
            result["frame_cond"] = emotion_frame

        return result

    def apply_input_conditioning(
        self, x: torch.Tensor, fused: torch.Tensor
    ) -> torch.Tensor:
        """Apply input-level embedding addition."""
        if self.use_input_add:
            return self.input_add_cond(x, fused)
        return x

    def apply_block_conditioning(
        self,
        x: torch.Tensor,
        block_idx: int,
        adaln_params: list[torch.Tensor] | None,
        frame_cond: torch.Tensor | None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply conditioning at a specific DiT block.
        Called from within the enhanced DiT forward pass.
        """
        # Cross-attention with frame-level features
        if (
            self.use_cross_attn
            and str(block_idx) in self.cross_attns
            and frame_cond is not None
        ):
            x = self.cross_attns[str(block_idx)](x, frame_cond, mask)

        return x

    def get_adaln_residual(
        self, block_idx: int, adaln_params: list[torch.Tensor]
    ) -> torch.Tensor:
        """Get the AdaLN residual for a specific block."""
        if self.use_adaln and adaln_params is not None:
            return adaln_params[block_idx]
        return None
