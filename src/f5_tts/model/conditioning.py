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
        self.gate = nn.Parameter(torch.zeros(1))

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
    ):
        super().__init__()
        self.speaker_dim = speaker_dim
        self.emotion_dim = emotion_dim
        self.model_dim = model_dim
        self.use_adaln = use_adaln
        self.use_input_add = use_input_add
        self.use_cross_attn = use_cross_attn

        self.speaker_raw_proj = nn.Linear(speaker_raw_dim, speaker_dim) if speaker_raw_dim else None
        self.emotion_raw_proj = nn.Linear(emotion_raw_dim, emotion_dim) if emotion_raw_dim else None
        self.frame_raw_proj = (
            nn.Sequential(nn.Linear(emotion_raw_dim, emotion_dim), nn.SiLU(), nn.Linear(emotion_dim, emotion_dim))
            if emotion_raw_dim else None
        )
        if self.frame_raw_proj is not None:
            nn.init.zeros_(self.frame_raw_proj[-1].weight)
            nn.init.zeros_(self.frame_raw_proj[-1].bias)

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

    def project_model_conditions(
        self,
        *,
        speaker_raw: torch.Tensor | None = None,
        emotion_global_raw: torch.Tensor | None = None,
        emotion_frame_raw: torch.Tensor | None = None,
        speaker_present: torch.Tensor | None = None,
        emotion_global_present: torch.Tensor | None = None,
        emotion_frame_mask: torch.Tensor | None = None,
    ) -> ModelConditionBatch:
        return ModelConditionBatch(
            speaker=self.project_speaker(speaker_raw),
            emotion_global=self.project_emotion(emotion_global_raw),
            emotion_frame=self.project_frame(emotion_frame_raw),
            speaker_present=speaker_present,
            emotion_global_present=emotion_global_present,
            emotion_frame_mask=emotion_frame_mask,
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

    def forward(self, conditions: ModelConditionBatch | None = None, drop_speaker: torch.Tensor | bool | None = None, drop_emotion: torch.Tensor | bool | None = None) -> ConditioningOutputs:
        if conditions is None:
            return ConditioningOutputs()

        speaker_emb = conditions.speaker
        emotion_global = conditions.emotion_global
        emotion_frame = conditions.emotion_frame
        speaker_present = conditions.speaker_present
        emotion_present = conditions.emotion_global_present
        emotion_frame_mask = conditions.emotion_frame_mask

        if speaker_emb is None and emotion_global is None and emotion_frame is None:
            return ConditioningOutputs()

        ref = speaker_emb if speaker_emb is not None else emotion_global
        if ref is None:
            ref = emotion_frame[:, 0]
        batch = ref.shape[0]

        fused = adaln = frame_cond = frame_mask = None
        fused_present = None
        if (speaker_emb is not None) or (emotion_global is not None):
            fused, fused_present = self.fuse(
                speaker_emb, emotion_global,
                speaker_present=speaker_present,
                emotion_present=emotion_present,
                drop_speaker=drop_speaker,
                drop_emotion=drop_emotion,
            )
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
            frame_cond = F.normalize(emotion_frame, p=2, dim=-1)
            frame_cond = frame_cond * frame_mask[..., None].to(frame_cond.dtype)

        return ConditioningOutputs(fused_global=fused, fused_present_mask=fused_present, adaln=adaln, frame_cond=frame_cond, frame_mask=frame_mask)

    def apply_input_conditioning(self, x: torch.Tensor, fused: torch.Tensor, present_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.input_add(x, fused, present_mask=present_mask) if self.use_input_add and fused is not None else x

    def apply_block_conditioning(self, x: torch.Tensor, block_idx: int, frame_cond: torch.Tensor | None, frame_mask: torch.Tensor | None = None, x_mask: torch.Tensor | None = None, checkpoint_activations: bool = False) -> torch.Tensor:
        key = str(block_idx)
        if self.use_cross_attn and key in self.cross_attns and frame_cond is not None:
            if checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(self.cross_attns[key], x, frame_cond, frame_mask, x_mask, use_reentrant=False)
            else:
                x = self.cross_attns[key](x, frame_cond, cond_mask=frame_mask, x_mask=x_mask)
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
            if cond_out.frame_cond is None:
                return x
            return self.apply_block_conditioning(
                x, block_idx, cond_out.frame_cond, frame_mask=cond_out.frame_mask, x_mask=x_mask, checkpoint_activations=checkpoint_activations
            )

        return ConditioningRuntime(
            apply_input=_apply_input,
            adaln_for_block=_adaln_for_block,
            apply_block=_apply_block,
        )
