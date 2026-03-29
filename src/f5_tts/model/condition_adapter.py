from __future__ import annotations

import torch
import torch.nn.functional as F

from f5_tts.model.condition_types import RawConditionBatch, ModelConditionBatch
from f5_tts.model.encoder_utils import interpolate_temporal


class ConditionAdapter:
    """Convert raw encoder-space conditions into model-space conditions.

    Not an nn.Module — has no trainable parameters of its own.
    """

    def __init__(self, conditioning_module):
        self.conditioning_module = conditioning_module
        if not hasattr(conditioning_module, "project_model_conditions"):
            raise TypeError(
                "ConditionAdapter requires a conditioning module with a public "
                "project_model_conditions(...) API"
            )

    def align_frame(self, emotion_frame, emotion_frame_mask, target_len):
        if emotion_frame is None:
            return None, None
        if emotion_frame.shape[1] != target_len:
            emotion_frame = interpolate_temporal(emotion_frame, target_len)
        if emotion_frame_mask is not None:
            mask_f = emotion_frame_mask.float().unsqueeze(1)
            if mask_f.shape[-1] != target_len:
                mask_f = F.interpolate(mask_f, size=target_len, mode="nearest")
            emotion_frame_mask = mask_f.squeeze(1).bool()
            emotion_frame = emotion_frame * emotion_frame_mask[..., None].to(emotion_frame.dtype)
        return emotion_frame, emotion_frame_mask

    def align_prosody(self, prosody_raw, prosody_mask, target_len):
        """Align prosody frame features to target sequence length."""
        if prosody_raw is None:
            return None, None
        if prosody_raw.shape[1] != target_len:
            prosody_raw = interpolate_temporal(prosody_raw, target_len)
        if prosody_mask is not None:
            mask_f = prosody_mask.float().unsqueeze(1)
            if mask_f.shape[-1] != target_len:
                mask_f = F.interpolate(mask_f, size=target_len, mode="nearest")
            prosody_mask = mask_f.squeeze(1).bool()
            prosody_raw = prosody_raw * prosody_mask[..., None].to(prosody_raw.dtype)
        return prosody_raw, prosody_mask

    def __call__(self, conditions: RawConditionBatch | None, *, target_len: int | None = None) -> ModelConditionBatch:
        if conditions is None:
            return ModelConditionBatch()

        emotion_frame, emotion_frame_mask = self.align_frame(
            conditions.emotion_frame_raw,
            conditions.emotion_frame_mask,
            target_len,
        ) if target_len is not None else (conditions.emotion_frame_raw, conditions.emotion_frame_mask)

        prosody_raw, prosody_mask = self.align_prosody(
            conditions.prosody_raw,
            conditions.prosody_mask,
            target_len,
        ) if target_len is not None else (conditions.prosody_raw, conditions.prosody_mask)

        return self.conditioning_module.project_model_conditions(
            speaker_raw=conditions.speaker_raw,
            emotion_global_raw=conditions.emotion_global_raw,
            emotion_frame_raw=emotion_frame,
            prosody_raw=prosody_raw,
            speaker_present=conditions.speaker_present,
            emotion_global_present=conditions.emotion_global_present,
            emotion_frame_mask=emotion_frame_mask,
            prosody_mask=prosody_mask,
        )
