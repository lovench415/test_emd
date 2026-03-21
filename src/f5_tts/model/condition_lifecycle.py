from __future__ import annotations

import torch
import torch.nn.functional as F

from f5_tts.model.condition_types import RawConditionBatch, ModelConditionBatch, ConditionPairBatch
from f5_tts.model.condition_adapter import ConditionAdapter


class ConditionLifecycleManager:
    """Single owner for condition lifecycle.

    Responsibilities:
      - normalize raw batch fields into RawConditionBatch
      - backfill missing rows from online encoders
      - validate condition invariants
      - adapt/project raw conditions into model-space conditions
      - build unconditional model-space conditions when needed

    Not an nn.Module — has no trainable parameters of its own.  The
    conditioning_module it references is already registered in the
    transformer's module tree (transformer.cond_aggregator).

    Notes on presence flags:
      - explicit per-row flags from upstream are preferred
      - lifecycle no longer synthesizes coarse global presence from mere tensor availability
      - if upstream omits global presence flags, lifecycle preserves them as None until a
        later stage can establish them exactly (for example, online backfill rows)
    """

    def __init__(self, conditioning_module):
        self.adapter = ConditionAdapter(conditioning_module)

    def _normalize_global_present(self, explicit: torch.Tensor | None) -> torch.Tensor | None:
        if explicit is None:
            return None
        return explicit.to(dtype=torch.bool)

    def prepare_raw_from_batch(self, batch: dict, device: torch.device | str) -> RawConditionBatch:
        def move(name):
            t = batch.get(name)
            return t.to(device) if t is not None else None

        raw = RawConditionBatch(
            speaker_raw=move("speaker_raw"),
            emotion_global_raw=move("emotion_global_raw"),
            emotion_frame_raw=move("emotion_frame_raw"),
            speaker_present=move("speaker_present"),
            emotion_global_present=move("emotion_global_present"),
            emotion_frame_mask=move("emotion_frame_mask"),
        )

        raw.speaker_present = self._normalize_global_present(raw.speaker_present)
        raw.emotion_global_present = self._normalize_global_present(raw.emotion_global_present)
        if raw.emotion_frame_mask is not None:
            raw.emotion_frame_mask = raw.emotion_frame_mask.to(dtype=torch.bool)

        return raw

    def backfill_missing(
        self,
        raw_conditions: RawConditionBatch,
        batch: dict,
        speaker_encoder=None,
        emotion_encoder=None,
        device: torch.device | str | None = None,
    ) -> RawConditionBatch:
        device = device or (
            raw_conditions.speaker_raw.device if raw_conditions.speaker_raw is not None else 'cpu'
        )
        raw_audio = batch.get("raw_audio")
        raw_audio_present = batch.get("raw_audio_present")
        if raw_audio is None or "sample_rate" not in batch:
            return raw_conditions
        raw_audio = raw_audio.to(device)
        if raw_audio_present is not None:
            raw_audio_present = raw_audio_present.to(device=device, dtype=torch.bool)

        batch_size = raw_audio.shape[0]
        speaker_present = self._normalize_global_present(raw_conditions.speaker_present)
        emotion_present = self._normalize_global_present(raw_conditions.emotion_global_present)
        frame_mask = raw_conditions.emotion_frame_mask

        known_speaker_present = speaker_present if speaker_present is not None else torch.zeros(batch_size, dtype=torch.bool, device=device)
        known_emotion_present = emotion_present if emotion_present is not None else torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Without explicit per-row presence flags, lifecycle does not guess mixed-row availability
        # from the mere existence of a batch tensor. It only backfills rows known to be missing.
        need_spk_rows = (~known_speaker_present) if speaker_encoder is not None else torch.zeros(batch_size, dtype=torch.bool, device=device)
        need_emo_g_rows = (~known_emotion_present) if emotion_encoder is not None else torch.zeros(batch_size, dtype=torch.bool, device=device)
        if emotion_encoder is not None:
            if raw_conditions.emotion_frame_raw is None or frame_mask is None:
                need_emo_f_rows = torch.ones(batch_size, dtype=torch.bool, device=device)
            else:
                need_emo_f_rows = ~frame_mask.any(dim=1)
        else:
            need_emo_f_rows = torch.zeros(batch_size, dtype=torch.bool, device=device)

        need_any_online = need_spk_rows | need_emo_g_rows | need_emo_f_rows
        if raw_audio_present is not None:
            need_any_online &= raw_audio_present
        if not need_any_online.any():
            raw_conditions.speaker_present = speaker_present
            raw_conditions.emotion_global_present = emotion_present
            raw_conditions.emotion_frame_mask = frame_mask
            return raw_conditions

        wav = raw_audio[need_any_online]
        sr = batch["sample_rate"]
        with torch.no_grad():
            online_spk = speaker_encoder.extract_raw(wav, sr=sr) if (speaker_encoder is not None and need_spk_rows[need_any_online].any()) else None
            if emotion_encoder is not None and (need_emo_g_rows[need_any_online].any() or need_emo_f_rows[need_any_online].any()):
                if hasattr(emotion_encoder, "extract_raw_with_mask"):
                    online_emo = emotion_encoder.extract_raw_with_mask(wav, sr=sr)
                else:
                    online_emo = emotion_encoder.extract_raw(wav, sr=sr)
            else:
                online_emo = None

        selected_idx = torch.nonzero(need_any_online, as_tuple=False).squeeze(1)

        if online_spk is not None:
            if raw_conditions.speaker_raw is None:
                raw_conditions.speaker_raw = torch.zeros(batch_size, online_spk.shape[-1], dtype=online_spk.dtype, device=device)
            fill_idx = selected_idx[need_spk_rows[selected_idx]]
            if fill_idx.numel() > 0:
                fill_values = online_spk[need_spk_rows[selected_idx]]
                raw_conditions.speaker_raw[fill_idx] = fill_values
                if speaker_present is None:
                    speaker_present = torch.zeros(batch_size, dtype=torch.bool, device=device)
                speaker_present[fill_idx] = True

        if online_emo is not None:
            if isinstance(online_emo, tuple) and len(online_emo) == 3:
                online_emo_g, online_emo_f, online_emo_mask = online_emo
            else:
                online_emo_g, online_emo_f = online_emo
                online_emo_mask = None
            if online_emo_g is not None:
                if raw_conditions.emotion_global_raw is None:
                    raw_conditions.emotion_global_raw = torch.zeros(batch_size, online_emo_g.shape[-1], dtype=online_emo_g.dtype, device=device)
                fill_idx = selected_idx[need_emo_g_rows[selected_idx]]
                if fill_idx.numel() > 0:
                    fill_values = online_emo_g[need_emo_g_rows[selected_idx]]
                    raw_conditions.emotion_global_raw[fill_idx] = fill_values
                    if emotion_present is None:
                        emotion_present = torch.zeros(batch_size, dtype=torch.bool, device=device)
                    emotion_present[fill_idx] = True

            if online_emo_f is not None:
                if online_emo_mask is None:
                    raise ValueError(
        "emotion_encoder returned emotion_frame without mask. "
        "Online emotion encoders must return frame mask explicitly."
    )
                else:
                    online_emo_mask = online_emo_mask.to(device=online_emo_f.device, dtype=torch.bool)

                if raw_conditions.emotion_frame_raw is None:
                    raw_conditions.emotion_frame_raw = torch.zeros(batch_size, online_emo_f.shape[1], online_emo_f.shape[2], dtype=online_emo_f.dtype, device=device)
                    frame_mask = torch.zeros(batch_size, online_emo_f.shape[1], dtype=torch.bool, device=device)
                elif raw_conditions.emotion_frame_raw.shape[1] < online_emo_f.shape[1]:
                    pad_t = online_emo_f.shape[1] - raw_conditions.emotion_frame_raw.shape[1]
                    raw_conditions.emotion_frame_raw = F.pad(raw_conditions.emotion_frame_raw, (0, 0, 0, pad_t))
                    frame_mask = F.pad(frame_mask, (0, pad_t), value=False)

                fill_idx = selected_idx[need_emo_f_rows[selected_idx]]
                if fill_idx.numel() > 0:
                    fill_values = online_emo_f[need_emo_f_rows[selected_idx]]
                    fill_masks = online_emo_mask[need_emo_f_rows[selected_idx]]
                    if fill_values.shape[1] < raw_conditions.emotion_frame_raw.shape[1]:
                        pad_t = raw_conditions.emotion_frame_raw.shape[1] - fill_values.shape[1]
                        fill_values = F.pad(fill_values, (0, 0, 0, pad_t))
                        fill_masks = F.pad(fill_masks, (0, pad_t), value=False)
                    raw_conditions.emotion_frame_raw[fill_idx] = fill_values
                    frame_mask[fill_idx] = fill_masks

        raw_conditions.speaker_present = speaker_present
        raw_conditions.emotion_global_present = emotion_present
        raw_conditions.emotion_frame_mask = frame_mask
        return raw_conditions

    def validate_raw(self, raw_conditions: RawConditionBatch | None) -> RawConditionBatch:
        if raw_conditions is None:
            return RawConditionBatch()
        raw_conditions.speaker_present = self._normalize_global_present(raw_conditions.speaker_present)
        raw_conditions.emotion_global_present = self._normalize_global_present(raw_conditions.emotion_global_present)
        if raw_conditions.emotion_frame_raw is not None and raw_conditions.emotion_frame_mask is None:
            raise ValueError("emotion_frame_raw requires emotion_frame_mask")
        if raw_conditions.emotion_frame_raw is None and raw_conditions.emotion_frame_mask is not None:
            raise ValueError("emotion_frame_mask was provided without emotion_frame_raw")
        if raw_conditions.emotion_frame_raw is not None and raw_conditions.emotion_frame_raw.shape[:2] != raw_conditions.emotion_frame_mask.shape:
            raise ValueError("emotion_frame_raw and emotion_frame_mask have inconsistent shapes")
        if raw_conditions.speaker_raw is not None and raw_conditions.speaker_present is not None and raw_conditions.speaker_raw.shape[0] != raw_conditions.speaker_present.shape[0]:
            raise ValueError("speaker_raw and speaker_present batch sizes differ")
        if raw_conditions.emotion_global_raw is not None and raw_conditions.emotion_global_present is not None and raw_conditions.emotion_global_raw.shape[0] != raw_conditions.emotion_global_present.shape[0]:
            raise ValueError("emotion_global_raw and emotion_global_present batch sizes differ")
        if raw_conditions.emotion_frame_raw is not None and raw_conditions.speaker_present is not None and raw_conditions.emotion_frame_raw.shape[0] != raw_conditions.speaker_present.shape[0]:
            raise ValueError("emotion_frame_raw and speaker_present batch sizes differ")
        if raw_conditions.emotion_frame_raw is not None and raw_conditions.emotion_global_present is not None and raw_conditions.emotion_frame_raw.shape[0] != raw_conditions.emotion_global_present.shape[0]:
            raise ValueError("emotion_frame_raw and emotion_global_present batch sizes differ")
        return raw_conditions

    def to_model_conditions(self, raw_conditions: RawConditionBatch | None, *, target_len: int | None) -> ModelConditionBatch:
        self.validate_raw(raw_conditions)
        return self.adapter(raw_conditions, target_len=target_len)

    def prepare(
        self,
        *,
        batch: dict | None = None,
        raw_conditions: RawConditionBatch | None = None,
        model_conditions: ModelConditionBatch | None = None,
        target_len: int | None,
        device=None,
        speaker_encoder=None,
        emotion_encoder=None,
    ) -> ModelConditionBatch:
        """Canonical public entrypoint for model-space conditions.

        Exactly one source must be provided:
        - batch -> normalize raw batch fields + optional online backfill + raw->model
        - raw_conditions -> validate raw + raw->model
        - model_conditions -> pass-through after light validation
        """
        provided = int(batch is not None) + int(raw_conditions is not None) + int(model_conditions is not None)
        if provided != 1:
            raise ValueError("Exactly one of batch, raw_conditions, or model_conditions must be provided")

        if model_conditions is not None:
            if not isinstance(model_conditions, ModelConditionBatch):
                raise TypeError("model_conditions must be a ModelConditionBatch")
            return model_conditions

        if batch is not None:
            if device is None:
                raise ValueError("device is required when preparing conditions from a batch")
            raw = self.prepare_raw_from_batch(batch, device)
            raw = self.backfill_missing(
                raw,
                batch,
                speaker_encoder=speaker_encoder,
                emotion_encoder=emotion_encoder,
                device=device,
            )
            return self.to_model_conditions(raw, target_len=target_len)

        return self.to_model_conditions(raw_conditions, target_len=target_len)

    def prepare_pair(
        self,
        *,
        batch: dict | None = None,
        raw_conditions: RawConditionBatch | None = None,
        model_conditions: ModelConditionBatch | None = None,
        target_len: int | None,
        use_unconditional: bool,
        device=None,
        speaker_encoder=None,
        emotion_encoder=None,
    ) -> ConditionPairBatch:
        cond = self.prepare(
            batch=batch,
            raw_conditions=raw_conditions,
            model_conditions=model_conditions,
            target_len=target_len,
            device=device,
            speaker_encoder=speaker_encoder,
            emotion_encoder=emotion_encoder,
        )
        return self.make_condition_pair(cond, use_unconditional=use_unconditional)


    def build_runtime(
        self,
        model_conditions: ModelConditionBatch | None,
        *,
        drop_speaker: torch.Tensor | bool | None = None,
        drop_emotion: torch.Tensor | bool | None = None,
    ):
        cond_module = getattr(self.adapter, "conditioning_module", None)
        if cond_module is None:
            raise RuntimeError("ConditionLifecycleManager requires a conditioning module to build runtime")
        if model_conditions is None:
            return cond_module.build_runtime(None)
        outputs = cond_module(
            conditions=model_conditions,
            drop_speaker=drop_speaker,
            drop_emotion=drop_emotion,
        )
        return cond_module.build_runtime(outputs)

    def prepare_runtime(
        self,
        *,
        batch: dict | None = None,
        raw_conditions: RawConditionBatch | None = None,
        model_conditions: ModelConditionBatch | None = None,
        target_len: int | None,
        device=None,
        speaker_encoder=None,
        emotion_encoder=None,
        drop_speaker: torch.Tensor | bool | None = None,
        drop_emotion: torch.Tensor | bool | None = None,
    ):
        cond = self.prepare(
            batch=batch,
            raw_conditions=raw_conditions,
            model_conditions=model_conditions,
            target_len=target_len,
            device=device,
            speaker_encoder=speaker_encoder,
            emotion_encoder=emotion_encoder,
        )
        return self.build_runtime(cond, drop_speaker=drop_speaker, drop_emotion=drop_emotion)

    def prepare_runtime_pair(
        self,
        *,
        batch: dict | None = None,
        raw_conditions: RawConditionBatch | None = None,
        model_conditions: ModelConditionBatch | None = None,
        target_len: int | None,
        use_unconditional: bool,
        device=None,
        speaker_encoder=None,
        emotion_encoder=None,
        drop_speaker: torch.Tensor | bool | None = None,
        drop_emotion: torch.Tensor | bool | None = None,
    ) -> tuple[object, object | None, ConditionPairBatch]:
        pair = self.prepare_pair(
            batch=batch,
            raw_conditions=raw_conditions,
            model_conditions=model_conditions,
            target_len=target_len,
            use_unconditional=use_unconditional,
            device=device,
            speaker_encoder=speaker_encoder,
            emotion_encoder=emotion_encoder,
        )
        cond_runtime = self.build_runtime(pair.cond, drop_speaker=drop_speaker, drop_emotion=drop_emotion)
        uncond_runtime = self.build_runtime(pair.uncond) if pair.uncond is not None else None
        return cond_runtime, uncond_runtime, pair

    def prepare_for_train(self, batch: dict, *, target_len: int, device, speaker_encoder=None, emotion_encoder=None) -> ModelConditionBatch:
        return self.prepare(
            batch=batch,
            target_len=target_len,
            device=device,
            speaker_encoder=speaker_encoder,
            emotion_encoder=emotion_encoder,
        )

    def prepare_for_infer(self, raw_conditions: RawConditionBatch | None, *, target_len: int | None) -> ModelConditionBatch:
        return self.prepare(raw_conditions=raw_conditions, target_len=target_len)

    def make_unconditional(self, model_conditions: ModelConditionBatch | None) -> ModelConditionBatch:
        if model_conditions is None:
            return ModelConditionBatch()
        zeros_sp = None if model_conditions.speaker_present is None else torch.zeros_like(model_conditions.speaker_present)
        zeros_em = None if model_conditions.emotion_global_present is None else torch.zeros_like(model_conditions.emotion_global_present)
        zeros_fm = None if model_conditions.emotion_frame_mask is None else torch.zeros_like(model_conditions.emotion_frame_mask)
        return ModelConditionBatch(
            speaker=model_conditions.speaker,
            emotion_global=model_conditions.emotion_global,
            emotion_frame=model_conditions.emotion_frame,
            speaker_present=zeros_sp,
            emotion_global_present=zeros_em,
            emotion_frame_mask=zeros_fm,
        )

    def make_condition_pair(self, model_conditions: ModelConditionBatch | None, *, use_unconditional: bool) -> ConditionPairBatch:
        cond = model_conditions if model_conditions is not None else ModelConditionBatch()
        uncond = self.make_unconditional(cond) if use_unconditional else None
        return ConditionPairBatch(cond=cond, uncond=uncond)

    def prepare_pair_for_infer(
        self,
        raw_conditions: RawConditionBatch | None,
        *,
        target_len: int | None,
        use_unconditional: bool,
    ) -> ConditionPairBatch:
        return self.prepare_pair(
            raw_conditions=raw_conditions,
            target_len=target_len,
            use_unconditional=use_unconditional,
        )
