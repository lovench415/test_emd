"""
Enhanced CFM (Conditional Flow Matching) for F5-TTS
=====================================================

Extended CFM that supports:
1. Speaker embeddings from reference audio (language-agnostic voice cloning)
2. Emotion embeddings (global + frame-level) for emotional transfer
3. Emotion-Guided Classifier-Free Guidance (EG-CFG) at inference
4. Multi-condition dropout during training for flexible inference

Key design:
- Training: randomly drop speaker/emotion/audio/text conditions
  to enable flexible CFG at inference (any combination)
- Inference (sample): emotion embeddings steer the ODE trajectory
  through the EG-CFG mechanism (F5-TTS-Emotional-CFG inspired)

The enhanced CFM wraps the EnhancedDiT backbone and manages the
full training/inference pipeline including embedding extraction.
"""

from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)


class EnhancedCFM(nn.Module):
    """
    Enhanced Conditional Flow Matching with speaker/emotion conditioning.
    
    Extends the original F5-TTS CFM with:
    - Speaker embedding injection for improved voice cloning
    - Emotion embedding injection for emotional speech synthesis
    - Emotion-Guided CFG (EG-CFG) for inference-time emotion control
    - Multi-condition dropout for flexible guidance strategies
    """

    def __init__(
        self,
        transformer: nn.Module,  # EnhancedDiT
        sigma=0.0,
        odeint_kwargs: dict = dict(method="euler"),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        speaker_drop_prob=0.1,
        emotion_drop_prob=0.1,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str, int] | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # Mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # CFG drop probabilities
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.speaker_drop_prob = speaker_drop_prob
        self.emotion_drop_prob = emotion_drop_prob

        # Transformer backbone (EnhancedDiT)
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # Flow matching
        self.sigma = sigma
        self.odeint_kwargs = odeint_kwargs

        # Vocab
        self.vocab_char_map = vocab_char_map

    @property
    def device(self):
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    # Inference: sample with emotion-guided CFG
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        text: torch.Tensor | list[str],
        duration: int | torch.Tensor,
        *,
        lens: torch.Tensor | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=65536,
        vocoder: Callable | None = None,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
        # ── New: conditioning embeddings ──
        speaker_emb: torch.Tensor | None = None,
        emotion_global: torch.Tensor | None = None,
        emotion_frame: torch.Tensor | None = None,
        emotion_cfg_strength: float = 0.0,  # EG-CFG: separate emotion guidance
    ):
        """
        Sample with optional emotion-guided classifier-free guidance.
        
        EG-CFG mechanism (when emotion_cfg_strength > 0):
            pred = pred_cond + cfg_strength * (pred_cond - pred_uncond)
                   + emotion_cfg_strength * (pred_emo_cond - pred_emo_uncond)
        
        This allows independent control of:
            - cfg_strength: overall generation quality/diversity
            - emotion_cfg_strength: how strongly to apply the emotion
        """
        self.eval()

        # Process mel spectrogram from raw wave
        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # Text processing
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # Duration
        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )
        duration = duration.clamp(max=max_duration)
        max_duration_val = duration.amax()

        # Duplicate test
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration_val - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration_val - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration_val - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

        if batch > 1:
            mask = lens_to_mask(duration)
        else:
            mask = None

        # Interpolate emotion_frame to target length
        if emotion_frame is not None and emotion_frame.shape[1] != max_duration_val:
            emotion_frame = emotion_frame.permute(0, 2, 1)
            emotion_frame = F.interpolate(
                emotion_frame, size=int(max_duration_val), mode="linear", align_corners=False,
            )
            emotion_frame = emotion_frame.permute(0, 2, 1)

        # ── ODE function with optional EG-CFG ──
        use_emotion_cfg = (
            emotion_cfg_strength > 0.01
            and emotion_global is not None
        )

        def fn(t, x):
            if cfg_strength < 1e-5 and not use_emotion_cfg:
                # No guidance at all
                pred = self.transformer(
                    x=x, cond=step_cond, text=text, time=t, mask=mask,
                    drop_audio_cond=False, drop_text=False, cache=True,
                    speaker_emb=speaker_emb,
                    emotion_global=emotion_global,
                    emotion_frame=emotion_frame,
                )
                return pred

            if use_emotion_cfg:
                # Three-way CFG: full cond, no emotion, fully uncond
                # Stack: [cond, no_emo, uncond]
                x3 = torch.cat([x, x, x], dim=0)
                cond3 = torch.cat([step_cond, step_cond, step_cond], dim=0)
                text3 = torch.cat([text, text, text], dim=0)
                time3 = t.repeat(3) if t.ndim > 0 else t

                mask3 = torch.cat([mask, mask, mask], dim=0) if mask is not None else None

                # Speaker embedding: present for cond and no_emo, zero for uncond
                spk3 = torch.cat([
                    speaker_emb,
                    speaker_emb,
                    torch.zeros_like(speaker_emb),
                ], dim=0) if speaker_emb is not None else None

                # Emotion: present for cond, zero for no_emo and uncond
                emo_g3 = torch.cat([
                    emotion_global,
                    torch.zeros_like(emotion_global),
                    torch.zeros_like(emotion_global),
                ], dim=0) if emotion_global is not None else None

                emo_f3 = torch.cat([
                    emotion_frame,
                    torch.zeros_like(emotion_frame),
                    torch.zeros_like(emotion_frame),
                ], dim=0) if emotion_frame is not None else None

                pred3 = self.transformer(
                    x=x3, cond=cond3, text=text3, time=time3, mask=mask3,
                    drop_audio_cond=False, drop_text=False, cache=False,
                    speaker_emb=spk3,
                    emotion_global=emo_g3,
                    emotion_frame=emo_f3,
                )

                pred_cond, pred_no_emo, pred_uncond = torch.chunk(pred3, 3, dim=0)

                # EG-CFG formula
                return (
                    pred_uncond
                    + cfg_strength * (pred_cond - pred_uncond)
                    + emotion_cfg_strength * (pred_cond - pred_no_emo)
                )

            else:
                # Standard 2-way CFG (same as original F5-TTS)
                pred_cfg = self.transformer(
                    x=x, cond=step_cond, text=text, time=t, mask=mask,
                    cfg_infer=True, cache=True,
                    speaker_emb=speaker_emb,
                    emotion_global=emotion_global,
                    emotion_frame=emotion_frame,
                )
                pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
                return pred + (pred - null_pred) * cfg_strength

        # Noise input
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        if t_start == 0 and use_epss:
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    # ------------------------------------------------------------------
    # Training: forward with multi-condition dropout
    # ------------------------------------------------------------------

    def forward(
        self,
        inp: torch.Tensor,  # (b, n, d) mel or (b, nw) raw wave
        text: torch.Tensor | list[str],
        *,
        lens: torch.Tensor | None = None,
        noise_scheduler: str | None = None,
        # ── New: conditioning embeddings ──
        speaker_emb: torch.Tensor | None = None,
        emotion_global: torch.Tensor | None = None,
        emotion_frame: torch.Tensor | None = None,
    ):
        """
        Training forward pass with multi-condition dropout.
        
        Dropout strategy (applied independently per sample in batch):
        - p_drop_audio: drop audio condition (original F5-TTS)
        - p_drop_text:  drop text condition  (original F5-TTS)
        - p_drop_speaker: drop speaker embedding
        - p_drop_emotion: drop emotion embedding
        - p_uncond: drop ALL conditions (for unconditional baseline)
        """
        # Handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device
        _σ1 = self.sigma

        # Text
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # Mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)

        # Random span mask for infilling
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)
        if exists(mask):
            rand_span_mask &= mask

        # mel = x1, noise = x0
        x1 = inp
        x0 = torch.randn_like(x1)

        # Timestep
        time = torch.rand((batch,), dtype=dtype, device=self.device)

        # Interpolate
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # Condition = unmasked part of mel
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # ── Multi-condition dropout ──
        drop_audio_cond = random() < self.audio_drop_prob
        drop_text = False
        drop_speaker = False
        drop_emotion = False

        if random() < self.cond_drop_prob:
            # Full unconditional
            drop_audio_cond = True
            drop_text = True
            drop_speaker = True
            drop_emotion = True
        else:
            # Independent dropout for speaker and emotion
            drop_speaker = random() < self.speaker_drop_prob
            drop_emotion = random() < self.emotion_drop_prob

        # Interpolate emotion_frame to seq_len
        if emotion_frame is not None and emotion_frame.shape[1] != seq_len:
            emotion_frame = emotion_frame.permute(0, 2, 1)
            emotion_frame = F.interpolate(
                emotion_frame, size=seq_len, mode="linear", align_corners=False,
            )
            emotion_frame = emotion_frame.permute(0, 2, 1)

        # Predict flow
        pred = self.transformer(
            x=φ, cond=cond, text=text, time=time,
            drop_audio_cond=drop_audio_cond, drop_text=drop_text, mask=mask,
            speaker_emb=speaker_emb,
            emotion_global=emotion_global,
            emotion_frame=emotion_frame,
            drop_speaker=drop_speaker,
            drop_emotion=drop_emotion,
        )

        # Flow matching loss (only on masked span)
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred
