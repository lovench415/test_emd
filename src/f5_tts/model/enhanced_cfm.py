"""
Enhanced CFM (Conditional Flow Matching) for F5-TTS.

Extends the original with:
  - Speaker/emotion embedding injection
  - Multi-condition dropout (independent speaker/emotion/audio/text)
  - Emotion-Guided CFG (EG-CFG) at inference
"""
from __future__ import annotations
from typing import Callable
from collections import Counter
import random as pyrandom
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default, exists, get_epss_timesteps,
    lens_to_mask, list_str_to_idx, list_str_to_tensor, mask_from_frac_lengths,
)
from f5_tts.model.condition_types import ModelConditionBatch
from f5_tts.model.condition_lifecycle import ConditionLifecycleManager
from f5_tts.model.duration_predictor import DurationPredictor
from f5_tts.model.prosody_encoder import PROSODY_RAW_DIM



class EnhancedCFM(nn.Module):
    """Enhanced CFM with speaker/emotion conditioning and EG-CFG."""

    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(method="euler"),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        speaker_drop_prob=0.1,
        emotion_drop_prob=0.1,
        dropout_mode_probs: dict[str, float] | None = None,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str, int] | None = None,
        use_duration_predictor: bool = False,
        speaker_emb_dim: int = 512,
    ):
        super().__init__()
        self.frac_lengths_mask = frac_lengths_mask
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.speaker_drop_prob = speaker_drop_prob
        self.emotion_drop_prob = emotion_drop_prob
        # Multi-condition dropout distribution (categorical, CFG-aligned).
        # This replaces independent Bernoulli drops with a single sampled mode,
        # so coverage of {uncond, voice_only, full, no_speaker} is controlled.
        # If dropout_mode_probs is not provided, we derive a sane default from
        # existing *_drop_prob args and then renormalize to sum to 1.

        if dropout_mode_probs is None:
            dropout_mode_probs = {
                'uncond': float(self.cond_drop_prob),       # 0.2  — trains u branch
                'voice_only': float(self.emotion_drop_prob),# 0.1  — trains v branch (speaker only)
                'no_emotion': 0.08,                         # trains e branch (speaker + prosody, no emotion)
                'no_prosody': 0.08,                         # trains disentanglement (speaker + emotion, no prosody)
                'no_speaker': float(self.speaker_drop_prob),# 0.1  — trains speaker-drop branch
                'textless': 0.10,                           # trains text-drop CFG branch
            }
            p_full = 1.0 - sum(dropout_mode_probs.values())
            dropout_mode_probs['full'] = max(0.0, p_full)
        # Renormalize (in case of rounding / user input)
        total_p = sum(dropout_mode_probs.values())

        if total_p <= 0:
            dropout_mode_probs = {'full': 1.0}
        else:
            dropout_mode_probs = {k: v / total_p for k, v in dropout_mode_probs.items()}

        self.dropout_mode_probs = dropout_mode_probs
        self._dropout_modes = list(dropout_mode_probs.keys())
        self._dropout_probs = [dropout_mode_probs[m] for m in self._dropout_modes]

        self.transformer = transformer
        self.condition_lifecycle = ConditionLifecycleManager(transformer.cond_aggregator) if hasattr(transformer, "cond_aggregator") else None
        self.dim = transformer.dim
        self.sigma = sigma
        self.odeint_kwargs = odeint_kwargs
        self.vocab_char_map = vocab_char_map

        # Duration predictor (optional, prosody-conditioned speaking rate)
        # Uses RAW speaker embeddings (not projected), so dim must match speaker_raw_dim.
        self.duration_predictor = None
        if use_duration_predictor:
            spk_raw_dim = getattr(transformer.cond_aggregator, 'speaker_raw_proj', None)
            spk_raw_dim = spk_raw_dim.in_features if spk_raw_dim is not None else speaker_emb_dim
            self.duration_predictor = DurationPredictor(
                prosody_global_dim=PROSODY_RAW_DIM,
                speaker_dim=spk_raw_dim,
                use_speaker=True,
            )

    @property
    def device(self):
        return next(self.parameters()).device

    # ── Shared helpers ────────────────────────────────────────────────

    def _to_mel(self, inp: torch.Tensor) -> torch.Tensor:
        """Convert raw waveform to mel if needed."""
        if inp.ndim == 2:
            inp = self.mel_spec(inp).permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels
        return inp

    def _to_text_ids(self, text, batch: int, device) -> torch.Tensor:
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch
        return text

    def _build_conditioning_runtime(self, conditions: ModelConditionBatch | None, *, drop_speaker=False, drop_emotion=False, drop_prosody=False):
        if conditions is None and self.condition_lifecycle is None:
            return None
        if self.condition_lifecycle is None:
            raise RuntimeError("condition_lifecycle is required to build conditioning runtime from non-None conditions")
        return self.condition_lifecycle.build_runtime(
            conditions,
            drop_speaker=drop_speaker,
            drop_emotion=drop_emotion,
            drop_prosody=drop_prosody,
        )

    def sample_condition_dropout(self, batch: int, device):
        probs = torch.tensor(self._dropout_probs, device=device, dtype=torch.float)
        mode_ids = torch.multinomial(probs, num_samples=batch, replacement=True)
        mode_names = [self._dropout_modes[i] for i in mode_ids.tolist()]

        drop_audio = torch.rand(batch, device=device) < self.audio_drop_prob
        drop_text = torch.zeros(batch, dtype=torch.bool, device=device)
        drop_speaker = torch.zeros(batch, dtype=torch.bool, device=device)
        drop_emotion = torch.zeros(batch, dtype=torch.bool, device=device)
        drop_prosody = torch.zeros(batch, dtype=torch.bool, device=device)

        for i, mode in enumerate(mode_names):
            if mode == "uncond":
                drop_audio[i] = True
                drop_text[i] = True
                drop_speaker[i] = True
                drop_emotion[i] = True
                drop_prosody[i] = True
            elif mode == "voice_only":
                # Voice = speaker identity only; drop all expression
                drop_emotion[i] = True
                drop_prosody[i] = True
            elif mode == "no_emotion":
                # Speaker + prosody, no emotion → trains 'e' branch for 4-branch CFG
                drop_emotion[i] = True
            elif mode == "no_prosody":
                # Speaker + emotion, no prosody → trains disentanglement
                drop_prosody[i] = True
            elif mode == "no_speaker":
                drop_speaker[i] = True
            elif mode == "textless":
                drop_text[i] = True

        return {
            "drop_audio": drop_audio,
            "drop_text": drop_text,
            "drop_speaker": drop_speaker,
            "drop_emotion": drop_emotion,
            "drop_prosody": drop_prosody,
            "mode_ids": mode_ids,
            "mode_names": mode_names,
        }

    # ── Inference ─────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self, cond, text, duration, *, lens=None, steps=32,
        cfg_strength=1.0, sway_sampling_coef=None, seed=None,
        max_duration=65536, vocoder=None, use_epss=True,
        no_ref_audio=False, duplicate_test=False, t_inter=0.1, edit_mask=None,
        conditions: ModelConditionBatch | None = None,
        emotion_cfg_strength=0.0, prosody_cfg_strength=0.0,
    ):
        """
        Sample with optional 4-branch EG-CFG for independent emotion/prosody control.

        4-branch EG-CFG:
            result = f + cfg·(f − u)
                     + emo·(f − e)          ← emotion boost
                     + pros·(e − v)          ← prosody boost

        where:
            f = p_full:    speaker + emotion + prosody + text + audio
            e = p_no_emo:  speaker + prosody + text + audio  (emotion dropped)
            v = p_voice:   speaker + text + audio            (emotion + prosody dropped)
            u = p_uncond:  all conditions dropped

        Decomposition:
            (f − e) = what emotion adds beyond (speaker + prosody)
            (e − v) = what prosody adds beyond (speaker only)
            (f − u) = total conditioning signal

        Edge cases:
            emo=0, pros=0  →  f + cfg·(f−u)                = standard 2-way CFG
            emo>0, pros=0  →  f + cfg·(f−u) + emo·(f−e)    = 3-branch (emotion)
            emo=0, pros>0  →  f + cfg·(f−u) + pros·(e−v)   = 4-branch (prosody)
            emo>0, pros>0  →  full 4-branch                 = independent control
        """
        self.eval()

        if cond.ndim == 2:
            cond = self.mel_spec(cond).permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels
        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        text = self._to_text_ids(text, batch, device)

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask
        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)
        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration,
        ).clamp(max=max_duration).long()
        max_dur = duration.amax()

        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_dur - 2 * cond_seq_len))

        cond = F.pad(cond, (0, 0, 0, max_dur - cond_seq_len))
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_dur - cond_mask.shape[-1]), value=False)
        step_cond = torch.where(cond_mask.unsqueeze(-1), cond, torch.zeros_like(cond))
        mask = lens_to_mask(duration) if batch > 1 else None

        model_conditions = conditions
        use_unconditional = (cfg_strength >= 1e-5 or emotion_cfg_strength > 0.01 or prosody_cfg_strength > 0.01)
        uncond_conditions = None
        if self.condition_lifecycle is not None and model_conditions is not None:
            cond_pair = self.condition_lifecycle.prepare_pair(
                model_conditions=model_conditions,
                target_len=max_dur,
                use_unconditional=use_unconditional,
            )
            model_conditions = cond_pair.cond
            uncond_conditions = cond_pair.uncond
        elif model_conditions is not None and use_unconditional:
            raise RuntimeError(
                "EnhancedCFM.sample() received model-space conditions, but condition_lifecycle is unavailable "
                "to build the conditional/unconditional pair required for CFG sampling."
            )

        has_emotion = (
            model_conditions is not None
            and (
                (model_conditions.emotion_global is not None
                 and (model_conditions.emotion_global_present is None
                      or bool(model_conditions.emotion_global_present.any().item())))
                or (model_conditions.emotion_frame is not None
                    and model_conditions.emotion_frame_mask is not None
                    and bool(model_conditions.emotion_frame_mask.any().item()))
            )
        )
        has_prosody = (
            model_conditions is not None
            and model_conditions.prosody_frame is not None
            and model_conditions.prosody_mask is not None
            and bool(model_conditions.prosody_mask.any().item())
        )
        use_emo_cfg = emotion_cfg_strength > 0.01 and has_emotion
        use_pros_cfg = prosody_cfg_strength > 0.01 and has_prosody
        use_expression_cfg = use_emo_cfg or use_pros_cfg

        # Pre-build all conditioning runtimes ONCE before the ODE loop.
        cond_runtime = self._build_conditioning_runtime(model_conditions) if model_conditions is not None else None
        uncond_runtime = self._build_conditioning_runtime(uncond_conditions) if uncond_conditions is not None else None

        # no_emotion_runtime: drop emotion only, keep prosody.
        # Used for: (f − e) = emotion contribution, (e − v) = prosody contribution.
        no_emotion_runtime = (
            self._build_conditioning_runtime(model_conditions, drop_emotion=True)
            if use_expression_cfg and model_conditions is not None else None
        )
        # voice_runtime: drop emotion + prosody. Speaker identity only.
        # Only needed when prosody_cfg > 0 (to isolate prosody contribution).
        voice_runtime = (
            self._build_conditioning_runtime(model_conditions, drop_emotion=True, drop_prosody=True)
            if use_pros_cfg and model_conditions is not None else None
        )

        def fn(t, x):
            kw = dict(cond=step_cond, text=text, time=t, mask=mask)

            if cfg_strength < 1e-5 and not use_expression_cfg:
                return self.transformer(x=x, **kw, cache=True, conditioning_runtime=cond_runtime)

            if use_expression_cfg:
                # ── 4-branch EG-CFG ──
                # f = full, e = no_emotion, v = voice, u = uncond
                #
                # result = f + cfg·(f−u) + emo·(f−e) + pros·(e−v)
                #
                # Branches built:
                #   f always, u always, e if emo or pros, v if pros.

                f = self.transformer(
                    x=x, **kw, cache=False,
                    conditioning_runtime=cond_runtime,
                ).float()

                u = self.transformer(
                    x=x, **kw, cache=False,
                    drop_audio_cond=True, drop_text=True,
                    conditioning_runtime=(uncond_runtime if uncond_runtime is not None else cond_runtime),
                ).float()

                result = f + cfg_strength * (f - u)

                if use_emo_cfg or use_pros_cfg:
                    e = self.transformer(
                        x=x, **kw, cache=False,
                        conditioning_runtime=no_emotion_runtime,
                    ).float()

                    if use_emo_cfg:
                        result = result + emotion_cfg_strength * (f - e)

                    if use_pros_cfg:
                        v = self.transformer(
                            x=x, **kw, cache=False,
                            conditioning_runtime=voice_runtime,
                        ).float()
                        result = result + prosody_cfg_strength * (e - v)

                return result.to(x.dtype)

            p_cond = self.transformer(
                x=x, **kw, cache=True,
                conditioning_runtime=cond_runtime,
            )
            p_uncond = self.transformer(
                x=x, **kw, cache=False,
                drop_audio_cond=True, drop_text=True,
                conditioning_runtime=(uncond_runtime if uncond_runtime is not None else cond_runtime),
            )
            dtype_orig = p_cond.dtype
            return (p_cond.float() + (p_cond.float() - p_uncond.float()) * cfg_strength).to(dtype_orig)

        # Initial noise
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

        # Timesteps MUST be float64 — bf16/fp16 precision causes adjacent values
        # to collapse after sway transform → odeint crashes with "t must be
        # strictly increasing or decreasing".
        if t_start == 0 and use_epss:
            t = get_epss_timesteps(steps, device=self.device, dtype=torch.float64)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=torch.float64)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        out = torch.where(cond_mask.unsqueeze(-1), cond, trajectory[-1])
        if exists(vocoder):
            out = vocoder(out.permute(0, 2, 1))
        return out, trajectory

    # ── Training ──────────────────────────────────────────────────────

    def forward(
        self, inp, text, *, lens=None, noise_scheduler=None,
        conditions=None,
        prosody_raw=None, prosody_mask=None,
        text_byte_lens=None, speaker_emb_for_dur=None,
    ):
        """Training forward with multi-condition dropout.

        Returns: (flow_loss, cond, pred, mode_names, dur_loss)
            dur_loss is 0.0 tensor if duration_predictor is None or inputs missing.
        """
        inp = self._to_mel(inp)
        batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device
        text = self._to_text_ids(text, batch, device)

        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device, dtype=torch.long)
        else:
            lens = lens.to(device=device, dtype=torch.long)  # arange requires integer
        mask = lens_to_mask(lens, length=seq_len)

        frac_lengths = torch.zeros((batch,), device=device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)
        if exists(mask):
            rand_span_mask &= mask

        x1 = inp
        x0 = torch.randn_like(x1)
        time = torch.rand((batch,), dtype=dtype, device=device)
        t = time.unsqueeze(-1).unsqueeze(-1)
        phi = (1 - t) * x0 + t * x1
        flow = x1 - x0
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        dropout = self.sample_condition_dropout(batch, device)
        mode_names = dropout["mode_names"]

        model_conditions = conditions

        cond_out = self._build_conditioning_runtime(
            model_conditions,
            drop_speaker=dropout["drop_speaker"],
            drop_emotion=dropout["drop_emotion"],
            drop_prosody=dropout["drop_prosody"],
        )
        pred = self.transformer(
            x=phi, cond=cond, text=text, time=time, mask=mask,
            drop_audio_cond=dropout["drop_audio"], drop_text=dropout["drop_text"],
            conditioning_runtime=cond_out,
        )

        loss = F.mse_loss(pred, flow, reduction="none")
        masked = loss[rand_span_mask]

        # Duration predictor loss (parallel head, does not affect flow matching gradient)
        dur_loss = torch.tensor(0.0, device=device)
        if (self.duration_predictor is not None
                and prosody_raw is not None and text_byte_lens is not None):
            dur_loss = self.duration_predictor.compute_loss(
                prosody_raw=prosody_raw.to(device),
                prosody_mask=prosody_mask.to(device) if prosody_mask is not None else None,
                text_byte_lens=text_byte_lens.to(device),
                target_mel_frames=lens,
                speaker_emb=speaker_emb_for_dur.to(device) if speaker_emb_for_dur is not None else None,
            )

        if masked.numel() == 0:
            return (pred * 0.0).sum(), cond, pred, mode_names, dur_loss
        return masked.mean(), cond, pred, mode_names, dur_loss
