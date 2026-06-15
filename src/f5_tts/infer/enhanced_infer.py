"""
Enhanced Inference Pipeline for F5-TTS.

Extends original with:
  - Speaker/emotion embedding extraction from reference audio
  - Emotion-Guided CFG for controlling emotion intensity
  - Cross-lingual voice cloning
"""

from __future__ import annotations

import os
from importlib.resources import files

import numpy as np
import torch
import torchaudio

from f5_tts.model.backbones.enhanced_dit import EnhancedDiT
from f5_tts.model.enhanced_cfm import EnhancedCFM
from f5_tts.model.condition_types import RawConditionBatch
from f5_tts.model.speaker_encoder import SpeakerEncoder
from f5_tts.model.emotion_encoder import EmotionEncoder
from f5_tts.model.prosody_encoder import ProsodyEncoder, PROSODY_RAW_DIM
from f5_tts.model.encoder_utils import SPEAKER_RAW_DIMS, EMOTION_RAW_DIMS
from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer

from f5_tts.infer.utils_infer import (
    load_vocoder,
    preprocess_ref_audio_text,
    chunk_text,
    target_sample_rate,
    n_mel_channels,
    hop_length,
    win_length,
    n_fft,
    mel_spec_type,
    target_rms,
    cross_fade_duration,
    ode_method,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    device,
)


# ── Audio utilities ───────────────────────────────────────────────────

def _safe_rms(audio: torch.Tensor) -> float:
    """Compute RMS, clamped away from zero."""
    return max(torch.sqrt(torch.mean(torch.square(audio))).item(), 1e-8)


def crossfade_waves(waves: list[np.ndarray], sr: int, fade_sec: float = 0.015) -> np.ndarray:
    """Concatenate waveforms with cross-fade."""
    if len(waves) == 1:
        return waves[0]

    cf_samples = int(fade_sec * sr)
    result = waves[0]
    for w in waves[1:]:
        cf = min(cf_samples, len(result), len(w))
        if cf <= 0:
            result = np.concatenate([result, w])
        else:
            fade_out = np.linspace(1, 0, cf)
            fade_in = np.linspace(0, 1, cf)
            xfade = result[-cf:] * fade_out + w[:cf] * fade_in
            result = np.concatenate([result[:-cf], xfade, w[cf:]])
    return result


# ── Load embedding extractors ────────────────────────────────────────

def load_embedding_extractors(
    speaker_backend: str = "wavlm_sv",
    emotion_backend: str = "emotion2vec_base",
    speaker_dim: int = 512,
    emotion_dim: int = 512,
    device_str: str = device,
) -> tuple[SpeakerEncoder, EmotionEncoder]:
    """Load frozen pretrained speaker + emotion encoders."""
    print(f"Loading speaker encoder ({speaker_backend})...")
    spk = SpeakerEncoder(backend=speaker_backend, output_dim=speaker_dim,
                         device=device_str).to(device_str).eval()

    print(f"Loading emotion encoder ({emotion_backend})...")
    emo = EmotionEncoder(backend=emotion_backend, output_dim=emotion_dim,
                         frame_level=True, device=device_str).to(device_str).eval()
    return spk, emo


def load_prosody_encoder(
    backend: str = "dio",
    output_dim: int = 256,
    device_str: str = device,
    rmvpe_model_path: str | None = None,
    prosody_stats_path: str | None = None,
) -> ProsodyEncoder:
    """Load prosody encoder for F0/energy extraction.

    prosody_stats_path: path to the prosody_stats.json written by prepare_data.
    These fixed normalization stats MUST match the ones used during
    preprocessing/training, otherwise the prosody conditioning signal differs
    between train and inference. If None or missing, the encoder's speech
    defaults are used (correct only if training also used the defaults).
    """
    print(f"Loading prosody encoder ({backend})...")
    stat_kwargs = {}
    if prosody_stats_path is not None and os.path.exists(prosody_stats_path):
        import json as _json
        with open(prosody_stats_path) as f:
            st = _json.load(f)
        if st.get("f0_norm_mean") is not None:
            stat_kwargs = dict(
                f0_norm_mean=st["f0_norm_mean"], f0_norm_std=st["f0_norm_std"],
                energy_norm_mean=st["energy_norm_mean"], energy_norm_std=st["energy_norm_std"],
            )
            print(f"  Using corpus prosody stats from {prosody_stats_path}")
    else:
        if prosody_stats_path is not None:
            print(f"  ⚠ prosody_stats not found at {prosody_stats_path}; using defaults "
                  f"(must match training, or prosody conditioning will be miscalibrated)")
    return ProsodyEncoder(
        backend=backend, output_dim=output_dim, device=device_str,
        rmvpe_model_path=rmvpe_model_path, **stat_kwargs,
    ).to(device_str).eval()


# ── Load enhanced model ──────────────────────────────────────────────

def load_enhanced_model(
    ckpt_path: str,
    model_cfg: dict | None = None,
    vocab_file: str = "",
    device_str: str = device,
    use_ema: bool = True,
    speaker_emb_dim: int = 512,
    emotion_emb_dim: int = 512,
    speaker_raw_dim: int | None = None,
    emotion_raw_dim: int | None = None,
    prosody_dim: int = 256,
    prosody_raw_dim: int | None = None,
):
    """Load enhanced F5-TTS model.

    Raw→target projection layers MUST exist at inference time when reference encoders
    return raw embeddings (e.g., 768-dim). If raw dims are not provided, we infer them
    from checkpoint weights before instantiating the model to avoid silently dropping
    projection weights as 'unexpected'.
    """
    if not vocab_file:
        vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, "custom")

    # --- Load checkpoint first (CPU) to infer raw dims if needed ---
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        checkpoint = load_file(ckpt_path, device="cpu")
    else:
        checkpoint = torch.load(ckpt_path, map_location="cpu")

    def _pick_state_dict(ckpt):
        if isinstance(ckpt, dict):
            if use_ema and "ema_model_state_dict" in ckpt:
                sd = ckpt["ema_model_state_dict"]
                if isinstance(sd, dict):
                    return {k.replace("ema_model.", ""): v for k, v in sd.items() if k not in ("initted", "step")}
            if "model_state_dict" in ckpt:
                return ckpt["model_state_dict"]
        return ckpt

    state_for_shapes = _pick_state_dict(checkpoint)

    def _infer_in_features(key_substr: str):
        for k, v in state_for_shapes.items():
            if key_substr in k and k.endswith("weight") and hasattr(v, "ndim") and v.ndim == 2:
                return int(v.shape[1])
        return None

    if speaker_raw_dim is None:
        speaker_raw_dim = _infer_in_features("speaker_raw_proj")
    if emotion_raw_dim is None:
        emotion_raw_dim = _infer_in_features("emotion_raw_proj") or _infer_in_features("frame_raw_proj")
    if prosody_raw_dim is None:
        prosody_raw_dim = _infer_in_features("prosody_raw_proj")

    # Detect whether checkpoint has prosody layers
    has_prosody = any("prosody" in k for k in state_for_shapes)
    # Detect whether checkpoint has duration predictor
    has_dur_pred = any("duration_predictor" in k for k in state_for_shapes)

    # --- Build model config ---
    if model_cfg is None:
        model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
            text_mask_padding=False, conv_layers=4, pe_attn_head=1,
            attn_backend="torch", attn_mask_enabled=False,
        )
    else:
        # Copy: the .update() below would otherwise mutate the caller's dict
        # in place, leaking speaker/emotion/prosody dims into it and corrupting
        # a second load_enhanced_model() call that reuses the same dict.
        model_cfg = dict(model_cfg)

    model_cfg.update(
        speaker_emb_dim=speaker_emb_dim, emotion_emb_dim=emotion_emb_dim,
        speaker_raw_dim=speaker_raw_dim, emotion_raw_dim=emotion_raw_dim,
        prosody_dim=prosody_dim, prosody_raw_dim=prosody_raw_dim,
        use_prosody_cross_attn=has_prosody,
    )

    # If the checkpoint persisted its architecture (fusion_mode, bottleneck,
    # prosody paths, dims), apply it so the model is rebuilt IDENTICALLY to
    # training. Without this, a model trained with non-default architecture
    # (e.g. fusion_mode="residual" or adaln_bottleneck_dim=512) would be rebuilt
    # with defaults → conditioning weights silently dropped → broken cloning.
    # Explicit kwargs above still win only for the dims we already inferred.
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("arch_config"), dict):
        ac = checkpoint["arch_config"]
        for k in ("fusion_mode", "adaln_bottleneck_dim", "use_adaln_cond",
                  "use_input_add_cond", "use_cross_attn_cond",
                  "use_prosody_cross_attn", "use_prosody_direct", "cross_attn_gate_floor",
                  "speaker_emb_dim", "emotion_emb_dim", "prosody_dim"):
            if k in ac and ac[k] is not None:
                model_cfg[k] = ac[k]
        # raw dims: prefer checkpoint arch_config, fall back to inferred
        for k in ("speaker_raw_dim", "emotion_raw_dim", "prosody_raw_dim"):
            if ac.get(k) is not None:
                model_cfg[k] = ac[k]
        print(f"  Loaded architecture from checkpoint: fusion_mode={ac.get('fusion_mode')}, "
              f"bottleneck={ac.get('adaln_bottleneck_dim')}, "
              f"prosody_direct={ac.get('use_prosody_direct')}, "
              f"prosody_xattn={ac.get('use_prosody_cross_attn')}")

    model = EnhancedCFM(
        transformer=EnhancedDiT(
            **model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels,
        ),
        mel_spec_kwargs=dict(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            n_mel_channels=n_mel_channels, target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(method=ode_method),
        vocab_char_map=vocab_char_map,
        use_duration_predictor=has_dur_pred,
        speaker_emb_dim=speaker_emb_dim,
    ).to(device_str)

    # Keep model in float32 for inference — bf16 causes accumulated ODE error
    # over 32 steps × 22 blocks.  Training uses mixed precision via Accelerate,
    # but inference needs full precision for the mel spec + ODE integrator.
    model = model.to(torch.float32)

    state = _pick_state_dict(checkpoint)

    for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
        state.pop(key, None)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"Loaded {len(state) - len(unexpected)} params | "
        f"Missing: {len(missing)} | "
        f"Unexpected: {len(unexpected)}"
    )

    # strict=False is intentional (EMA wrappers, optional prosody/duration heads),
    # but it can silently drop or skip conditioning weights, which then run with
    # random-initialized values and produce garbage audio with NO error. The most
    # common cause is a raw-projection dim that _infer_in_features failed to detect
    # (slightly different key name): the projection module is then not built, the
    # checkpoint's projection weights land in `unexpected`, and project_*() returns
    # the un-projected raw embedding. Surface that explicitly.
    _critical = (
        "speaker_raw_proj", "emotion_raw_proj", "frame_raw_proj", "prosody_raw_proj",
        "cross_attns", "prosody_cross_attns", "adaln_cond", "input_add",
    )

    def _hits(names, keys):
        return sorted({n for n in names for k in keys if n in k})

    # Conditioning weights present in the checkpoint but dropped (module not built).
    dropped = _hits(_critical, unexpected)
    # Conditioning weights the model expects but the checkpoint did not provide
    # (left random-initialized).
    uninit = _hits(_critical, missing)
    if dropped:
        print(
            "  ⚠ WARNING: conditioning/projection weights in the checkpoint were "
            f"DROPPED as unexpected: {dropped}. The corresponding modules were not "
            "built — likely a raw-projection input dim was not detected. These signals "
            "will be UN-projected/ignored at inference. Pass the correct *_raw_dim "
            "explicitly to load_enhanced_model()."
        )
    if uninit:
        print(
            "  ⚠ WARNING: conditioning/projection modules are present in the model but "
            f"MISSING from the checkpoint (random-initialized): {uninit}. Inference "
            "output for these signals will be garbage."
        )

    del checkpoint
    if "cuda" in device_str:
        torch.cuda.empty_cache()
    return model.to(device_str).eval()



# ── Extract embeddings from reference ─────────────────────────────────

@torch.no_grad()
def extract_reference_embeddings(
    ref_audio_path: str,
    speaker_encoder: SpeakerEncoder,
    emotion_encoder: EmotionEncoder,
    device_str: str = device,
    prosody_encoder: ProsodyEncoder | None = None,
    use_emotion_frame: bool = True,   # set False to NOT feed emotion FRAME cross-attn
                                      # (emotion2vec frames carry phonetics → leak
                                      # reference content for out-of-distribution,
                                      # e.g. cross-language, references). Global
                                      # emotion is kept (phonetics averaged out).
    use_emotion: bool = True,         # False → drop emotion entirely (global+frame)
    use_prosody: bool = True,         # False → drop prosody conditioning
) -> RawConditionBatch:
    """Extract raw speaker + emotion + prosody embeddings from any-language reference.

    Returns RAW (un-projected) embeddings — the ConditioningAggregator inside
    the model handles projection from raw→target dim for consistency with
    the training pipeline.

    For out-of-distribution references (different language from training), the
    emotion FRAME path is the main content-leak vector: emotion2vec frame
    embeddings encode phonetics, and the position-agnostic cross-attention can
    reorder/copy them into the output (heard as reference leakage and sentence
    reordering). use_emotion_frame=False keeps speaker + global emotion +
    prosody while removing that vector.
    """
    audio, sr = torchaudio.load(ref_audio_path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.to(device_str)

    # Speaker embedding for a single, un-padded reference. This is the
    # "per_sample" regime (no padding, no attention_mask). It matches a training
    # cache built with SPEAKER_EMB_MODE=per_sample (and the original REA
    # pipeline). If the cache was built with SPEAKER_EMB_MODE=masked (batched +
    # attention_mask), the training and inference embeddings come from slightly
    # different WavLM-SV regimes → a distribution shift that can weaken cloning.
    # Rule of thumb: build the cache with per_sample to match this path exactly.
    spk_raw = speaker_encoder.extract_raw(audio, sr=sr)
    if use_emotion and hasattr(emotion_encoder, "extract_raw_with_mask"):
        emo_global_raw, emo_frame_raw, emo_frame_mask = emotion_encoder.extract_raw_with_mask(audio, sr=sr)
    elif use_emotion:
        emo_global_raw, emo_frame_raw = emotion_encoder.extract_raw(audio, sr=sr)
        emo_frame_mask = (
            torch.ones(emo_frame_raw.shape[:2], dtype=torch.bool, device=emo_frame_raw.device)
            if emo_frame_raw is not None else None
        )
    else:
        emo_global_raw = emo_frame_raw = emo_frame_mask = None

    # Drop only the FRAME path (keep global emotion as a phonetics-free summary).
    if not use_emotion_frame:
        emo_frame_raw = None
        emo_frame_mask = None

    # Prosody extraction (F0 + energy + voicing)
    prosody_raw = None
    prosody_mask = None
    if prosody_encoder is not None and use_prosody:
        prosody_raw, prosody_mask = prosody_encoder.extract_raw(audio, sr=sr)

    return RawConditionBatch(
        speaker_raw=spk_raw,
        emotion_global_raw=emo_global_raw,
        emotion_frame_raw=emo_frame_raw,
        prosody_raw=prosody_raw,
        speaker_present=(
            torch.ones(spk_raw.shape[0], dtype=torch.bool, device=spk_raw.device)
            if spk_raw is not None else None
        ),
        emotion_global_present=(
            torch.ones(emo_global_raw.shape[0], dtype=torch.bool, device=emo_global_raw.device)
            if emo_global_raw is not None else None
        ),
        emotion_frame_mask=emo_frame_mask,
        prosody_mask=prosody_mask,
    )


# ── Main inference function ──────────────────────────────────────────

def infer_enhanced(
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    model: EnhancedCFM,
    vocoder,
    speaker_encoder: SpeakerEncoder,
    emotion_encoder: EmotionEncoder,
    prosody_encoder: ProsodyEncoder | None = None,
    nfe_step: int = 32,
    cfg_strength: float = 2.0,
    emotion_cfg_strength: float = 0.5,
    prosody_cfg_strength: float = 0.0,
    sway_sampling_coef: float = -1.0,
    speed: float = 1.0,
    fix_duration: float | None = None,
    seed: int | None = None,
    target_rms: float = 0.1,
    show_info=print,
    device_str: str = device,
    use_emotion_frame: bool = True,   # False recommended for cross-language refs
    use_emotion: bool = True,
    use_prosody: bool = True,
    no_ref_audio: bool = False,       # DIAGNOSTIC/clean-speaker mode: zero out the
                                      # reference MEL prefix. F5-TTS does in-context
                                      # continuation from the ref mel; with an
                                      # out-of-distribution (e.g. cross-language)
                                      # reference the model continues that foreign
                                      # acoustic context → leakage that is NOT fixed
                                      # by dropping emotion/prosody. With no_ref_audio
                                      # the voice comes only from the speaker
                                      # embedding path (weaker cloning, but no mel
                                      # in-context leak). If leakage vanishes here,
                                      # the ref-mel prefix is the cause.
    ignore_ref_text: bool = False,    # DIAGNOSTIC: drop ref_text from the text
                                      # prefix (use only gen_text). F5-TTS feeds
                                      # text as [ref_text + gen_text]; an
                                      # out-of-distribution ref_text (foreign
                                      # language) biases the model to synthesize
                                      # foreign acoustics even when the mel prefix
                                      # is zeroed. NOTE: F5-TTS is trained WITH a
                                      # ref_text prefix, so removing it is
                                      # off-distribution and may reduce quality —
                                      # this is primarily to localize the leak.
                                      # When set, duration ignores the ref/gen text
                                      # ratio (uses fix_duration or a gen-only
                                      # estimate) to avoid the divide-by-tiny blowup.
    trim_leading_silence: bool = False,  # strip leading near-silence the model
                                         # may emit by continuing the reference's
                                         # trailing silence (keeps a 20ms lead-in).
    duration_scale: float = 1.0,         # multiply the duration-predictor output
                                         # (use <1, e.g. 0.85, if it runs long and
                                         # you hear padded/leaked tails). Ignored
                                         # when fix_duration is set.
) -> tuple[np.ndarray, int]:
    """Full enhanced inference: reference audio → generated speech."""
    ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(
        ref_audio, ref_text, show_info=show_info,
    )

    show_info("Extracting embeddings from reference...")
    embs = extract_reference_embeddings(
        ref_audio_processed, speaker_encoder, emotion_encoder, device_str,
        prosody_encoder=prosody_encoder,
        use_emotion_frame=use_emotion_frame,
        use_emotion=use_emotion,
        use_prosody=use_prosody,
    )

    audio, sr = torchaudio.load(ref_audio_processed)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    rms = _safe_rms(audio)
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        audio = torchaudio.transforms.Resample(sr, target_sample_rate)(audio)
    audio = audio.to(device_str)

    # Chunk text
    audio_dur = max(audio.shape[-1] / target_sample_rate, 0.1)
    # Chars, not UTF-8 bytes — byte counts skew across scripts (Cyrillic=2B/char),
    # inflating max_chars for Latin gen_text after a Cyrillic reference.
    ref_chars = max(len(ref_text_processed), 1)
    max_chars = int(ref_chars / audio_dur * max(22 - audio_dur, 1) * speed)
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)

    show_info(f"Generating {len(gen_text_batches)} chunk(s), emo_cfg={emotion_cfg_strength}, pros_cfg={prosody_cfg_strength}")

    waves = []
    for batch_text in gen_text_batches:
        local_speed = 0.3 if len(batch_text) < 10 else speed  # chars not bytes: script-neutral short-text threshold
        eff_ref_text = "" if ignore_ref_text else ref_text_processed
        text_list = [eff_ref_text + batch_text]
        final_text = convert_char_to_pinyin(text_list)

        ref_len = audio.shape[-1] // hop_length
        if fix_duration is not None:
            # fix_duration is the desired GENERATED speech length (seconds).
            # The reference prefix occupies ref_len frames and is trimmed off the
            # output afterwards, so it must be ADDED on top — otherwise the trim
            # eats most of fix_duration (e.g. fix_duration=4s with a 3s reference
            # left only ~1s of speech). When the ref prefix is dropped entirely
            # (no_ref_audio+ignore_ref_text) nothing is trimmed, so add nothing.
            gen_frames = int(fix_duration * target_sample_rate / hop_length)
            duration = (0 if (ignore_ref_text and no_ref_audio) else ref_len) + gen_frames
        elif getattr(model, "duration_predictor", None) is not None and embs.prosody_raw is not None:
            # Trained duration predictor, conditioned on REFERENCE prosody. Its
            # rate (frames/char) is derived from the reference clip, so pauses or
            # slow/trailing speech in the reference inflate the rate → an
            # over-long duration. F5-TTS fills the whole duration, so the excess
            # is padded/filled with reference-like content → audible leakage.
            # Two guards:
            #   • duration_scale: global multiplier to correct systematic bias
            #     (e.g. 0.85 if the predictor consistently runs ~15% long).
            #   • a hard CAP from a plain chars→frames estimate, so the predicted
            #     length can never exceed what the text could plausibly need.
            gen_text_chars = len(batch_text)  # chars, not bytes — must match training units
            spk_emb = embs.speaker_raw  # (1, speaker_dim) raw speaker embedding
            pred_gen = model.duration_predictor.predict_duration(
                prosody_raw=embs.prosody_raw,
                prosody_mask=embs.prosody_mask,
                text_byte_len=gen_text_chars,  # param name is historical; unit is CHARS now
                speaker_emb=spk_emb,
            )
            pred_gen = int(pred_gen * duration_scale)
            # Upper cap: nominal speaking rate ~14 chars/sec → frames/char ≈
            # (target_sample_rate/hop)/14. Allow generous headroom (×1.5) so we
            # only clip pathological over-predictions, not natural variation.
            fps = target_sample_rate / hop_length
            cap = int(gen_text_chars * (fps / 14.0) * 1.5 / max(local_speed, 1e-3))
            pred_gen = min(pred_gen, cap)
            duration = ref_len + pred_gen  # predictor returns gen frames, add ref prefix
        elif ignore_ref_text:
            # No ref_text in the prefix → there is no text↔mel reference pair to
            # continue from. Estimate generation length from gen_text alone.
            gen_text_len = len(batch_text)
            FRAMES_PER_CHAR = 8  # ~ conservative; tune via fix_duration for precision
            gen_frames = int(gen_text_len * FRAMES_PER_CHAR / local_speed)
            # When the mel prefix is ALSO dropped (no_ref_audio), there is no
            # reference region at all → do NOT reserve/trim ref_len, otherwise the
            # first ref_len frames of REAL generated speech get cut (heard as the
            # first sentence disappearing). Keep ref_len only if the mel prefix is
            # still present.
            duration = (ref_len if not no_ref_audio else 0) + gen_frames
        else:
            # Fallback: linear formula.
            # Count Unicode CHARS, not UTF-8 bytes: byte counts skew the ratio
            # across scripts (Cyrillic=2 bytes/char, Latin=1), so a Russian
            # reference + English gen_text would halve the estimated duration
            # (sped-up/truncated speech) and vice versa. Character count is a
            # script-neutral proxy for spoken length.
            ref_text_len = max(len(ref_text_processed), 1)
            gen_text_len = len(batch_text)
            duration = ref_len + int(ref_len / ref_text_len * gen_text_len / local_speed)

        with torch.inference_mode():
            # Effective reference-prefix length. When BOTH text and mel prefixes
            # are removed there is no reference region: cond_mask must be empty and
            # the output must NOT be trimmed (else real generated speech is cut).
            eff_ref_len = 0 if (ignore_ref_text and no_ref_audio) else ref_len
            model_conditions = None
            if getattr(model, "condition_lifecycle", None) is not None:
                lens = torch.tensor([eff_ref_len], device=device_str, dtype=torch.long)
                text_ids = model._to_text_ids(final_text, 1, device_str)
                duration_t = torch.tensor([duration], device=device_str, dtype=torch.long)
                target_len = int(torch.maximum(torch.maximum((text_ids != -1).sum(dim=-1), lens) + 1, duration_t).clamp(max=65536).amax().item())
                model_conditions = model.condition_lifecycle.prepare(raw_conditions=embs, target_len=target_len)
                # NOTE: prosody_direct ref-region masking is handled inside
                # sample() via cond_mask — no truncation needed here.
            gen, _ = model.sample(
                cond=audio, text=final_text, duration=duration,
                steps=nfe_step, cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef, seed=seed,
                conditions=model_conditions,
                emotion_cfg_strength=emotion_cfg_strength,
                prosody_cfg_strength=prosody_cfg_strength,
                no_ref_audio=no_ref_audio,
                lens=(torch.tensor([eff_ref_len], device=device_str, dtype=torch.long)
                      if eff_ref_len != ref_len else None),
            )
            gen = gen.to(torch.float32)[:, eff_ref_len:, :].permute(0, 2, 1)

            if mel_spec_type == "vocos":
                wave = vocoder.decode(gen)
            else:
                wave = vocoder(gen)

            if rms < target_rms:
                wave = wave * rms / target_rms
            w = wave.squeeze().cpu().numpy()

            # Optional: strip leading near-silence. The model often continues the
            # reference's trailing silence, so generated speech can begin with a
            # short zero/quiet pad. trim_leading_silence removes samples before the
            # first frame whose amplitude exceeds a small threshold, leaving a tiny
            # lead-in so onsets aren't clipped.
            if trim_leading_silence:
                import numpy as _np
                amp = _np.abs(w)
                thr = max(float(amp.max()) * 0.02, 1e-4)  # 2% of peak
                above = _np.where(amp > thr)[0]
                if len(above) > 0:
                    keep = max(above[0] - int(0.02 * target_sample_rate), 0)  # 20ms lead-in
                    w = w[keep:]
            waves.append(w)

    return crossfade_waves(waves, target_sample_rate, cross_fade_duration), target_sample_rate


# ── Quick inference (all-in-one) ─────────────────────────────────────

def quick_inference(
    ref_audio_path: str,
    gen_text: str,
    ckpt_path: str,
    ref_text: str = "",
    emotion_cfg_strength: float = 0.5,
    prosody_cfg_strength: float = 0.0,
    prosody_backend: str = "dio",
    output_path: str = "output.wav",
    device_str: str = device,
):
    """All-in-one: loads models, extracts embeddings, generates, saves."""
    import soundfile as sf

    vocoder = load_vocoder(device=device_str)
    model = load_enhanced_model(ckpt_path, device_str=device_str)
    spk_enc, emo_enc = load_embedding_extractors(device_str=device_str)

    # Load prosody encoder if model has prosody layers
    has_prosody = hasattr(model.transformer, 'cond_aggregator') and hasattr(model.transformer.cond_aggregator, 'prosody_cross_attns')
    prosody_enc = load_prosody_encoder(backend=prosody_backend, device_str=device_str) if has_prosody else None

    wave, sr = infer_enhanced(
        ref_audio=ref_audio_path, ref_text=ref_text, gen_text=gen_text,
        model=model, vocoder=vocoder,
        speaker_encoder=spk_enc, emotion_encoder=emo_enc,
        prosody_encoder=prosody_enc,
        emotion_cfg_strength=emotion_cfg_strength,
        prosody_cfg_strength=prosody_cfg_strength,
        device_str=device_str,
    )
    sf.write(output_path, wave, sr)
    print(f"Saved to {output_path}")
    return wave, sr
