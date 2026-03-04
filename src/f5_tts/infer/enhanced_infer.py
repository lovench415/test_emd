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
from f5_tts.model.speaker_encoder import SpeakerEncoder
from f5_tts.model.emotion_encoder import EmotionEncoder
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

    # --- Build model config ---
    if model_cfg is None:
        model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
            text_mask_padding=False, conv_layers=4, pe_attn_head=1,
            attn_backend="torch", attn_mask_enabled=False,
        )

    model_cfg.update(
        speaker_emb_dim=speaker_emb_dim, emotion_emb_dim=emotion_emb_dim,
        speaker_raw_dim=speaker_raw_dim, emotion_raw_dim=emotion_raw_dim,
    )

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
    ).to(device_str)

    # Prefer bfloat16 (wider dynamic range, avoids overflow in attention/norm)
    if "cuda" in device_str and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif "cuda" in device_str:
        dtype = torch.float16
    else:
        dtype = torch.float32
    model = model.to(dtype)

    state = _pick_state_dict(checkpoint)

    for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
        state.pop(key, None)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"Loaded {len(state) - len(unexpected)} params | "
        f"Missing: {len(missing)} | "
        f"Unexpected: {len(unexpected)}"
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
) -> dict[str, torch.Tensor]:
    """Extract raw speaker + emotion embeddings from any-language reference.

    Returns RAW (un-projected) embeddings — the ConditioningAggregator inside
    the model handles projection from raw→target dim for consistency with
    the training pipeline.
    """
    audio, sr = torchaudio.load(ref_audio_path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.to(device_str)

    spk_raw = speaker_encoder.extract_raw(audio, sr=sr)
    emo_global_raw, emo_frame_raw = emotion_encoder.extract_raw(audio, sr=sr)
    return {
        "speaker_emb": spk_raw,
        "emotion_global": emo_global_raw,
        "emotion_frame": emo_frame_raw,
    }


# ── Main inference function ──────────────────────────────────────────

def infer_enhanced(
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    model: EnhancedCFM,
    vocoder,
    speaker_encoder: SpeakerEncoder,
    emotion_encoder: EmotionEncoder,
    nfe_step: int = 32,
    cfg_strength: float = 2.0,
    emotion_cfg_strength: float = 1.0,
    sway_sampling_coef: float = -1.0,
    speed: float = 1.0,
    fix_duration: float | None = None,
    seed: int | None = None,
    target_rms: float = 0.1,
    show_info=print,
    device_str: str = device,
) -> tuple[np.ndarray, int]:
    """Full enhanced inference: reference audio → generated speech."""
    ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(
        ref_audio, ref_text, show_info=show_info,
    )

    show_info("Extracting embeddings from reference...")
    embs = extract_reference_embeddings(
        ref_audio_processed, speaker_encoder, emotion_encoder, device_str,
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
    ref_bytes = max(len(ref_text_processed.encode("utf-8")), 1)
    max_chars = int(ref_bytes / audio_dur * max(22 - audio_dur, 1) * speed)
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)

    show_info(f"Generating {len(gen_text_batches)} chunk(s), emo_cfg={emotion_cfg_strength}")

    waves = []
    for batch_text in gen_text_batches:
        local_speed = 0.3 if len(batch_text.encode("utf-8")) < 10 else speed
        text_list = [ref_text_processed + batch_text]
        final_text = convert_char_to_pinyin(text_list)

        ref_len = audio.shape[-1] // hop_length
        if fix_duration is not None:
            duration = int(fix_duration * target_sample_rate / hop_length)
        else:
            ref_text_len = max(len(ref_text_processed.encode("utf-8")), 1)
            gen_text_len = len(batch_text.encode("utf-8"))
            duration = ref_len + int(ref_len / ref_text_len * gen_text_len / local_speed)

        with torch.inference_mode():
            gen, _ = model.sample(
                cond=audio, text=final_text, duration=duration,
                steps=nfe_step, cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef, seed=seed,
                speaker_emb=embs["speaker_emb"],
                emotion_global=embs["emotion_global"],
                emotion_frame=embs.get("emotion_frame"),
                emotion_cfg_strength=emotion_cfg_strength,
            )
            gen = gen.to(torch.float32)[:, ref_len:, :].permute(0, 2, 1)

            if mel_spec_type == "vocos":
                wave = vocoder.decode(gen)
            else:
                wave = vocoder(gen)

            if rms < target_rms:
                wave = wave * rms / target_rms
            waves.append(wave.squeeze().cpu().numpy())

    return crossfade_waves(waves, target_sample_rate, cross_fade_duration), target_sample_rate


# ── Quick inference (all-in-one) ─────────────────────────────────────

def quick_inference(
    ref_audio_path: str,
    gen_text: str,
    ckpt_path: str,
    ref_text: str = "",
    emotion_cfg_strength: float = 1.0,
    output_path: str = "output.wav",
    device_str: str = device,
):
    """All-in-one: loads models, extracts embeddings, generates, saves."""
    import soundfile as sf

    vocoder = load_vocoder(device=device_str)
    model = load_enhanced_model(ckpt_path, device_str=device_str)
    spk_enc, emo_enc = load_embedding_extractors(device_str=device_str)

    wave, sr = infer_enhanced(
        ref_audio=ref_audio_path, ref_text=ref_text, gen_text=gen_text,
        model=model, vocoder=vocoder,
        speaker_encoder=spk_enc, emotion_encoder=emo_enc,
        emotion_cfg_strength=emotion_cfg_strength, device_str=device_str,
    )
    sf.write(output_path, wave, sr)
    print(f"Saved to {output_path}")
    return wave, sr
