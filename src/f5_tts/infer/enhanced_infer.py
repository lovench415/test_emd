"""
Enhanced Inference Pipeline for F5-TTS
========================================

Extends the original F5-TTS inference with:
1. Speaker embedding extraction from reference audio (any language)
2. Emotion embedding extraction from reference audio (any language)
3. Emotion-Guided CFG for controlling emotion intensity
4. Cross-lingual voice cloning: reference in any language -> Russian output

Usage:
    model = load_enhanced_model(...)
    speaker_enc, emotion_enc = load_embedding_extractors(device)
    
    audio = infer_enhanced(
        ref_audio="reference.wav",       # any language
        gen_text="Привет, как дела?",     # Russian text
        model=model,
        vocoder=vocoder,
        speaker_encoder=speaker_enc,
        emotion_encoder=emotion_enc,
        emotion_cfg_strength=1.5,         # control emotion intensity
    )
"""

from __future__ import annotations

import os
import tempfile
from importlib.resources import files

import numpy as np
import torch
import torchaudio
from pydub import AudioSegment, silence

from f5_tts.model.backbones.enhanced_dit import EnhancedDiT
from f5_tts.model.enhanced_cfm import EnhancedCFM
from f5_tts.model.speaker_encoder import SpeakerEncoder
from f5_tts.model.emotion_encoder import EmotionEncoder
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


# ------------------------------------------------------------------
# Load embedding extractors
# ------------------------------------------------------------------

def load_embedding_extractors(
    speaker_backend: str = "wavlm_sv",
    emotion_backend: str = "emotion2vec_base",
    speaker_dim: int = 512,
    emotion_dim: int = 512,
    device_str: str = device,
) -> tuple[SpeakerEncoder, EmotionEncoder]:
    """
    Load pretrained speaker and emotion embedding extractors.
    These are frozen models used only for inference-time feature extraction.
    """
    print(f"Loading speaker encoder ({speaker_backend})...")
    speaker_enc = SpeakerEncoder(
        backend=speaker_backend,
        output_dim=speaker_dim,
        device=device_str,
    ).to(device_str)
    speaker_enc.eval()

    print(f"Loading emotion encoder ({emotion_backend})...")
    emotion_enc = EmotionEncoder(
        backend=emotion_backend,
        output_dim=emotion_dim,
        frame_level=True,
        device=device_str,
    ).to(device_str)
    emotion_enc.eval()

    return speaker_enc, emotion_enc


# ------------------------------------------------------------------
# Load enhanced model
# ------------------------------------------------------------------

def load_enhanced_model(
    ckpt_path: str,
    model_cfg: dict | None = None,
    vocab_file: str = "",
    device_str: str = device,
    use_ema: bool = True,
    speaker_emb_dim: int = 512,
    emotion_emb_dim: int = 512,
):
    """
    Load the enhanced F5-TTS model with conditioning support.
    
    Loads original F5-TTS weights with strict=False, so new conditioning
    parameters are initialized from scratch (zero-initialized).
    """
    if vocab_file == "":
        vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, "custom")

    if model_cfg is None:
        model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            text_mask_padding=False,
            conv_layers=4,
            pe_attn_head=1,
            attn_backend="torch",
            attn_mask_enabled=False,
        )

    # Add conditioning params
    model_cfg["speaker_emb_dim"] = speaker_emb_dim
    model_cfg["emotion_emb_dim"] = emotion_emb_dim

    model = EnhancedCFM(
        transformer=EnhancedDiT(
            **model_cfg,
            text_num_embeds=vocab_size,
            mel_dim=n_mel_channels,
        ),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(method=ode_method),
        vocab_char_map=vocab_char_map,
    ).to(device_str)

    # Load checkpoint (strict=False to skip new conditioning params)
    dtype = torch.float16 if "cuda" in device_str else torch.float32
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file
        checkpoint = load_file(ckpt_path, device=device_str)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device_str, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        model_state = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }
    else:
        if ckpt_type == "safetensors":
            model_state = checkpoint
        else:
            model_state = checkpoint.get("model_state_dict", checkpoint)

    # Remove mel_spec keys that may cause issues
    for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
        model_state.pop(key, None)

    # Load with strict=False — new params (cond_aggregator, etc.) stay randomly init
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    
    n_new = len(missing)
    n_loaded = len(model_state) - len(unexpected)
    print(f"Loaded {n_loaded} params from checkpoint")
    print(f"New conditioning params (randomly initialized): {n_new}")
    if unexpected:
        print(f"Unexpected keys (skipped): {len(unexpected)}")

    del checkpoint
    torch.cuda.empty_cache()

    return model.to(device_str).eval()


# ------------------------------------------------------------------
# Extract embeddings from reference audio
# ------------------------------------------------------------------

@torch.no_grad()
def extract_reference_embeddings(
    ref_audio_path: str,
    speaker_encoder: SpeakerEncoder,
    emotion_encoder: EmotionEncoder,
    device_str: str = device,
) -> dict[str, torch.Tensor]:
    """
    Extract speaker and emotion embeddings from reference audio.
    Works with any language — embeddings are language-agnostic.
    
    Returns dict with:
        speaker_emb:    (1, speaker_dim)
        emotion_global: (1, emotion_dim)
        emotion_frame:  (1, T, emotion_dim)
    """
    audio, sr = torchaudio.load(ref_audio_path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    audio = audio.to(device_str)

    # Speaker embedding
    spk_emb = speaker_encoder(audio, sr=sr)  # (1, speaker_dim)

    # Emotion embedding
    emo_result = emotion_encoder(audio, sr=sr)  # dict with 'global' and 'frame'

    return {
        "speaker_emb": spk_emb,
        "emotion_global": emo_result["global"],
        "emotion_frame": emo_result.get("frame"),
    }


# ------------------------------------------------------------------
# Enhanced inference function
# ------------------------------------------------------------------

def infer_enhanced(
    ref_audio: str,
    ref_text: str,
    gen_text: str,
    model: EnhancedCFM,
    vocoder,
    speaker_encoder: SpeakerEncoder,
    emotion_encoder: EmotionEncoder,
    # Generation params
    nfe_step: int = 32,
    cfg_strength: float = 2.0,
    emotion_cfg_strength: float = 1.0,
    sway_sampling_coef: float = -1.0,
    speed: float = 1.0,
    fix_duration: float | None = None,
    seed: int | None = None,
    # Audio params
    target_rms: float = 0.1,
    show_info=print,
    device_str: str = device,
) -> tuple[np.ndarray, int]:
    """
    Full enhanced inference pipeline.
    
    Args:
        ref_audio:            Path to reference audio (any language)
        ref_text:             Reference text (or empty for ASR transcription)
        gen_text:             Text to generate (Russian or any language)
        model:                Enhanced CFM model
        vocoder:              Mel-to-waveform vocoder
        speaker_encoder:      Pretrained speaker encoder
        emotion_encoder:      Pretrained emotion encoder
        emotion_cfg_strength: How strongly to apply emotion (0 = neutral, 2+ = strong)
        
    Returns:
        (waveform_numpy, sample_rate)
    """
    # Preprocess reference audio and text
    ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(
        ref_audio, ref_text, show_info=show_info
    )

    # Extract speaker and emotion embeddings from reference
    show_info("Extracting speaker and emotion embeddings from reference...")
    embeddings = extract_reference_embeddings(
        ref_audio_processed, speaker_encoder, emotion_encoder, device_str
    )
    speaker_emb = embeddings["speaker_emb"]
    emotion_global = embeddings["emotion_global"]
    emotion_frame = embeddings.get("emotion_frame")

    # Load and preprocess audio
    audio, sr = torchaudio.load(ref_audio_processed)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device_str)

    # Chunk text
    max_chars = int(
        len(ref_text_processed.encode("utf-8"))
        / (audio.shape[-1] / target_sample_rate)
        * (22 - audio.shape[-1] / target_sample_rate)
        * speed
    )
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)

    show_info(f"Generating audio in {len(gen_text_batches)} batches with emotion guidance={emotion_cfg_strength}...")

    generated_waves = []

    for i, batch_text in enumerate(gen_text_batches):
        local_speed = speed
        if len(batch_text.encode("utf-8")) < 10:
            local_speed = 0.3

        # Prepare text
        text_list = [ref_text_processed + batch_text]
        final_text_list = convert_char_to_pinyin(text_list)

        ref_audio_len = audio.shape[-1] // hop_length
        if fix_duration is not None:
            duration = int(fix_duration * target_sample_rate / hop_length)
        else:
            ref_text_len = len(ref_text_processed.encode("utf-8"))
            gen_text_len = len(batch_text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)

        # Generate with enhanced model
        with torch.inference_mode():
            generated, _ = model.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                seed=seed,
                # Enhanced conditioning
                speaker_emb=speaker_emb,
                emotion_global=emotion_global,
                emotion_frame=emotion_frame,
                emotion_cfg_strength=emotion_cfg_strength,
            )

            generated = generated.to(torch.float32)
            generated = generated[:, ref_audio_len:, :]
            generated = generated.permute(0, 2, 1)

            if mel_spec_type == "vocos":
                generated_wave = vocoder.decode(generated)
            elif mel_spec_type == "bigvgan":
                generated_wave = vocoder(generated)

            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms

            generated_wave = generated_wave.squeeze().cpu().numpy()
            generated_waves.append(generated_wave)

    # Concatenate with cross-fade
    if len(generated_waves) == 1:
        final_wave = generated_waves[0]
    else:
        final_wave = generated_waves[0]
        cross_fade_samples = int(cross_fade_duration * target_sample_rate)
        for i in range(1, len(generated_waves)):
            prev_wave = final_wave
            next_wave = generated_waves[i]
            cf = min(cross_fade_samples, len(prev_wave), len(next_wave))
            if cf <= 0:
                final_wave = np.concatenate([prev_wave, next_wave])
            else:
                fade_out = np.linspace(1, 0, cf)
                fade_in = np.linspace(0, 1, cf)
                cross_faded = prev_wave[-cf:] * fade_out + next_wave[:cf] * fade_in
                final_wave = np.concatenate([prev_wave[:-cf], cross_faded, next_wave[cf:]])

    return final_wave, target_sample_rate


# ------------------------------------------------------------------
# Quick inference helper (all-in-one)
# ------------------------------------------------------------------

def quick_inference(
    ref_audio_path: str,
    gen_text: str,
    ckpt_path: str,
    ref_text: str = "",
    emotion_cfg_strength: float = 1.0,
    output_path: str = "output.wav",
    device_str: str = device,
):
    """
    All-in-one inference: loads models, extracts embeddings, generates audio.
    Good for scripts and demos, not for batch processing (reloads models each time).
    """
    import soundfile as sf

    # Load models
    vocoder = load_vocoder(device=device_str)
    model = load_enhanced_model(ckpt_path, device_str=device_str)
    speaker_enc, emotion_enc = load_embedding_extractors(device_str=device_str)

    # Generate
    wave, sr = infer_enhanced(
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        gen_text=gen_text,
        model=model,
        vocoder=vocoder,
        speaker_encoder=speaker_enc,
        emotion_encoder=emotion_enc,
        emotion_cfg_strength=emotion_cfg_strength,
        device_str=device_str,
    )

    sf.write(output_path, wave, sr)
    print(f"Saved to {output_path}")
    return wave, sr
