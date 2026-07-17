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
) -> ProsodyEncoder:
    """Load prosody encoder for F0/energy extraction."""
    print(f"Loading prosody encoder ({backend})...")
    import os as _os
    # MUST match the mode used to build the training cache (F0_NORM_MODE), or the
    # prosody features at inference differ from training → degraded prosody.
    _f0_mode = _os.environ.get("F0_NORM_MODE", "legacy")
    return ProsodyEncoder(
        backend=backend, output_dim=output_dim, device=device_str,
        rmvpe_model_path=rmvpe_model_path, f0_norm_mode=_f0_mode,
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
        # weights_only=False explicitly: a training checkpoint may carry non-tensor
        # payload (arch_flags dict, optimizer/scheduler state). The torch default for
        # weights_only changed across PyTorch versions (True on 2.6+), which would
        # reject such checkpoints version-dependently. Be explicit so loading behaves
        # the same everywhere and stays consistent with the resume path.
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

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
    # Detect emotion bottleneck from weight shapes. Legacy: emotion_raw_proj is a
    # single Linear → key "...emotion_raw_proj.weight". Bottleneck: it's a Sequential
    # (Linear(768,B), SiLU, Linear(B,dim)) → keys "...emotion_raw_proj.0.weight" etc.
    # The bottleneck width B = first Linear's out_features. This is a weight-shape
    # change, so it MUST be reconstructed or the state dict won't load.
    # Detect fusion_mode from weights: residual builds fusion_speaker.* (+ fusion_emotion
    # + gate); concat builds a single fusion.*. Weight-based detection is more robust
    # than arch_flags (works for checkpoints saved before arch_flags existed).
    fusion_mode_detected = None
    if any(".fusion_speaker." in k for k in state_for_shapes):
        fusion_mode_detected = "residual"
    elif any(k.rstrip("0123456789.").endswith("cond_aggregator.fusion") for k in state_for_shapes) \
            or any(".fusion.0.weight" in k for k in state_for_shapes):
        fusion_mode_detected = "concat"

    emotion_bottleneck_dim = 0
    for k, v in state_for_shapes.items():
        if k.endswith("emotion_raw_proj.0.weight"):
            emotion_bottleneck_dim = v.shape[0]  # out_features of the narrow layer
            break
    # Detect AdaLN bottleneck width from block_projs.0.0.weight = [bottleneck, dim].
    # Trained with --adaln_bottleneck_dim N → first projection's out_features = N.
    # Default DiT is 256; if the checkpoint used 512 it MUST be reconstructed or the
    # state dict won't load (this is the most common inference shape mismatch).
    adaln_bottleneck_dim = 256
    for k, v in state_for_shapes.items():
        if k.endswith("adaln_cond.block_projs.0.0.weight"):
            adaln_bottleneck_dim = v.shape[0]
            break
    # Detect whether the trained duration predictor used speaker input, from the
    # first Linear's in_features. prosody_global(7)+log_text_len(1) = 8 without
    # speaker; +speaker_dim with it. Reconstruct the matching architecture so the
    # weights load (otherwise shape mismatch).
    dur_use_speaker = True
    for k, v in state_for_shapes.items():
        if k.endswith("duration_predictor.net.0.weight"):
            in_features = v.shape[1]
            dur_use_speaker = in_features > 8 + 1  # >9 → speaker dim included
            break

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
        prosody_dim=prosody_dim, prosody_raw_dim=prosody_raw_dim,
        # Variant A: prosody cross-attn presence is detected from checkpoint keys
        # (prosody_cross_attns.* absent → trained without it → keep off here).
        use_prosody_cross_attn=has_prosody and any(
            "prosody_cross_attns" in k for k in state_for_shapes),
        # Emotion cross-attn presence detected from cond_aggregator.cross_attns.* keys.
        # Use a prefix that does NOT also match "prosody_cross_attns" (which ends in
        # the same substring). Absent → lightweight model without emotion cross-attn.
        use_cross_attn_cond=any(
            k.split("cond_aggregator.")[-1].startswith("cross_attns.")
            for k in state_for_shapes if "cond_aggregator." in k),
        # input_add presence detected from input_add.proj.* keys. Absent → trained
        # with --no_input_add → don't build it (otherwise 4 missing params). Note:
        # input_add is normally CRITICAL for cloning; if it's missing the model was
        # trained without it, and inference must match that to load.
        use_input_add_cond=any(
            ".input_add." in k for k in state_for_shapes),
        emotion_bottleneck_dim=emotion_bottleneck_dim,
        adaln_bottleneck_dim=adaln_bottleneck_dim,
        # emotion_direct presence is detectable from its projection weights
        # (emotion_direct_proj.*). Reconstruct so the weights load.
        use_emotion_direct=any("emotion_direct_proj" in k for k in state_for_shapes),
        # Variant C: AdaLN speaker-only is detected from absence of prosody_global
        # enrichment under a speaker-only trained checkpoint. The flag is structural
        # (no extra weights), so it must be set to match training; detect via the
        # presence of prosody_global_proj being unused is unreliable, so honor the
        # explicit model_cfg override if the caller passed one.
        # adaln_speaker_only is a STRUCTURAL flag (no weight-shape change), so it
        # can't be detected from the state dict. Read it from arch_flags saved at
        # training time; fall back to explicit model_cfg override, then False.
        adaln_speaker_only=(
            checkpoint.get("arch_flags", {}).get("adaln_speaker_only")
            if isinstance(checkpoint, dict) and checkpoint.get("arch_flags")
            else model_cfg.get("adaln_speaker_only", False)
        ),
        # prosody_in_adaln is structural too (decides whether prosody_global enriches
        # AdaLN under speaker_only). Changes which params train, not weight SHAPES, so
        # it must come from arch_flags to match training at inference.
        prosody_in_adaln=(
            checkpoint.get("arch_flags", {}).get("prosody_in_adaln", False)
            if isinstance(checkpoint, dict) and checkpoint.get("arch_flags")
            else model_cfg.get("prosody_in_adaln", False)
        ),
        # use_timbre_encoder ADDS weights (the TimbreEncoder + timbre_proj/gate), so it
        # must match training or state-dict loading fails. Read from arch_flags; the
        # presence of timbre weights in the checkpoint is the ground truth, but
        # arch_flags is the explicit record saved at train time.
        use_timbre_encoder=(
            checkpoint.get("arch_flags", {}).get("use_timbre_encoder", False)
            if isinstance(checkpoint, dict) and checkpoint.get("arch_flags")
            else model_cfg.get("use_timbre_encoder", False)
        ),
        # normalize_speaker is also structural (just an F.normalize, no weight change)
        # → must come from arch_flags, same as adaln_speaker_only.
        normalize_speaker=(
            checkpoint.get("arch_flags", {}).get("normalize_speaker", False)
            if isinstance(checkpoint, dict) and checkpoint.get("arch_flags")
            else model_cfg.get("normalize_speaker", False)
        ),
        # fusion_mode changes weight structure (residual builds fusion_speaker +
        # fusion_emotion + gate; concat builds a single fusion). Prefer weight-based
        # detection; fall back to arch_flags, then model_cfg, then concat.
        fusion_mode=(
            fusion_mode_detected
            or (checkpoint.get("arch_flags", {}).get("fusion_mode")
                if isinstance(checkpoint, dict) and checkpoint.get("arch_flags") else None)
            or model_cfg.get("fusion_mode", "concat")
        ),
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
        use_duration_predictor=has_dur_pred,
        duration_use_speaker=dur_use_speaker,
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
    # Print the actual names so architecture mismatches are diagnosable. Missing =
    # in the model but not the checkpoint (model built a module the ckpt lacks).
    # Unexpected = in the checkpoint but not the model (ckpt has a module the
    # inference model didn't build — usually a flag not auto-detected/reconstructed).
    if missing:
        print("  ── Missing (model has, checkpoint lacks) ──")
        for k in missing:
            print(f"     {k}")
    if unexpected:
        print("  ── Unexpected (checkpoint has, model lacks) ──")
        for k in unexpected:
            print(f"     {k}")

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
    ablate_speaker: bool = False,
) -> RawConditionBatch:
    """Extract raw speaker + emotion + prosody embeddings from any-language reference.

    Returns RAW (un-projected) embeddings — the ConditioningAggregator inside
    the model handles projection from raw→target dim for consistency with
    the training pipeline.
    """
    audio, sr = torchaudio.load(ref_audio_path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.to(device_str)

    # Pass explicit true length so the speaker (and emotion) encoders take the
    # SAME masked-pooling path as prepare_data caching. For a single unpadded file
    # this equals the full length, but passing it makes the train/inference path
    # identical instead of relying on "single file ⇒ no padding ⇒ probably same",
    # which would break the moment ref audio is ever batched.
    spk_lengths = torch.tensor([audio.shape[-1]], device=device_str, dtype=torch.long)
    spk_raw = speaker_encoder.extract_raw(audio, sr=sr, lengths=spk_lengths)
    if hasattr(emotion_encoder, "extract_raw_with_mask"):
        emo_global_raw, emo_frame_raw, emo_frame_mask = emotion_encoder.extract_raw_with_mask(audio, sr=sr)
    else:
        emo_global_raw, emo_frame_raw = emotion_encoder.extract_raw(audio, sr=sr)
        emo_frame_mask = (
            torch.ones(emo_frame_raw.shape[:2], dtype=torch.bool, device=emo_frame_raw.device)
            if emo_frame_raw is not None else None
        )

    # Prosody extraction (F0 + energy + voicing)
    prosody_raw = None
    prosody_mask = None
    if prosody_encoder is not None:
        prosody_raw, prosody_mask = prosody_encoder.extract_raw(audio, sr=sr)

    # Match the training cache's fp16 quantization. prepare_data's save_worker stores
    # speaker/emotion/prosody raw embeddings as float16 to save disk, so during
    # training the projections see fp16-rounded inputs. At inference these come fresh
    # in float32, a slightly different (finer) input distribution than the projection
    # heads were trained on. Round-tripping through fp16 here makes the inference
    # input distribution identical to training. The effect is small, but it removes
    # one more train/inference mismatch.
    def _match_cache_fp16(t):
        return t.half().float() if isinstance(t, torch.Tensor) and t.is_floating_point() else t
    # Only emotion_* are fp16 in the cache now, so only they are fp16-rounded here to
    # match training. speaker_raw and prosody_raw are cached in fp32 (see prepare_data),
    # so they must stay full fp32 at inference — rounding them would re-introduce the
    # train/inference mismatch this matching is meant to remove.
    emo_global_raw = _match_cache_fp16(emo_global_raw)
    emo_frame_raw = _match_cache_fp16(emo_frame_raw)
    # spk_raw: NOT fp16-rounded — cached fp32 (carries the pitch-level signal).
    # prosody_raw: NOT fp16-rounded — cached fp32 (channel 5 = log_f0_absolute).

    return RawConditionBatch(
        speaker_raw=spk_raw,
        emotion_global_raw=emo_global_raw,
        emotion_frame_raw=emo_frame_raw,
        prosody_raw=prosody_raw,
        speaker_present=(
            # ablate_speaker=True marks the speaker as ABSENT (present=False), which
            # routes through the same multiply-mask the model saw for voice-dropped
            # samples in training → a clean "no speaker" generation for the ablation
            # test (does the speaker path actually affect the output?). Normal path
            # uses ones (speaker present).
            torch.zeros(spk_raw.shape[0], dtype=torch.bool, device=spk_raw.device)
            if (spk_raw is not None and ablate_speaker)
            else torch.ones(spk_raw.shape[0], dtype=torch.bool, device=spk_raw.device)
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
    duration_mode: str = "auto",  # "auto" (predictor if present) | "heuristic" (ref-rate scaling)
    seed: int | None = None,
    target_rms: float = 0.1,
    show_info=print,
    device_str: str = device,
    ablate_speaker: bool = False,
) -> tuple[np.ndarray, int]:
    """Full enhanced inference: reference audio → generated speech.

    ablate_speaker=True marks the speaker as absent (speaker_present=False) so the
    speaker path contributes nothing — used to test whether speaker conditioning is
    trained: compare output with vs without it. If the two sound the same, the
    speaker path is being ignored (under-trained).
    """
    ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(
        ref_audio, ref_text, show_info=show_info,
    )

    show_info("Extracting embeddings from reference...")
    embs = extract_reference_embeddings(
        ref_audio_processed, speaker_encoder, emotion_encoder, device_str,
        prosody_encoder=prosody_encoder,
        ablate_speaker=ablate_speaker,
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

    show_info(f"Generating {len(gen_text_batches)} chunk(s), emo_cfg={emotion_cfg_strength}, pros_cfg={prosody_cfg_strength}")

    waves = []
    for batch_text in gen_text_batches:
        local_speed = 0.3 if len(batch_text.encode("utf-8")) < 10 else speed
        text_list = [ref_text_processed + batch_text]
        final_text = convert_char_to_pinyin(text_list)

        ref_len = audio.shape[-1] // hop_length
        _heuristic = (duration_mode == "heuristic")
        if fix_duration is not None:
            # fix_duration is the desired GENERATED speech length (seconds).
            # The reference prefix occupies ref_len frames and is trimmed off the
            # output afterwards, so it must be ADDED on top — otherwise the trim
            # eats most of fix_duration (e.g. fix_duration=4s with a 3s reference
            # would leave only ~1s of actual generated speech). This matches the
            # duration_predictor branch below (ref_len + gen frames).
            gen_frames = int(fix_duration * target_sample_rate / hop_length)
            duration = ref_len + gen_frames
        elif (not _heuristic
              and getattr(model, "duration_predictor", None) is not None
              and embs.prosody_raw is not None):
            # Use trained duration predictor conditioned on reference prosody
            gen_text_bytes = len(batch_text.encode("utf-8"))
            spk_emb = embs.speaker_raw  # (1, speaker_dim) raw speaker embedding
            duration = model.duration_predictor.predict_duration(
                prosody_raw=embs.prosody_raw,
                prosody_mask=embs.prosody_mask,
                text_byte_len=gen_text_bytes,
                speaker_emb=spk_emb,
            )
            duration = ref_len + duration  # predictor returns gen frames, add ref prefix
        else:
            # Heuristic (level 0a): scale the reference's OWN measured rate
            # (ref_len / ref_text_len) by the generated text length. Uses the real
            # tempo of THIS reference → no learned weights, no per-speaker
            # memorisation, no over-prediction. This is also the default fallback
            # when no predictor is trained. local_speed > 1 speeds up, < 1 slows.
            ref_text_len = max(len(ref_text_processed.encode("utf-8")), 1)
            gen_text_len = len(batch_text.encode("utf-8"))
            duration = ref_len + int(ref_len / ref_text_len * gen_text_len / local_speed)

        with torch.inference_mode():
            model_conditions = None
            if getattr(model, "condition_lifecycle", None) is not None:
                lens = torch.tensor([ref_len], device=device_str, dtype=torch.long)
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
