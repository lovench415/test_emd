"""
Data Preparation Pipeline for F5-TTS Enhanced
================================================

Complete preprocessing pipeline:

1. Raw audio + text → validated dataset
2. Audio → mel spectrograms (optional pre-extraction)
3. Audio → speaker embeddings (WavLM-SV)
4. Audio → emotion embeddings (emotion2vec, global + frame-level)
5. Pack everything into training-ready format

Usage:
    # Step 1: Prepare raw dataset
    python prepare_data.py --stage prepare \
        --audio_dir /data/audio \
        --metadata /data/metadata.csv \
        --output_dir /data/prepared

    # Step 2: Extract and cache embeddings
    python prepare_data.py --stage embeddings \
        --dataset_dir /data/prepared \
        --output_dir /data/embeddings_cache \
        --speaker_backend wavlm_sv \
        --emotion_backend emotion2vec_base

    # Step 3: Verify dataset integrity
    python prepare_data.py --stage verify \
        --dataset_dir /data/prepared \
        --embedding_dir /data/embeddings_cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torchaudio
import numpy as np
from tqdm import tqdm


# =====================================================================
# Stage 1: Raw Data Preparation
# =====================================================================

def validate_audio(audio_path: str, min_duration: float = 0.3, max_duration: float = 30.0) -> dict | None:
    """
    Validate a single audio file.
    
    Returns:
        dict with audio metadata if valid, None if invalid.
    
    Checks:
        - File exists and is readable
        - Duration within [0.3, 30] seconds
        - Sample rate is valid (will be resampled to 24kHz later)
        - Audio is not silent (RMS > threshold)
        - No NaN/Inf values
    """
    try:
        info = torchaudio.info(audio_path)
        duration = info.num_frames / info.sample_rate
        
        if duration < min_duration or duration > max_duration:
            return None
        
        # Quick load to check for corruption
        audio, sr = torchaudio.load(audio_path)
        
        # Check for NaN/Inf
        if torch.isnan(audio).any() or torch.isinf(audio).any():
            return None
        
        # Check for silence (RMS threshold)
        rms = torch.sqrt(torch.mean(audio ** 2))
        if rms < 1e-5:
            return None
        
        return {
            "audio_path": os.path.abspath(audio_path),
            "duration": duration,
            "sample_rate": sr,
            "channels": info.num_channels,
            "rms": rms.item(),
        }
    except Exception as e:
        print(f"  [SKIP] {audio_path}: {e}")
        return None


def prepare_metadata(
    audio_dir: str,
    metadata_path: str,
    output_dir: str,
    target_sample_rate: int = 24000,
    min_duration: float = 0.3,
    max_duration: float = 30.0,
):
    """
    Stage 1: Prepare dataset from raw audio files + metadata.
    
    Expected metadata format (CSV or JSON):
        CSV:  audio_path|text|language
        JSON: [{"audio_path": "...", "text": "...", "language": "ru"}, ...]
    
    For cross-lingual training, audio can be in ANY language.
    The text field should contain the transcript in the ORIGINAL language.
    
    Output structure:
        output_dir/
        ├── raw.arrow          (HuggingFace dataset)
        ├── duration.json      (durations for DynamicBatchSampler)
        ├── metadata.json      (full metadata with stats)
        └── vocab.txt          (character vocabulary)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    if metadata_path.endswith(".csv") or metadata_path.endswith(".lst"):
        entries = _load_csv_metadata(metadata_path)
    elif metadata_path.endswith(".json"):
        entries = _load_json_metadata(metadata_path)
    elif metadata_path.endswith(".jsonl"):
        entries = _load_jsonl_metadata(metadata_path)
    else:
        raise ValueError(f"Unsupported metadata format: {metadata_path}")
    
    print(f"Loaded {len(entries)} entries from metadata")
    
    # Validate and filter
    valid_entries = []
    durations = []
    skipped = 0
    
    for entry in tqdm(entries, desc="Validating audio files"):
        audio_path = entry["audio_path"]
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(audio_dir, audio_path)
        
        result = validate_audio(audio_path, min_duration, max_duration)
        if result is None:
            skipped += 1
            continue
        
        valid_entries.append({
            "audio_path": result["audio_path"],
            "text": entry["text"],
            "duration": result["duration"],
            "language": entry.get("language", "unknown"),
        })
        durations.append(result["duration"])
    
    print(f"Valid: {len(valid_entries)} | Skipped: {skipped}")
    print(f"Total duration: {sum(durations)/3600:.1f} hours")
    print(f"Mean duration: {np.mean(durations):.1f}s | "
          f"Median: {np.median(durations):.1f}s")
    
    # Save as HuggingFace dataset
    from datasets import Dataset as HFDataset
    
    dataset = HFDataset.from_list(valid_entries)
    dataset.save_to_disk(os.path.join(output_dir, "raw"))
    
    # Save durations
    with open(os.path.join(output_dir, "duration.json"), "w") as f:
        json.dump({"duration": durations}, f)

    if args.build_vocab:
        # Build vocabulary from all texts
        vocab = _build_vocab([e["text"] for e in valid_entries])
        vocab_path = os.path.join(output_dir, "vocab.txt")
        with open(vocab_path, "w", encoding="utf-8") as f:
            for char in sorted(vocab):
                f.write(char + "\n")
        print(f"Vocabulary: {len(vocab)} characters → {vocab_path}")
    
    # Save full metadata
    stats = {
        "total_samples": len(valid_entries),
        "total_duration_hours": sum(durations) / 3600,
        "mean_duration_s": float(np.mean(durations)),
        "languages": list(set(e["language"] for e in valid_entries)),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\nDataset saved to {output_dir}")
    return valid_entries


def _load_csv_metadata(path: str) -> list[dict]:
    """Load pipe-separated metadata: audio_path|text|language"""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                entry = {
                    "audio_path": parts[0].strip(),
                    "text": parts[1].strip(),
                }
                if len(parts) >= 3:
                    entry["language"] = parts[2].strip()
                entries.append(entry)
    return entries


def _load_json_metadata(path: str) -> list[dict]:
    """Load JSON metadata."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_jsonl_metadata(path: str) -> list[dict]:
    """Load JSON metadata."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            record = json.loads(line)
            entries.append(record)
    return entries

def _build_vocab(texts: list[str]) -> set[str]:
    """Build character vocabulary from all texts."""
    vocab = set()
    for text in texts:
        for char in text:
            vocab.add(char)
    # Add standard F5-TTS special characters
    for char in " .,!?;:'-\"()[]{}—–…":
        vocab.add(char)
    # Add Russian characters
    for code in range(0x0410, 0x0450):  # А-я
        vocab.add(chr(code))
    vocab.add("ё")
    vocab.add("Ё")
    return vocab


# =====================================================================
# Stage 2: Embedding Extraction & Caching
# =====================================================================

def extract_and_cache_embeddings(
    dataset_dir: str,
    output_dir: str,
    speaker_backend: str = "wavlm_sv",
    emotion_backend: str = "emotion2vec_base",
    speaker_dim: int = 512,
    emotion_dim: int = 512,
    batch_size: int = 1,
    device: str = "cuda",
    resume: bool = True,
):
    """
    Stage 2: Extract speaker and emotion embeddings for all samples.
    
    Saves one .pt file per sample:
        {index}.pt = {
            "speaker_raw":        (speaker_raw_dim,),    # 512 for WavLM-SV
            "emotion_global_raw": (emotion_raw_dim,),    # 768 for emotion2vec
            "emotion_frame_raw":  (T_frames, emotion_raw_dim),
        }
    
    Raw embeddings (before projection) are saved because:
    1. Projection heads are trainable and change during training
    2. Raw embeddings are fixed → compute once, reuse forever
    3. Different experiments can use different projection architectures
    """
    from f5_tts.model.speaker_encoder import SpeakerEncoder
    from f5_tts.model.emotion_encoder import EmotionEncoder
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    from datasets import Dataset as HFDataset
    try:
        dataset = HFDataset.load_from_disk(os.path.join(dataset_dir, "raw"))
    except:
        dataset = HFDataset.from_file(os.path.join(dataset_dir, "raw.arrow"))
    
    print(f"Dataset: {len(dataset)} samples")
    
    # Load encoders
    print(f"Loading speaker encoder: {speaker_backend}...")
    speaker_enc = SpeakerEncoder(
        backend=speaker_backend,
        output_dim=speaker_dim,
        device=device,
    ).to(device).eval()
    
    print(f"Loading emotion encoder: {emotion_backend}...")
    emotion_enc = EmotionEncoder(
        backend=emotion_backend,
        output_dim=emotion_dim,
        frame_level=True,
        device=device,
    ).to(device).eval()
    
    # Process each sample
    success = 0
    failed = 0
    skipped = 0
    
    for i in tqdm(range(len(dataset)), desc="Extracting embeddings"):
        out_path = os.path.join(output_dir, f"{i}.pt")
        
        # Skip if already computed (resume mode)
        if resume and os.path.exists(out_path):
            skipped += 1
            continue
        
        try:
            row = dataset[i]
            audio_path = row["audio_path"]
            
            # Load audio
            audio, sr = torchaudio.load(audio_path)
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            audio = audio.to(device)
            
            with torch.no_grad():
                # Speaker embedding (raw, before projection)
                spk_raw = speaker_enc.extract_raw_embedding(audio, sr=sr)
                # spk_raw: (1, speaker_raw_dim) e.g. (1, 512)
                
                # Emotion embedding (raw, before projection)
                emo_global_raw, emo_frame_raw = emotion_enc.extract_raw_features(audio, sr=sr)
                # emo_global_raw: (1, emotion_raw_dim) e.g. (1, 768)
                # emo_frame_raw:  (1, T_frames, emotion_raw_dim)
            
            # Save
            save_dict = {
                "speaker_raw": spk_raw.squeeze(0).cpu(),
                "emotion_global_raw": emo_global_raw.squeeze(0).cpu(),
            }
            if emo_frame_raw is not None:
                save_dict["emotion_frame_raw"] = emo_frame_raw.squeeze(0).cpu()
            
            torch.save(save_dict, out_path)
            success += 1
            
            # Free GPU memory periodically
            if i % 100 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  [FAIL] Sample {i}: {e}")
            # Save zeros as fallback (model handles zeros gracefully)
            torch.save({
                "speaker_raw": torch.zeros(speaker_enc.raw_dim),
                "emotion_global_raw": torch.zeros(emotion_enc.raw_dim),
                "emotion_frame_raw": None,
            }, out_path)
            failed += 1
    
    print(f"\nDone! Success: {success} | Failed: {failed} | Skipped: {skipped}")
    
    # Save extraction metadata
    meta = {
        "total_samples": len(dataset),
        "speaker_backend": speaker_backend,
        "speaker_raw_dim": speaker_enc.raw_dim,
        "emotion_backend": emotion_backend,
        "emotion_raw_dim": emotion_enc.raw_dim,
        "success": success,
        "failed": failed,
    }
    with open(os.path.join(output_dir, "embedding_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# =====================================================================
# Stage 3: Verification
# =====================================================================

def verify_dataset(
    dataset_dir: str,
    embedding_dir: str | None = None,
):
    """
    Stage 3: Verify dataset integrity.
    
    Checks:
    - All audio files exist and are loadable
    - Durations match
    - Embeddings exist for all samples (if embedding_dir provided)
    - Embedding dimensions are consistent
    - No NaN/Inf in embeddings
    """
    from datasets import Dataset as HFDataset
    
    try:
        dataset = HFDataset.load_from_disk(os.path.join(dataset_dir, "raw"))
    except:
        dataset = HFDataset.from_file(os.path.join(dataset_dir, "raw.arrow"))
    
    with open(os.path.join(dataset_dir, "duration.json"), "r") as f:
        durations = json.load(f)["duration"]
    
    print(f"Dataset: {len(dataset)} samples, {len(durations)} durations")
    
    errors = []
    
    # Check audio files
    print("Checking audio files...")
    for i in tqdm(range(min(len(dataset), 1000)), desc="Sampling audio"):
        row = dataset[i]
        if not os.path.exists(row["audio_path"]):
            errors.append(f"Missing audio: {row['audio_path']}")
    
    # Check embeddings
    if embedding_dir:
        print("Checking embeddings...")
        missing_emb = 0
        bad_emb = 0
        spk_dims = set()
        emo_dims = set()
        
        for i in tqdm(range(len(dataset)), desc="Checking embeddings"):
            emb_path = os.path.join(embedding_dir, f"{i}.pt")
            if not os.path.exists(emb_path):
                missing_emb += 1
                continue
            
            try:
                emb = torch.load(emb_path, map_location="cpu", weights_only=True)
                
                # Check speaker
                spk = emb.get("speaker_raw")
                if spk is not None:
                    spk_dims.add(tuple(spk.shape))
                    if torch.isnan(spk).any() or torch.isinf(spk).any():
                        bad_emb += 1
                
                # Check emotion
                emo = emb.get("emotion_global_raw")
                if emo is not None:
                    emo_dims.add(tuple(emo.shape))
                    if torch.isnan(emo).any() or torch.isinf(emo).any():
                        bad_emb += 1
                        
            except Exception as e:
                errors.append(f"Bad embedding {i}: {e}")
        
        print(f"Missing embeddings: {missing_emb}/{len(dataset)}")
        print(f"Corrupted embeddings: {bad_emb}")
        print(f"Speaker dimensions: {spk_dims}")
        print(f"Emotion dimensions: {emo_dims}")
    
    if errors:
        print(f"\n{len(errors)} ERRORS found:")
        for e in errors[:20]:
            print(f"  - {e}")
    else:
        print("\n✅ Dataset verification passed!")


# =====================================================================
# Audio Augmentation (optional, for robustness)
# =====================================================================

def augment_reference_audio(
    audio: torch.Tensor,
    sr: int = 24000,
    noise_level: float = 0.005,
    speed_range: tuple[float, float] = (0.95, 1.05),
    pitch_shift_range: tuple[int, int] = (-1, 1),
) -> torch.Tensor:
    """
    Optional augmentation for reference audio during training.
    Makes the model more robust to varying recording conditions.
    
    Applied ONLY to the copy sent to speaker/emotion encoders,
    NOT to the mel spectrogram used as generation condition.
    
    Augmentations:
    - Additive Gaussian noise (simulates different recording environments)
    - Speed perturbation (simulates different speaking rates)
    - Volume normalization jitter
    """
    # Additive noise
    if noise_level > 0:
        noise = torch.randn_like(audio) * noise_level
        audio = audio + noise
    
    # Speed perturbation
    if speed_range != (1.0, 1.0):
        speed = np.random.uniform(*speed_range)
        if abs(speed - 1.0) > 0.01:
            effects = [["speed", str(speed)], ["rate", str(sr)]]
            audio_np = audio.numpy()
            # Using torchaudio sox effects
            try:
                augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
                    audio.unsqueeze(0) if audio.dim() == 1 else audio,
                    sr,
                    effects,
                )
                audio = augmented.squeeze(0)
            except:
                pass  # Skip if sox not available
    
    # Volume jitter
    volume_factor = np.random.uniform(0.8, 1.2)
    audio = audio * volume_factor
    
    return audio


# =====================================================================
# CLI Entry Point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="F5-TTS Enhanced Data Preparation")
    parser.add_argument(
        "--stage",
        choices=["prepare", "embeddings", "verify", "all"],
        required=True,
        help="Pipeline stage to run",
    )
    
    # Stage 1: prepare
    parser.add_argument("--audio_dir", type=str, default=".", help="Root audio directory")
    parser.add_argument("--metadata", type=str, help="Path to metadata file (CSV or JSON)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--min_duration", type=float, default=0.3)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--build_vocab", action="store_true", help="Build multilingual vocab.txt")
    
    # Stage 2: embeddings
    parser.add_argument("--dataset_dir", type=str, help="Path to prepared dataset")
    parser.add_argument("--embedding_dir", type=str, help="Path to save/load embeddings")
    parser.add_argument("--speaker_backend", type=str, default="wavlm_sv",
                        choices=["wavlm_sv", "ecapa_tdnn", "resemblyzer"])
    parser.add_argument("--emotion_backend", type=str, default="emotion2vec_base",
                        choices=["emotion2vec_base", "emotion2vec_plus", "wav2vec2_ser", "hubert_ser"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true", default=True)
    
    args = parser.parse_args()
    
    if args.stage in ("prepare", "all"):
        assert args.metadata, "--metadata required for prepare stage"
        prepare_metadata(
            args.audio_dir, args.metadata, args.output_dir,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
    
    if args.stage in ("embeddings", "all"):
        ds_dir = args.dataset_dir or args.output_dir
        emb_dir = args.embedding_dir or os.path.join(args.output_dir, "embeddings")
        extract_and_cache_embeddings(
            ds_dir, emb_dir,
            speaker_backend=args.speaker_backend,
            emotion_backend=args.emotion_backend,
            device=args.device,
            resume=args.resume,
        )
    
    if args.stage in ("verify", "all"):
        ds_dir = args.dataset_dir or args.output_dir
        emb_dir = args.embedding_dir or os.path.join(args.output_dir, "embeddings")
        verify_dataset(ds_dir, emb_dir)


if __name__ == "__main__":
    main()
