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
from tinytag import TinyTag

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
        info = TinyTag.get(audio_path)
        duration = info.duration
        
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
            "channels": info.channels,
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
            "speaker": entry.get("speaker", "default"),
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
        "vocab_size": len(vocab),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\nDataset saved to {output_dir}")
    return valid_entries


def _load_csv_metadata(path: str) -> list[dict]:
    """Load pipe-separated metadata: audio_path|text|language[|speaker]"""
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
                entry["speaker"] = parts[2].strip()
            entry["language"] = 'ru'

            entries.append(entry)
    return entries


def _load_json_metadata(path: str) -> list[dict]:
    """Load JSON metadata."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    prosody_backend: str = "dio",
    batch_size: int = 1,
    device: str = "cuda",
    resume: bool = True,
):
    """
    Stage 2: Extract speaker, emotion, and prosody embeddings for all samples.
    
    Saves one .pt file per sample:
        {index}.pt = {
            "speaker_raw":        (speaker_raw_dim,),    # 512 for WavLM-SV
            "emotion_global_raw": (emotion_raw_dim,),    # 768 for emotion2vec
            "emotion_frame_raw":  (T_frames, emotion_raw_dim),
            "prosody_raw":        (T_frames, 5),          # log_f0, voicing, log_energy, Δf0, Δenergy
            "prosody_mask":       (T_frames,),             # bool
        }
    
    Raw embeddings (before projection) are saved because:
    1. Projection heads are trainable and change during training
    2. Raw embeddings are fixed → compute once, reuse forever
    3. Different experiments can use different projection architectures
    """
    from f5_tts.model.speaker_encoder import SpeakerEncoder
    from f5_tts.model.emotion_encoder import EmotionEncoder
    from f5_tts.model.prosody_encoder import ProsodyEncoder
    
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

    print(f"Loading prosody encoder: {prosody_backend}...")
    import os as _os
    _f0_mode = _os.environ.get("F0_NORM_MODE", "legacy")
    prosody_enc = ProsodyEncoder(
        backend=prosody_backend,
        f0_norm_mode=_f0_mode,
        device=device,
    )
    
    # ── Determine which samples to process ──
    indices = []
    skipped = 0
    for i in range(len(dataset)):
        out_path = os.path.join(output_dir, f"{i}.pt")
        if resume and os.path.exists(out_path):
            skipped += 1
        else:
            indices.append(i)
    
    if skipped:
        print(f"Skipping {skipped} already-computed embeddings")
    print(f"Processing {len(indices)} samples...")

    # ── Batched extraction with I/O pipeline ──
    # GPU encoders (speaker, emotion) benefit from batching.
    # Prosody (pyworld) is CPU-bound → runs in parallel via ThreadPool.
    # Audio loading is I/O-bound → prefetched in background.
    
    BATCH_SIZE = 8  # 4× larger — GPU handles batches well
    prosody_on_gpu = prosody_backend in ("rmvpe", "crepe")
    success = 0
    failed = 0

    from concurrent.futures import ThreadPoolExecutor
    import queue

    def load_audio(idx):
        """Load + resample single audio file (runs in thread)."""
        try:
            row = dataset[idx]
            audio, sr = torchaudio.load(row["audio_path"])
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            return idx, audio, sr, None
        except Exception as e:
            return idx, None, None, e

    def extract_prosody_cpu(audio, sr):
        """Extract prosody on CPU (pyworld/crepe) — can overlap with GPU work."""
        try:
            p_raw, p_mask = prosody_enc.extract_raw(audio, sr=sr)
            return p_raw.squeeze(0).cpu(), p_mask.squeeze(0).cpu()
        except Exception:
            return None, None

    # Prefetch audio in background (4 batches ahead)
    prefetch_q = queue.Queue(maxsize=BATCH_SIZE * 4)

    def prefetch_worker():
        for idx in indices:
            prefetch_q.put(load_audio(idx))
        prefetch_q.put(None)  # sentinel

    import threading
    prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
    prefetch_thread.start()

    # Background save — non-blocking disk I/O, GPU never waits for writes
    save_q = queue.Queue(maxsize=BATCH_SIZE * 8)

    def save_worker():
        while True:
            item = save_q.get()
            if item is None:
                save_q.task_done()
                break
            out_path, save_dict = item
            try:
                # fp16 to save disk, BUT keep prosody_raw AND speaker_raw in fp32.
                # Prosody carries channel 5 = log_f0_absolute (values ~4.4–6.0), where
                # fp16's fixed significant-digit budget gives a coarser ULP than on the
                # normalized channels (~4.5× larger absolute error). That channel feeds
                # the per-frame prosody_direct path and the global-stats / duration
                # signals, so we avoid quantizing it.
                # speaker_raw is kept fp32 too: it's the WavLM-SV embedding that carries
                # the speaker identity AND (empirically) the absolute pitch level
                # (linear-probe R²≈0.71), which AdaLN/fusion use to set the target's
                # register. fp16 would quantize that level signal; keeping it fp32
                # preserves it. Both tensors are small, so cache size is barely affected.
                # emotion_* stay fp16 (learned embedding, ~3 sig-digits is fine there).
                _FP32_KEYS = {"prosody_raw", "speaker_raw"}
                save_dict_f16 = {
                    k: (v.half() if isinstance(v, torch.Tensor)
                        and v.dtype == torch.float32 and k not in _FP32_KEYS else v)
                    for k, v in save_dict.items()
                }
                torch.save(save_dict_f16, out_path)
            except Exception as e:
                print(f"  [SAVE FAIL] {out_path}: {e}")
            finally:
                save_q.task_done()

    save_thread = threading.Thread(target=save_worker, daemon=True)
    save_thread.start()

    # Process in batches
    batch_data = []  # [(idx, audio, sr), ...]
    pbar = tqdm(total=len(indices), desc="Extracting embeddings")

    def flush_batch(batch):
        """Process a batch: GPU encoders batched, prosody overlaps with GPU."""
        nonlocal success, failed
        if not batch:
            return

        # Sort by audio length — minimises padding waste for emotion batching
        batch = sorted(batch, key=lambda x: x[1].shape[-1])

        # 1. Resample all audio to 16kHz once — speaker/emotion encoders use 16kHz
        #    Avoids repeated resampling inside each encoder call
        sr_orig = batch[0][2]
        if sr_orig != 16000:
            resampler = torchaudio.transforms.Resample(sr_orig, 16000).to(device)
        else:
            resampler = None

        audios_16k = []
        for _, audio, _ in batch:
            a = audio.to(device)
            if resampler is not None:
                a = resampler(a)
            audios_16k.append(a)

        max_len_16k = max(a.shape[-1] for a in audios_16k)
        # True (un-padded) lengths at 16kHz — passed to speaker/emotion encoders so
        # they mask out padding when pooling. Without this, WavLM-SV and emotion2vec
        # average over the padding zeros, blurring the speaker embedding and
        # contaminating emotion frames, AND mismatch inference (single un-padded file).
        lengths_16k = [a.shape[-1] for a in audios_16k]
        padded_16k = torch.zeros(len(batch), 1, max_len_16k, device=device)
        for j, a in enumerate(audios_16k):
            padded_16k[j, :, :a.shape[-1]] = a
        lengths_tensor = torch.tensor(lengths_16k, device=device, dtype=torch.long)

        # 2. Batched speaker extraction (lengths mask out padding in pooling)
        with torch.no_grad():
            try:
                spk_raws = speaker_enc.extract_raw(padded_16k, sr=16000, lengths=lengths_tensor)
            except Exception:
                spk_raws = []
                for a in audios_16k:
                    try:
                        s = speaker_enc.extract_raw(a, sr=16000)
                        spk_raws.append(s.squeeze(0))
                    except Exception:
                        spk_raws.append(torch.zeros(speaker_enc.raw_dim, device=device))
                spk_raws = torch.stack(spk_raws)

        # 3. Emotion: pass full padded batch at once
        #    _extract_hf handles (B, T) natively via feature_extractor(padding=True)
        #    _extract_funasr loops internally but avoids Python dispatch overhead
        emo_globals, emo_frames = [], []
        with torch.no_grad():
            try:
                # Pass lengths: padded_16k is equal-length, so the feature
                # extractor's own mask would be all-ones (contaminating global mean
                # + frames with padding). Explicit lengths fix this. The returned
                # frame_mask gives each sample's true valid frame count (accounts
                # for conv downsampling) → exact slicing, no length-ratio guess.
                eg_batch, ef_batch, mask_batch = emotion_enc.extract_raw_with_mask(
                    padded_16k, sr=16000, lengths=lengths_tensor
                )
                for j in range(len(batch)):
                    emo_globals.append(eg_batch[j].cpu())
                    if ef_batch is not None:
                        true_frames = max(1, int(mask_batch[j].sum().item()))
                        emo_frames.append(ef_batch[j, :true_frames].cpu())
                    else:
                        emo_frames.append(None)
            except Exception:
                # Fallback: per-sample on UNPADDED audio (no contamination)
                for j, a in enumerate(audios_16k):
                    try:
                        eg, ef = emotion_enc.extract_raw(a.unsqueeze(0), sr=16000)
                        emo_globals.append(eg.squeeze(0).cpu())
                        emo_frames.append(
                            ef.squeeze(0).cpu() if ef is not None else None
                        )
                    except Exception:
                        emo_globals.append(torch.zeros(emotion_enc.raw_dim))
                        emo_frames.append(None)

        # Free GPU memory before prosody (RMVPE also uses GPU)
        del padded_16k, audios_16k
        torch.cuda.empty_cache()

        # 4. Prosody: CPU backends run in parallel threads
        #    GPU backends (RMVPE/crepe) run sequentially to avoid VRAM contention
        if prosody_on_gpu:
            prosody_results = []
            with torch.no_grad():
                for j, (_, audio, sr) in enumerate(batch):
                    p_raw, p_mask = prosody_enc.extract_raw(audio.to(device), sr=sr)
                    prosody_results.append((p_raw.squeeze(0).cpu(), p_mask.squeeze(0).cpu()))
        else:
            n_workers = min(os.cpu_count() or 4, len(batch))
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                prosody_futures = [
                    pool.submit(extract_prosody_cpu, audio, sr)
                    for _, audio, sr in batch
                ]
                prosody_results = [f.result() for f in prosody_futures]

        # 5. Queue saves — non-blocking, save_worker handles disk I/O
        for j, (idx, _, _) in enumerate(batch):
            out_path = os.path.join(output_dir, f"{idx}.pt")
            try:
                save_dict = {
                    "speaker_raw": spk_raws[j].cpu() if torch.is_tensor(spk_raws) else spk_raws[j],
                    "emotion_global_raw": emo_globals[j],
                }
                if emo_frames[j] is not None:
                    save_dict["emotion_frame_raw"] = emo_frames[j]
                p_raw, p_mask = prosody_results[j]
                save_dict["prosody_raw"] = p_raw
                save_dict["prosody_mask"] = p_mask
                save_q.put((out_path, save_dict))
                success += 1
            except Exception as e:
                print(f"  [FAIL] Sample {idx}: {e}")
                save_q.put((out_path, {
                    "speaker_raw": torch.zeros(speaker_enc.raw_dim, dtype=torch.float32),  # fp32 (cached fp32)
                    "emotion_global_raw": torch.zeros(emotion_enc.raw_dim, dtype=torch.float16),
                    "emotion_frame_raw": None,
                    "prosody_raw": None,
                    "prosody_mask": None,
                }))
                failed += 1
            pbar.update(1)

    # Main loop: read from prefetch queue, batch, flush
    while True:
        item = prefetch_q.get()
        if item is None:
            break
        idx, audio, sr, err = item
        if err is not None or audio is None:
            print(f"  [FAIL] Sample {idx}: {err}")
            out_path = os.path.join(output_dir, f"{idx}.pt")
            save_q.put((out_path, {
                "speaker_raw": torch.zeros(speaker_enc.raw_dim, dtype=torch.float32),  # fp32 (cached fp32)
                "emotion_global_raw": torch.zeros(emotion_enc.raw_dim, dtype=torch.float16),
                "emotion_frame_raw": None,
                "prosody_raw": None,
                "prosody_mask": None,
            }))
            failed += 1
            pbar.update(1)
            continue

        batch_data.append((idx, audio, sr))
        if len(batch_data) >= BATCH_SIZE:
            flush_batch(batch_data)
            batch_data = []

    # Flush remaining
    flush_batch(batch_data)
    # Signal save_worker to stop, then wait for all pending saves
    save_q.put(None)
    save_q.join()
    save_thread.join(timeout=30)
    pbar.close()
    prefetch_thread.join(timeout=5)
    
    print(f"\nDone! Success: {success} | Failed: {failed} | Skipped: {skipped}")
    
    # Save extraction metadata
    meta = {
        "total_samples": len(dataset),
        "speaker_backend": speaker_backend,
        "speaker_raw_dim": speaker_enc.raw_dim,
        "emotion_backend": emotion_backend,
        "emotion_raw_dim": emotion_enc.raw_dim,
        "prosody_backend": prosody_backend,
        "prosody_raw_dim": prosody_enc.raw_dim,
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
            # Using torchaudio sox effects
            try:
                augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
                    audio.unsqueeze(0) if audio.dim() == 1 else audio,
                    sr,
                    effects,
                )
                audio = augmented.squeeze(0)
            except Exception:
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
    
    # Stage 2: embeddings
    parser.add_argument("--dataset_dir", type=str, help="Path to prepared dataset")
    parser.add_argument("--embedding_dir", type=str, help="Path to save/load embeddings")
    parser.add_argument("--speaker_backend", type=str, default="wavlm_sv",
                        choices=["wavlm_sv", "ecapa_tdnn", "resemblyzer"])
    parser.add_argument("--emotion_backend", type=str, default="emotion2vec_base",
                        choices=["emotion2vec_base", "emotion2vec_plus", "wav2vec2_ser", "hubert_ser", "emotion2vec_onnx"])
    parser.add_argument("--prosody_backend", type=str, default="dio",
                        choices=["dio", "harvest", "crepe", "rmvpe"])
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
            prosody_backend=args.prosody_backend,
            device=args.device,
            resume=args.resume,
        )
    
    if args.stage in ("verify", "all"):
        ds_dir = args.dataset_dir or args.output_dir
        emb_dir = args.embedding_dir or os.path.join(args.output_dir, "embeddings")
        verify_dataset(ds_dir, emb_dir)


if __name__ == "__main__":
    main()
