#!/usr/bin/env python3
"""
Prepare Multilingual + Emotion Dataset
========================================
Подготовка данных для файнтюнинга:
1. Обработка мультиязычных аудио-данных (ресемплинг, нормализация)
2. Генерация metadata.csv в формате: audio_path|text|language|duration|emotion
3. Построение multilingual vocab.txt
4. Разделение на train/val

Использование:
    python scripts/prepare_multilingual_dataset.py \
        --data_dirs /data/emilia_en /data/emilia_zh /data/your_lang \
        --emotion_datasets /data/esd /data/ravdess \
        --output_dir /data/prepared \
        --target_sr 24000 \
        --build_vocab
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from collections import Counter
import json

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from f5_tts_enhanced.data.multilingual_vocab import (
    build_multilingual_vocab,
    save_vocab,
    detect_language,
)


def process_audio_file(
    input_path: str,
    output_path: str,
    target_sr: int = 24000,
    max_duration: float = 30.0,
    min_duration: float = 0.3,
    normalize: bool = True,
) -> float | None:
    """
    Обработка одного аудио файла.
    Returns: duration в секундах или None если файл невалидный.
    """
    try:
        audio, sr = torchaudio.load(input_path)
    except Exception as e:
        print(f"  [SKIP] Cannot load {input_path}: {e}")
        return None

    # Mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Resample
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)

    duration = audio.shape[1] / target_sr

    # Filter by duration
    if duration < min_duration or duration > max_duration:
        return None

    # Normalize
    if normalize:
        peak = audio.abs().max()
        if peak > 0:
            audio = audio / peak * 0.95

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, audio, target_sr)

    return duration


def scan_dataset_with_metadata(
    data_dir: str,
    language: str = "auto",
) -> list:
    """
    Сканирует директорию датасета и собирает метаданные.

    Поддерживаемые форматы:
    1. Emilia-style: audio_path|text в metadata файле
    2. LJSpeech-style: wav_id|text|normalized_text
    3. Flat directory: .wav + .txt пары
    """
    data_dir = Path(data_dir)
    samples = []
    #["metadata.csv", "metadata.txt", "metadata.list", "transcript.txt", "manifest.jsonl"]
    # Ищем metadata файлы
    for meta_name in ["manifest.jsonl"]:
        meta_path = data_dir / meta_name
        print(meta_path)
        #if os.path.exists(meta_path):
        print("meta_path exist")
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "manifest.jsonl" in meta_name:
                    print(meta_name)
                    record = json.loads(line)
                    audio_file = record["audio_path"].strip()
                    text = record["text"].strip()
                    audio_path = data_dir / audio_file
                    lang = record["language"]
                    samples.append({
                        "audio_path": str(audio_path),
                        "text": text,
                        "language": lang,
                    })
                else:
                    parts = line.split("|")
                    if len(parts) >= 2:
                        audio_file = parts[0].strip()
                        text = parts[1].strip()

                        # Resolve path
                        audio_path = data_dir / audio_file
                        if not audio_path.suffix:
                            audio_path = audio_path.with_suffix(".wav")

                        lang = language if language != "auto" else detect_language(text)

                        samples.append({
                            "audio_path": str(audio_path),
                            "text": text,
                            "language": lang,
                        })
            break

    # Fallback: .wav + .txt пары
    if not samples:
        for wav_path in sorted(data_dir.rglob("*.wav")):
            txt_path = wav_path.with_suffix(".txt")
            if txt_path.exists():
                text = txt_path.read_text(encoding="utf-8").strip()
                lang = language if language != "auto" else detect_language(text)
                samples.append({
                    "audio_path": str(wav_path),
                    "text": text,
                    "language": lang,
                })

    return samples


def prepare_dataset(args):
    """Основная функция подготовки данных."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    lang_counter = Counter()
    char_set = set()

    # --- Обработка мультиязычных данных ---
    print("\n=== Processing multilingual datasets ===")
    for data_dir in (args.data_dirs or []):
        print(f"\nScanning: {data_dir}")
        samples = scan_dataset_with_metadata(data_dir)
        print(f"  Found {len(samples)} samples")

        processed = 0
        for i, sample in enumerate(samples):
            if not os.path.exists(sample["audio_path"]):
                continue

            # Определяем output path
            rel_name = f"{sample['language']}_{processed:06d}.mp3"
            out_audio = str(output_dir / "mp3" / rel_name)

            duration = process_audio_file(
                sample["audio_path"],
                out_audio,
                target_sr=args.target_sr,
                max_duration=args.max_duration,
                min_duration=args.min_duration,
            )

            if duration is not None:
                all_samples.append({
                    "audio_path": f"mp3/{rel_name}",
                    "text": sample["text"],
                    "language": sample["language"],
                    "duration": f"{duration:.3f}",
                    "emotion": "",
                })
                lang_counter[sample["language"]] += 1
                char_set.update(sample["text"])
                processed += 1

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{len(samples)}...")

        print(f"  → Kept {processed} samples")

    # --- Обработка emotional данных ---
    if args.emotion_datasets:
        print("\n=== Processing emotion datasets ===")
        for emo_dir in args.emotion_datasets:
            print(f"\nScanning emotion: {emo_dir}")
            # Emotion datasets обрабатываются для emotion extractor (16kHz)
            # и для основного TTS (24kHz)
            emo_samples = scan_emotion_dataset(emo_dir)
            print(f"  Found {len(emo_samples)} samples")

            for sample in emo_samples:
                if not os.path.exists(sample["audio_path"]):
                    continue

                rel_name = f"emo_{sample['emotion']}_{lang_counter.get('en', 0):06d}.wav"
                out_audio = str(output_dir / "wavs" / rel_name)

                duration = process_audio_file(
                    sample["audio_path"],
                    out_audio,
                    target_sr=args.target_sr,
                )

                if duration is not None:
                    all_samples.append({
                        "audio_path": f"wavs/{rel_name}",
                        "text": sample.get("text", ""),
                        "language": sample.get("language", "en"),
                        "duration": f"{duration:.3f}",
                        "emotion": sample.get("emotion", ""),
                    })
                    lang_counter[sample.get("language", "en")] += 1

    # --- Сохранение metadata ---
    print(f"\n=== Saving metadata ({len(all_samples)} samples) ===")

    # Train/val split (95/5)
    import random
    random.shuffle(all_samples)
    val_count = max(int(len(all_samples) * 0.05), 50)
    train_samples = all_samples[val_count:]
    val_samples = all_samples[:val_count]

    save_metadata(train_samples, output_dir / "metadata.csv")
    save_metadata(val_samples, output_dir / "metadata_val.csv")

    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val:   {len(val_samples)} samples")

    # Language distribution
    print(f"\n  Language distribution:")
    for lang, count in lang_counter.most_common():
        print(f"    {lang}: {count}")

    # --- Build vocab ---
    if args.build_vocab:
        print(f"\n=== Building multilingual vocab ===")
        vocab = build_multilingual_vocab(
            include_ipa=False,
            include_chinese_chars=True,
            chinese_char_count=5000,
        )
        # Добавить символы из данных, которых нет в стандартном vocab
        vocab_set = set(vocab)
        extra_chars = char_set - vocab_set
        if extra_chars:
            print(f"  Adding {len(extra_chars)} extra chars from dataset")
            vocab.extend(sorted(extra_chars))

        vocab_path = output_dir / "vocab.txt"
        save_vocab(vocab, str(vocab_path))

    print(f"\n✅ Dataset prepared in {output_dir}")


def scan_emotion_dataset(data_dir: str) -> list:
    """Сканирует ESD / RAVDESS / CREMA-D датасеты."""
    data_dir = Path(data_dir)
    samples = []

    EMOTION_MAP_DIR = {
        "Neutral": "neutral", "neutral": "neutral",
        "Happy": "happy", "happy": "happy",
        "Sad": "sad", "sad": "sad",
        "Angry": "angry", "angry": "angry",
        "Surprise": "surprise", "surprise": "surprise",
    }

    for wav_path in sorted(data_dir.rglob("*.wav")):
        emotion = None
        text = ""

        # Check parent dir for emotion label
        parent = wav_path.parent.name
        if parent in EMOTION_MAP_DIR:
            emotion = EMOTION_MAP_DIR[parent]

        # Check for transcript
        txt_path = wav_path.with_suffix(".txt")
        if txt_path.exists():
            text = txt_path.read_text(encoding="utf-8").strip()

        if emotion:
            samples.append({
                "audio_path": str(wav_path),
                "text": text,
                "language": "en",
                "emotion": emotion,
            })

    return samples


def save_metadata(samples: list, path: Path):
    """Сохранить metadata.csv."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        for sample in samples:
            line = "|".join([
                sample["audio_path"],
                sample["text"],
                sample["language"],
                sample["duration"],
                sample.get("emotion", ""),
            ])
            f.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", nargs="*", default=[],
                        help="Paths to multilingual speech datasets")
    parser.add_argument("--emotion_datasets", nargs="*", default=[],
                        help="Paths to emotion speech datasets (ESD, RAVDESS, etc.)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for prepared dataset")
    parser.add_argument("--target_sr", type=int, default=24000)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--min_duration", type=float, default=0.3)
    parser.add_argument("--build_vocab", action="store_true",
                        help="Build multilingual vocab.txt")
    args = parser.parse_args()

    prepare_dataset(args)
