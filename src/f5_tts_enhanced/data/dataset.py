"""
Multilingual Emotion TTS Dataset
==================================
Dataset и DataLoader для файнтюнинга F5-TTS Enhanced.

Формат metadata.csv (pipe-separated):
    audio_path|text|language|duration|emotion
    wavs/en_000001.wav|Hello world|en|2.5|happy
    wavs/ru_000001.wav|Привет мир|ru|2.3|

Батчинг: по суммарному числу mel-фреймов (frame-based dynamic batching),
как в оригинальном F5-TTS.
"""

import csv
import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, Sampler, DataLoader

from f5_tts_enhanced.data.multilingual_vocab import detect_language


# =========================================================================
# Language map (must match adapters.py LanguageEmbedding)
# =========================================================================

LANG2ID = {
    "en": 0, "zh": 1, "ru": 2, "de": 3, "fr": 4,
    "es": 5, "ja": 6, "ko": 7, "it": 8, "pt": 9,
    "ar": 10, "hi": 11, "other": 12,
}


def lang_to_id(lang: str) -> int:
    return LANG2ID.get(lang.lower()[:2], LANG2ID["other"])


# =========================================================================
# Mel Spectrogram (matching F5-TTS)
# =========================================================================

class MelSpectrogramExtractor:
    """
    Mel spectrogram compatible with F5-TTS / Vocos.
    Использует torchaudio для извлечения.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 100,
        f_min: float = 0.0,
        f_max: float = None,
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max or sample_rate // 2,
            power=1.0,
            norm="slaney",
            mel_scale="slaney",
        )

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: (1, samples) or (samples,)
        Returns:
            mel: (n_mels, T)
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        mel = self.mel_transform(audio).squeeze(0)  # (n_mels, T)
        # Log mel
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel


# =========================================================================
# Dataset
# =========================================================================

class TTSDataset(Dataset):
    """
    Dataset для мультиязычного эмоционального TTS.

    Читает metadata.csv и загружает:
    - mel спектрограмму (для flow matching)
    - текст (в виде list[int] token ids)
    - язык (int)
    - путь к оригинальному аудио (для emotion extraction через emotion2vec)
    """

    def __init__(
        self,
        metadata_path: str,
        data_root: str,
        vocab_path: str,
        target_sr: int = 24000,
        n_mels: int = 100,
        hop_length: int = 256,
        max_duration: float = 30.0,
        min_duration: float = 0.3,
    ):
        self.data_root = Path(data_root)
        self.target_sr = target_sr
        self.max_frames = int(max_duration * target_sr / hop_length)
        self.min_frames = int(min_duration * target_sr / hop_length)

        # Mel extractor
        self.mel_extractor = MelSpectrogramExtractor(
            sample_rate=target_sr,
            n_mels=n_mels,
            hop_length=hop_length,
        )

        # Vocab (char → id)
        self.vocab = self._load_vocab(vocab_path)

        # Load metadata
        self.samples = self._load_metadata(metadata_path)
        print(f"[Dataset] Loaded {len(self.samples)} samples from {metadata_path}")

    def _load_vocab(self, path: str) -> dict:
        """Load vocab.txt → {char: id}."""
        vocab = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                char = line.strip()
                if char:
                    vocab[char] = len(vocab)
        # Special tokens
        if "<pad>" not in vocab:
            vocab["<pad>"] = len(vocab)
        if "<unk>" not in vocab:
            vocab["<unk>"] = len(vocab)
        return vocab

    def _load_metadata(self, path: str) -> list:
        """Load metadata.csv (pipe-separated)."""
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|")
                if len(parts) < 3:
                    continue

                audio_path = parts[0].strip()
                text = parts[1].strip()
                language = parts[2].strip()
                duration = float(parts[3].strip()) if len(parts) > 3 and parts[3].strip() else 0.0

                # Полный путь
                full_audio_path = str(self.data_root / audio_path)
                if not os.path.exists(full_audio_path):
                    continue

                # Duration в фреймах
                mel_frames = int(duration * self.target_sr / self.mel_extractor.hop_length)
                if mel_frames > self.max_frames or mel_frames < self.min_frames:
                    continue

                samples.append({
                    "audio_path": full_audio_path,
                    "text": text,
                    "language": language,
                    "duration": duration,
                    "mel_frames": mel_frames,
                })

        # Сортировка по длительности (для эффективного батчинга)
        samples.sort(key=lambda x: x["mel_frames"])
        return samples

    def text_to_ids(self, text: str) -> list:
        """Символьная токенизация (как в F5-TTS)."""
        unk_id = self.vocab.get("<unk>", 0)
        return [self.vocab.get(c, unk_id) for c in text]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load audio
        audio, sr = torchaudio.load(sample["audio_path"])
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            audio = torchaudio.functional.resample(audio, sr, self.target_sr)

        # Mel spectrogram
        mel = self.mel_extractor(audio)  # (n_mels, T)

        # Text
        text_ids = self.text_to_ids(sample["text"])

        # Language
        lang_id = lang_to_id(sample["language"])

        return {
            "mel": mel,                           # (n_mels, T)
            "text_ids": text_ids,                  # list[int]
            "lang_id": lang_id,                    # int
            "audio_path": sample["audio_path"],    # str — для emotion2vec
            "mel_frames": mel.shape[1],            # int
            "text_len": len(text_ids),             # int
        }


# =========================================================================
# Frame-Based Dynamic Batch Sampler (как в F5-TTS)
# =========================================================================

class FrameBatchSampler(Sampler):
    """
    Формирует батчи по суммарному числу mel-фреймов.

    Вместо фиксированного batch_size, каждый батч содержит ≤ max_frames фреймов.
    Это означает:
    - Короткие utterance'ы → больше в батче
    - Длинные utterance'ы → меньше в батче
    - VRAM usage более-менее постоянный

    Использует bucket-based подход: сортирует по длительности,
    формирует bucket'ы из соседних сэмплов, шаффлит bucket'ы.
    """

    def __init__(
        self,
        dataset: TTSDataset,
        max_frames: int = 3200,      # макс. суммарных фреймов в батче
        max_samples: int = 64,        # макс. сэмплов в батче
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.max_frames = max_frames
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self.seed = seed

        # Precompute batches
        self._batches = self._build_batches()

    def _build_batches(self) -> list:
        """Собрать батчи из индексов."""
        # Индексы уже отсортированы по mel_frames в dataset
        indices = list(range(len(self.dataset)))

        batches = []
        current_batch = []
        current_frames = 0

        for idx in indices:
            sample_frames = self.dataset.samples[idx]["mel_frames"]

            # Если добавление сэмпла превысит лимит — начать новый батч
            if (current_frames + sample_frames > self.max_frames
                    or len(current_batch) >= self.max_samples) and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_frames = 0

            current_batch.append(idx)
            current_frames += sample_frames

        if current_batch and not self.drop_last:
            batches.append(current_batch)

        return batches

    def __iter__(self):
        batches = self._batches.copy()
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(batches)
        for batch in batches:
            yield batch
        self.epoch += 1

    def __len__(self):
        return len(self._batches)

    def set_epoch(self, epoch: int):
        self.epoch = epoch


# =========================================================================
# Collate Function
# =========================================================================

def collate_fn(batch: list) -> dict:
    """
    Collate с padding по максимальной длине в батче.

    Returns dict:
        mel: (B, n_mels, T_max) — padded mel
        mel_lengths: (B,) — реальные длины
        text_ids: (B, L_max) — padded text
        text_lengths: (B,) — реальные длины текста
        lang_ids: (B,)
        audio_paths: list[str]
        mask: (B, T_max) — True для реальных фреймов
    """
    n_mels = batch[0]["mel"].shape[0]

    # Max lengths
    max_mel_len = max(item["mel_frames"] for item in batch)
    max_text_len = max(item["text_len"] for item in batch)
    B = len(batch)

    # Allocate tensors
    mel_padded = torch.zeros(B, n_mels, max_mel_len)
    text_padded = torch.zeros(B, max_text_len, dtype=torch.long)
    mel_lengths = torch.zeros(B, dtype=torch.long)
    text_lengths = torch.zeros(B, dtype=torch.long)
    lang_ids = torch.zeros(B, dtype=torch.long)
    audio_paths = []

    for i, item in enumerate(batch):
        mel_len = item["mel"].shape[1]
        text_len = item["text_len"]

        mel_padded[i, :, :mel_len] = item["mel"]
        text_padded[i, :text_len] = torch.tensor(item["text_ids"], dtype=torch.long)
        mel_lengths[i] = mel_len
        text_lengths[i] = text_len
        lang_ids[i] = item["lang_id"]
        audio_paths.append(item["audio_path"])

    # Mask: True for valid frames
    mask = torch.arange(max_mel_len).unsqueeze(0) < mel_lengths.unsqueeze(1)  # (B, T_max)

    return {
        "mel": mel_padded,              # (B, n_mels, T_max)
        "mel_lengths": mel_lengths,      # (B,)
        "text_ids": text_padded,         # (B, L_max)
        "text_lengths": text_lengths,    # (B,)
        "lang_ids": lang_ids,            # (B,)
        "audio_paths": audio_paths,      # list[str]
        "mask": mask,                    # (B, T_max)
    }


# =========================================================================
# Create DataLoader
# =========================================================================

def create_dataloader(
    metadata_path: str,
    data_root: str,
    vocab_path: str,
    max_frames_per_batch: int = 3200,
    max_samples_per_batch: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    target_sr: int = 24000,
    n_mels: int = 100,
    hop_length: int = 256,
    max_duration: float = 30.0,
) -> tuple:
    """
    Создаёт Dataset + DataLoader с frame-based батчингом.

    Returns:
        (dataset, dataloader)
    """
    dataset = TTSDataset(
        metadata_path=metadata_path,
        data_root=data_root,
        vocab_path=vocab_path,
        target_sr=target_sr,
        n_mels=n_mels,
        hop_length=hop_length,
        max_duration=max_duration,
    )

    sampler = FrameBatchSampler(
        dataset=dataset,
        max_frames=max_frames_per_batch,
        max_samples=max_samples_per_batch,
        shuffle=shuffle,
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return dataset, dataloader
