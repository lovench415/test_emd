"""
Multilingual Emotion TTS Dataset
==================================
Dataset и DataLoader для файнтюнинга F5-TTS Enhanced.

КРИТИЧНО: mel выходит в формате (T, mel_dim) = channels LAST,
чтобы совпадать с F5-TTS DiT: x shape (B, N, mel_dim).

Формат metadata.csv (pipe-separated):
    audio_path|text|language|duration|emotion
    wavs/en_000001.wav|Hello world|en|2.5|happy
"""

import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, Sampler, DataLoader


# =========================================================================
# Language map (matches adapters.py LanguageEmbedding)
# =========================================================================

LANG2ID = {
    "en": 0, "zh": 1, "ru": 2, "de": 3, "fr": 4,
    "es": 5, "ja": 6, "ko": 7, "it": 8, "pt": 9,
    "ar": 10, "hi": 11, "other": 12,
}


def lang_to_id(lang: str) -> int:
    return LANG2ID.get(lang.lower()[:2], LANG2ID["other"])


# =========================================================================
# Mel Spectrogram (matching F5-TTS / Vocos)
# =========================================================================

class MelSpectrogramExtractor:
    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 100,
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
            power=1.0,
            norm="slaney",
            mel_scale="slaney",
        )

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: (1, samples) or (samples,)
        Returns:
            mel: (T, n_mels) — channels LAST (matching F5-TTS DiT)
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        mel = self.mel_transform(audio).squeeze(0)  # (n_mels, T)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        mel = mel.transpose(0, 1)  # → (T, n_mels) channels LAST
        return mel


# =========================================================================
# Dataset
# =========================================================================

class TTSDataset(Dataset):
    """
    Возвращает:
        mel:        (T, n_mels)   — channels LAST
        text_ids:   list[int]
        lang_id:    int
        audio_path: str           — для emotion2vec
        mel_frames: int
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
        self.hop_length = hop_length
        self.max_frames = int(max_duration * target_sr / hop_length)
        self.min_frames = int(min_duration * target_sr / hop_length)

        self.mel_extractor = MelSpectrogramExtractor(
            sample_rate=target_sr, n_mels=n_mels, hop_length=hop_length,
        )
        self.vocab = self._load_vocab(vocab_path)
        self.pad_id = self.vocab.get("<pad>", 0)
        self.samples = self._load_metadata(metadata_path)
        print(f"[Dataset] {len(self.samples)} samples from {metadata_path}")

    def _load_vocab(self, path: str) -> dict:
        vocab = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                char = line.strip()
                if char:
                    vocab[char] = len(vocab)
        vocab.setdefault("<pad>", len(vocab))
        vocab.setdefault("<unk>", len(vocab))
        return vocab

    def _load_metadata(self, path: str) -> list:
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

                full_path = str(self.data_root / audio_path)
                if not os.path.exists(full_path):
                    continue

                mel_frames = int(duration * self.target_sr / self.hop_length)
                if self.max_frames and mel_frames > self.max_frames:
                    continue
                if mel_frames < self.min_frames:
                    continue

                samples.append({
                    "audio_path": full_path,
                    "text": text,
                    "language": language,
                    "mel_frames": mel_frames,
                })

        samples.sort(key=lambda x: x["mel_frames"])
        return samples

    def text_to_ids(self, text: str) -> list:
        unk = self.vocab.get("<unk>", 0)
        return [self.vocab.get(c, unk) for c in text]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        audio, sr = torchaudio.load(sample["audio_path"])
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            audio = torchaudio.functional.resample(audio, sr, self.target_sr)

        mel = self.mel_extractor(audio)  # (T, n_mels) channels LAST
        text_ids = self.text_to_ids(sample["text"])
        lang_id = lang_to_id(sample["language"])

        return {
            "mel": mel,                           # (T, n_mels) channels LAST
            "text_ids": text_ids,
            "lang_id": lang_id,
            "audio_path": sample["audio_path"],
            "mel_frames": mel.shape[0],           # T
            "text_len": len(text_ids),
        }


# =========================================================================
# Frame-Based Dynamic Batch Sampler
# =========================================================================

class FrameBatchSampler(Sampler):
    """Формирует батчи по суммарному числу mel-фреймов."""

    def __init__(
        self,
        dataset: TTSDataset,
        max_frames: int = 3200,
        max_samples: int = 64,
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
        self._batches = self._build_batches()

    def _build_batches(self) -> list:
        indices = list(range(len(self.dataset)))
        batches = []
        batch, frames = [], 0
        for idx in indices:
            sf = self.dataset.samples[idx]["mel_frames"]
            if (frames + sf > self.max_frames
                    or len(batch) >= self.max_samples) and batch:
                batches.append(batch)
                batch, frames = [], 0
            batch.append(idx)
            frames += sf
        if batch and not self.drop_last:
            batches.append(batch)
        return batches

    def __iter__(self):
        batches = self._batches.copy()
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        yield from batches
        self.epoch += 1

    def __len__(self):
        return len(self._batches)

    def set_epoch(self, epoch: int):
        self.epoch = epoch


# =========================================================================
# Collate — all tensors channels LAST
# =========================================================================

def collate_fn(batch: list) -> dict:
    """
    Returns:
        mel:         (B, T_max, n_mels)  — channels LAST
        mel_lengths: (B,)
        text_ids:    (B, L_max)
        text_lengths:(B,)
        lang_ids:    (B,)
        audio_paths: list[str]
        mask:        (B, T_max)  — True for valid frames
    """
    n_mels = batch[0]["mel"].shape[1]
    max_mel = max(item["mel_frames"] for item in batch)
    max_txt = max(item["text_len"] for item in batch)
    B = len(batch)

    mel_padded = torch.zeros(B, max_mel, n_mels)          # channels LAST
    text_padded = torch.zeros(B, max_txt, dtype=torch.long)
    mel_lengths = torch.zeros(B, dtype=torch.long)
    text_lengths = torch.zeros(B, dtype=torch.long)
    lang_ids = torch.zeros(B, dtype=torch.long)
    audio_paths = []

    for i, item in enumerate(batch):
        ml = item["mel"].shape[0]
        tl = item["text_len"]
        mel_padded[i, :ml, :] = item["mel"]
        text_padded[i, :tl] = torch.tensor(item["text_ids"], dtype=torch.long)
        mel_lengths[i] = ml
        text_lengths[i] = tl
        lang_ids[i] = item["lang_id"]
        audio_paths.append(item["audio_path"])

    mask = torch.arange(max_mel).unsqueeze(0) < mel_lengths.unsqueeze(1)

    return {
        "mel": mel_padded,
        "mel_lengths": mel_lengths,
        "text_ids": text_padded,
        "text_lengths": text_lengths,
        "lang_ids": lang_ids,
        "audio_paths": audio_paths,
        "mask": mask,
    }


# =========================================================================
# Factory
# =========================================================================

def create_dataloader(
    metadata_path: str,
    data_root: str,
    vocab_path: str,
    max_frames_per_batch: int = 3200,
    max_samples_per_batch: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs,
) -> tuple:
    dataset = TTSDataset(
        metadata_path=metadata_path,
        data_root=data_root,
        vocab_path=vocab_path,
        **dataset_kwargs,
    )
    sampler = FrameBatchSampler(
        dataset, max_frames=max_frames_per_batch,
        max_samples=max_samples_per_batch, shuffle=shuffle,
    )
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return dataset, loader
