"""
Enhanced Dataset for F5-TTS — loads audio + text + cached speaker/emotion embeddings.

Two modes:
  1. Precomputed cache: loads pre-extracted embeddings from disk (fast).
  2. Online extraction: returns raw audio for on-the-fly extraction (flexible).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as HFDataset_
from torch import nn
from torch.utils.data import Dataset

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default


class EnhancedDataset(Dataset):
    """Dataset with audio, text, and optionally cached embeddings."""

    def __init__(
        self,
        hf_dataset: HFDataset_,
        durations=None,
        target_sample_rate: int = 24_000,
        hop_length: int = 256,
        n_mel_channels: int = 100,
        n_fft: int = 1024,
        win_length: int = 1024,
        mel_spec_type: str = "vocos",
        preprocessed_mel: bool = False,
        mel_spec_module: nn.Module | None = None,
        embedding_cache_dir: str | None = None,
        embedding_index_map: list[int] | None = None,
        speaker_raw_dim: int = 512,
        emotion_raw_dim: int = 768,
    ):
        super().__init__()
        self.data = hf_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.preprocessed_mel = preprocessed_mel
        self.embedding_cache_dir = embedding_cache_dir
        # Maps dataset index → original cache file index.
        # When train/val split shuffles indices, cache files
        # keep their original numbering (0.pt, 1.pt, ...).
        self.embedding_index_map = embedding_index_map
        self.speaker_raw_dim = speaker_raw_dim
        self.emotion_raw_dim = emotion_raw_dim
        
        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index: int) -> float:
        dur = self.durations[index] if self.durations else self.data[index]["duration"]
        return dur * self.target_sample_rate / self.hop_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        # Skip invalid durations — try up to N neighbors before giving up
        original_index = index
        for attempts in range(min(100, len(self.data))):
            row = self.data[index]
            dur = row["duration"]
            if 0.3 <= dur <= 30:
                break
            index = (index + 1) % len(self.data)
        else:
            # All sampled entries invalid — log warning and use last attempted
            import warnings
            warnings.warn(
                f"EnhancedDataset: no valid sample found near index {original_index} "
                f"(checked {min(100, len(self.data))} neighbors). Using index {index} with dur={dur:.2f}s."
            )

        # Mel spectrogram
        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
        else:
            audio, sr = torchaudio.load(row["audio_path"])
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            if sr != self.target_sample_rate:
                audio = torchaudio.transforms.Resample(sr, self.target_sample_rate)(audio)
            mel_spec = self.mel_spectrogram(audio).squeeze(0)

        result = {"mel_spec": mel_spec, "text": row["text"]}

        # Cached embeddings
        if self.embedding_cache_dir is not None:
            # Resolve original index for cache file lookup
            cache_idx = (self.embedding_index_map[index]
                         if self.embedding_index_map is not None else index)
            emb_path = os.path.join(self.embedding_cache_dir, f"{cache_idx}.pt")
            if os.path.exists(emb_path):
                emb = torch.load(emb_path, map_location="cpu", weights_only=True)
                result["speaker_raw"] = emb.get("speaker_raw",
                                                 torch.zeros(self.speaker_raw_dim))
                result["emotion_global_raw"] = emb.get("emotion_global_raw",
                                                        torch.zeros(self.emotion_raw_dim))
                result["emotion_frame_raw"] = emb.get("emotion_frame_raw")
            else:
                result["speaker_raw"] = torch.zeros(self.speaker_raw_dim)
                result["emotion_global_raw"] = torch.zeros(self.emotion_raw_dim)
                result["emotion_frame_raw"] = None
        elif not self.preprocessed_mel:
            result["raw_audio"] = audio.squeeze(0)
            result["sample_rate"] = self.target_sample_rate

        return result


def collate_fn(batch: list[dict]) -> dict:
    """Collate mel specs, text, and embeddings with padding."""
    # Ensure mel specs are 2D (D, T) — handle both (1, D, T) and (D, T) cases
    mel_specs = []
    for item in batch:
        m = item["mel_spec"]
        if m.ndim == 3 and m.shape[0] == 1:
            m = m.squeeze(0)
        mel_specs.append(m)
    mel_lengths = torch.LongTensor([s.shape[-1] for s in mel_specs])
    max_len = mel_lengths.amax()
    mel_specs = torch.stack([F.pad(s, (0, max_len - s.size(-1))) for s in mel_specs])

    result = {
        "mel": mel_specs,
        "mel_lengths": mel_lengths,
        "text": [item["text"] for item in batch],
    }

    # Stack embeddings if present
    for key in ("speaker_raw", "emotion_global_raw"):
        if key in batch[0]:
            result[key] = torch.stack([item[key] for item in batch])

    if "emotion_frame_raw" in batch[0]:
        frames = [item.get("emotion_frame_raw") for item in batch]
        # Some samples may have None (extraction failed / cache miss).
        # Replace None with zero tensor inferred from non-None items.
        valid = [f for f in frames if f is not None]
        if valid:
            raw_dim = valid[0].shape[-1]
            frames = [f if f is not None else torch.zeros(1, raw_dim) for f in frames]
            max_f = max(f.shape[0] for f in frames)
            result["emotion_frame_raw"] = torch.stack(
                [F.pad(f, (0, 0, 0, max_f - f.shape[0])) for f in frames]
            )

    if "raw_audio" in batch[0]:
        audios = [item["raw_audio"] for item in batch]
        max_a = max(a.shape[0] for a in audios)
        result["raw_audio"] = torch.stack([F.pad(a, (0, max_a - a.shape[0])) for a in audios])
        result["sample_rate"] = batch[0]["sample_rate"]

    return result
