"""
Enhanced Dataset for F5-TTS — loads audio + text + cached speaker/emotion embeddings.

Two modes:
  1. Precomputed cache: loads pre-extracted embeddings from disk (fast).
  2. Online extraction: returns raw audio for on-the-fly extraction (flexible).
"""

from __future__ import annotations

import os
from pathlib import Path

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
        
        self.valid_indices = []
        for idx in range(len(self.data)):
            row = self.data[idx]
            duration = self.durations[idx] if self.durations is not None else row["duration"]
            if 0.3 <= duration <= 30:
                self.valid_indices.append(idx)
        
        if not self.valid_indices:
            raise RuntimeError("No valid samples found in CustomDataset")
        
    def _resolve_index(self, index):
        return self.valid_indices[index]
    
    def get_frame_len(self, index: int) -> float:
        raw_index = self._resolve_index(index)
        if self.durations is not None:
            return int(self.durations[raw_index] * self.target_sample_rate / self.hop_length)
        return int(self.data[raw_index]["duration"] * self.target_sample_rate / self.hop_length)
    
    
    def get_speaker(self, index):
        row = self.data[self._resolve_index(index)]
        if "speaker" in row:
            return row["speaker"]
        if "speaker_id" in row:
            return row["speaker_id"]
        if "audio_path" in row:
            return Path(row["audio_path"]).parent.name
        return "default"
    
    
    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, index: int) -> dict:
        
        row = self.data[self._resolve_index(index)]
        
        #original_index = index
        #for _ in range(min(100, len(self.data))):
         #   row = self.data[index]
          #  dur = row["duration"]
           # if 0.3 <= dur <= 30:
            #    break
            #index = (index + 1) % len(self.data)

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
            audio = None
        else:
            audio, sr = torchaudio.load(row["audio_path"])
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            if sr != self.target_sample_rate:
                audio = torchaudio.transforms.Resample(sr, self.target_sample_rate)(audio)
            mel_spec = self.mel_spectrogram(audio).squeeze(0)

        result = {"mel_spec": mel_spec, "text": row["text"]}

        if self.embedding_cache_dir is not None:
            raw_idx = self._resolve_index(index)
            cache_idx = self.embedding_index_map[raw_idx] if self.embedding_index_map is not None else raw_idx
            emb_path = os.path.join(self.embedding_cache_dir, f"{cache_idx}.pt")
            if os.path.exists(emb_path):
                emb = torch.load(emb_path, map_location="cpu", weights_only=True)
                speaker_raw = emb.get("speaker_raw")
                emotion_global_raw = emb.get("emotion_global_raw")
                emotion_frame_raw = emb.get("emotion_frame_raw")
            else:
                speaker_raw = None
                emotion_global_raw = None
                emotion_frame_raw = None

            result["speaker_raw"] = speaker_raw
            result["emotion_global_raw"] = emotion_global_raw
            result["emotion_frame_raw"] = emotion_frame_raw
            result["speaker_present"] = speaker_raw is not None
            result["emotion_global_present"] = emotion_global_raw is not None

        if not self.preprocessed_mel and audio is not None:
            result["raw_audio"] = audio.squeeze(0)
            result["sample_rate"] = self.target_sample_rate

        return result


def collate_fn(batch: list[dict]) -> dict:
    """Collate mel specs, text, and embeddings with explicit masks."""
    mel_specs = []
    for item in batch:
        m = item["mel_spec"]
        if m.ndim == 3 and m.shape[0] == 1:
            m = m.squeeze(0)
        mel_specs.append(m)
    mel_lengths = torch.LongTensor([s.shape[-1] for s in mel_specs])
    max_len = int(mel_lengths.amax().item())
    mel_specs = torch.stack([F.pad(s, (0, max_len - s.size(-1))) for s in mel_specs])

    result = {
        "mel": mel_specs,
        "mel_lengths": mel_lengths,
        "text": [item["text"] for item in batch],
    }

    if "speaker_raw" in batch[0] or any("speaker_raw" in item for item in batch):
        valid = [item.get("speaker_raw") for item in batch if item.get("speaker_raw") is not None]
        result["speaker_present"] = torch.tensor([
            bool(item.get("speaker_present", item.get("speaker_raw") is not None))
            for item in batch
        ], dtype=torch.bool)
        if valid:
            dim = valid[0].shape[-1]
            result["speaker_raw"] = torch.stack([
                item["speaker_raw"] if item.get("speaker_raw") is not None else torch.zeros(dim)
                for item in batch
            ])

    if "emotion_global_raw" in batch[0] or any("emotion_global_raw" in item for item in batch):
        valid = [item.get("emotion_global_raw") for item in batch if item.get("emotion_global_raw") is not None]
        result["emotion_global_present"] = torch.tensor([
            bool(item.get("emotion_global_present", item.get("emotion_global_raw") is not None))
            for item in batch
        ], dtype=torch.bool)
        if valid:
            dim = valid[0].shape[-1]
            result["emotion_global_raw"] = torch.stack([
                item["emotion_global_raw"] if item.get("emotion_global_raw") is not None else torch.zeros(dim)
                for item in batch
            ])

    if "emotion_frame_raw" in batch[0] or any("emotion_frame_raw" in item for item in batch):
        frames = [item.get("emotion_frame_raw") for item in batch]
        valid = [f for f in frames if f is not None]
        frame_lengths = torch.tensor([0 if f is None else f.shape[0] for f in frames], dtype=torch.long)
        result["emotion_frame_lengths"] = frame_lengths
        max_f = int(frame_lengths.max().item()) if frame_lengths.numel() else 0
        if max_f > 0:
            result["emotion_frame_mask"] = torch.arange(max_f).unsqueeze(0) < frame_lengths.unsqueeze(1)
        # else: all frames None — don't set mask (validate_raw requires both or neither)
        if valid:
            raw_dim = valid[0].shape[-1]
            padded = []
            for f in frames:
                if f is None:
                    f = torch.zeros(max_f, raw_dim)
                else:
                    f = F.pad(f, (0, 0, 0, max_f - f.shape[0]))
                padded.append(f)
            result["emotion_frame_raw"] = torch.stack(padded)

    if any("raw_audio" in item for item in batch):
        audios = [item.get("raw_audio") for item in batch]
        valid_audios = [a for a in audios if a is not None]
        if valid_audios:
            max_a = max(a.shape[0] for a in valid_audios)
            padded = []
            for a in audios:
                if a is None:
                    a = torch.zeros(max_a)
                else:
                    a = F.pad(a, (0, max_a - a.shape[0]))
                padded.append(a)
            result["raw_audio"] = torch.stack(padded)
            result["raw_audio_present"] = torch.tensor([a is not None for a in audios], dtype=torch.bool)
            sr_item = next((item for item in batch if "sample_rate" in item), None)
            if sr_item is not None:
                result["sample_rate"] = sr_item["sample_rate"]

    return result
