"""
Enhanced Dataset for F5-TTS
=============================

Extends the original CustomDataset to include speaker and emotion embeddings.

Two modes:
1. **Online extraction**: Extracts embeddings on-the-fly during training.
   Slower but requires no preprocessing.
   
2. **Precomputed cache**: Loads pre-extracted embeddings from disk.
   Faster training; requires a one-time preprocessing step.

The dataset returns:
    mel_spec:      (d, t) mel spectrogram
    text:          str
    speaker_raw:   (speaker_raw_dim,) raw speaker embedding (precomputed)
    emotion_global_raw: (emotion_raw_dim,) raw global emotion embedding
    emotion_frame_raw:  (T_frames, emotion_raw_dim) raw frame-level features
"""

from __future__ import annotations

import json
import os
from importlib.resources import files

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as HFDataset_
from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default


class EnhancedCustomDataset(Dataset):
    """
    Dataset that loads audio, text, and optionally precomputed
    speaker/emotion embeddings.
    
    If embedding cache dir is provided, loads precomputed embeddings.
    Otherwise, returns raw audio for online extraction.
    """

    def __init__(
        self,
        custom_dataset: HFDataset_,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
        # ── New: embedding cache ──
        embedding_cache_dir: str | None = None,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.preprocessed_mel = preprocessed_mel
        self.embedding_cache_dir = embedding_cache_dir

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

    def get_frame_len(self, index):
        if self.durations is not None:
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            text = row["text"]
            duration = row["duration"]
            if 0.3 <= duration <= 30:
                break
            index = (index + 1) % len(self.data)

        # Mel spectrogram
        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
        else:
            audio, source_sample_rate = torchaudio.load(audio_path)
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)
            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)  # (d, t)

        result = {
            "mel_spec": mel_spec,
            "text": text,
        }

        # Load precomputed embeddings if available
        if self.embedding_cache_dir is not None:
            emb_path = os.path.join(self.embedding_cache_dir, f"{index}.pt")
            if os.path.exists(emb_path):
                emb_data = torch.load(emb_path, map_location="cpu", weights_only=True)
                result["speaker_raw"] = emb_data.get("speaker_raw", torch.zeros(512))
                result["emotion_global_raw"] = emb_data.get("emotion_global_raw", torch.zeros(768))
                result["emotion_frame_raw"] = emb_data.get("emotion_frame_raw", None)
            else:
                result["speaker_raw"] = torch.zeros(512)
                result["emotion_global_raw"] = torch.zeros(768)
                result["emotion_frame_raw"] = None
        else:
            # Return raw audio for online extraction
            if not self.preprocessed_mel:
                result["raw_audio"] = audio.squeeze(0)  # (samples,)
                result["sample_rate"] = self.target_sample_rate

        return result


def enhanced_collate_fn(batch):
    """
    Collation function that handles mel specs, text, and embeddings.
    """
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)
    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    result = dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,
        text=text,
        text_lengths=text_lengths,
    )

    # Collate embeddings if present
    if "speaker_raw" in batch[0]:
        result["speaker_raw"] = torch.stack([item["speaker_raw"] for item in batch])

    if "emotion_global_raw" in batch[0]:
        result["emotion_global_raw"] = torch.stack([item["emotion_global_raw"] for item in batch])

    if "emotion_frame_raw" in batch[0] and batch[0]["emotion_frame_raw"] is not None:
        frame_feats = [item["emotion_frame_raw"] for item in batch]
        max_frames = max(f.shape[0] for f in frame_feats)
        padded_frames = [F.pad(f, (0, 0, 0, max_frames - f.shape[0])) for f in frame_feats]
        result["emotion_frame_raw"] = torch.stack(padded_frames)

    # Collate raw audio if present (for online extraction)
    if "raw_audio" in batch[0]:
        raw_audios = [item["raw_audio"] for item in batch]
        max_audio_len = max(a.shape[0] for a in raw_audios)
        padded_audios = [F.pad(a, (0, max_audio_len - a.shape[0])) for a in raw_audios]
        result["raw_audio"] = torch.stack(padded_audios)
        result["sample_rate"] = batch[0]["sample_rate"]

    return result


# ==================================================================
# Preprocessing script: extract and cache embeddings
# ==================================================================


def precompute_embeddings(
    dataset,
    speaker_encoder,
    emotion_encoder,
    output_dir: str,
    batch_size: int = 16,
    device: str = "cuda",
):
    """
    Precompute speaker and emotion embeddings for the entire dataset.
    Saves one .pt file per sample with:
        speaker_raw: (speaker_raw_dim,)
        emotion_global_raw: (emotion_raw_dim,)
        emotion_frame_raw: (T, emotion_raw_dim)
    """
    os.makedirs(output_dir, exist_ok=True)

    speaker_encoder = speaker_encoder.to(device).eval()
    emotion_encoder = emotion_encoder.to(device).eval()

    for i in tqdm(range(len(dataset)), desc="Precomputing embeddings"):
        item = dataset[i]
        out_path = os.path.join(output_dir, f"{i}.pt")
        if os.path.exists(out_path):
            continue

        if "raw_audio" in item:
            wav = item["raw_audio"].unsqueeze(0).to(device)
            sr = item["sample_rate"]

            with torch.no_grad():
                spk_raw = speaker_encoder.extract_raw_embedding(wav, sr)
                emo_feats = emotion_encoder.precompute_embeddings(wav, sr)

            torch.save({
                "speaker_raw": spk_raw.squeeze(0).cpu(),
                "emotion_global_raw": emo_feats["global_raw"].squeeze(0).cpu(),
                "emotion_frame_raw": (
                    emo_feats["frame_raw"].squeeze(0).cpu()
                    if emo_feats["frame_raw"] is not None else None
                ),
            }, out_path)

    print(f"Embeddings cached to {output_dir}")
