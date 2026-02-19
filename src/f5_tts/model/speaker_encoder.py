"""
Speaker Encoder Module for F5-TTS Enhanced
===========================================

Extracts language-agnostic speaker identity embeddings from reference audio.
Uses pretrained models (WavLM + ECAPA-TDNN via SpeechBrain, or resemblyzer)
to capture voice timbre, pitch range, formant structure, and other speaker
characteristics — independently of the spoken language.

Design choices:
- WavLM-based speaker verification (microsoft/wavlm-base-plus-sv) gives
  state-of-the-art speaker embeddings that generalize across languages.
- We freeze the encoder and only train lightweight projection heads,
  so no speaker-labeled data is needed for finetuning F5-TTS.
- Embeddings are L2-normalized to lie on a unit hypersphere, which
  stabilizes conditioning and prevents magnitude collapse.

Architecture reference: PEFT-TTS (adapter-based speaker conditioning),
                        TTS-CtrlNet (external encoder with frozen weights).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerEncoder(nn.Module):
    """
    Extracts a fixed-dim speaker embedding from raw waveform using a 
    pretrained speaker verification model.
    
    Supported backends:
        - "wavlm_sv"  : microsoft/wavlm-base-plus-sv (default, best quality)
        - "ecapa_tdnn" : speechbrain/spkrec-ecapa-voxceleb (fast, robust)
        - "resemblyzer" : GE2E-based (lightweight, CPU-friendly)
    
    The encoder is always frozen. Only the projection MLP is trainable.
    """

    def __init__(
        self,
        backend: str = "wavlm_sv",
        output_dim: int = 512,
        normalize: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.backend = backend
        self.output_dim = output_dim
        self.normalize = normalize
        self._device = device

        # Will be set by _load_backend
        self.encoder = None
        self.raw_dim: int = 0

        self._load_backend(backend, device)

        # Trainable projection: raw_dim -> output_dim
        self.proj = nn.Sequential(
            nn.Linear(self.raw_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    # ------------------------------------------------------------------
    # Backend loading
    # ------------------------------------------------------------------

    def _load_backend(self, backend: str, device: str):
        if backend == "wavlm_sv":
            self._load_wavlm_sv(device)
        elif backend == "ecapa_tdnn":
            self._load_ecapa_tdnn(device)
        elif backend == "resemblyzer":
            self._load_resemblyzer(device)
        else:
            raise ValueError(f"Unknown speaker encoder backend: {backend}")

    def _load_wavlm_sv(self, device: str):
        """
        microsoft/wavlm-base-plus-sv outputs a single 512-d speaker embedding.
        We use the Wav2Vec2ForXVector wrapper from HuggingFace.
        """
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        )
        self.encoder = WavLMForXVector.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        ).to(device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.raw_dim = 512  # WavLM-SV output dim
        self.sample_rate = 16000

    def _load_ecapa_tdnn(self, device: str):
        """
        SpeechBrain ECAPA-TDNN trained on VoxCeleb.
        Produces 192-d embeddings.
        """
        from speechbrain.inference.speaker import EncoderClassifier

        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        self.raw_dim = 192
        self.sample_rate = 16000

    def _load_resemblyzer(self, device: str):
        """
        Resemblyzer (GE2E encoder). Lightweight, ~256-d embeddings.
        """
        from resemblyzer import VoiceEncoder

        self.encoder = VoiceEncoder(device=device)
        self.raw_dim = 256
        self.sample_rate = 16000

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_raw_embedding(self, wav: torch.Tensor, sr: int = 24000) -> torch.Tensor:
        """
        Extract raw speaker embedding from waveform.
        
        Args:
            wav: (batch, samples) or (samples,) at any sample rate
            sr:  original sample rate of wav
            
        Returns:
            emb: (batch, raw_dim) raw speaker embedding
        """
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.ndim == 3:
            wav = wav.squeeze(1)  # (b, 1, t) -> (b, t)

        # Resample to 16kHz if needed
        if sr != self.sample_rate:
            import torchaudio
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate).to(wav.device)
            wav = resampler(wav)

        if self.backend == "wavlm_sv":
            return self._extract_wavlm(wav)
        elif self.backend == "ecapa_tdnn":
            return self._extract_ecapa(wav)
        elif self.backend == "resemblyzer":
            return self._extract_resemblyzer(wav)

    def _extract_wavlm(self, wav: torch.Tensor) -> torch.Tensor:
        """WavLM-SV extraction."""
        # Process through feature extractor
        inputs = self.feature_extractor(
            wav.cpu().numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(wav.device)
        
        with torch.no_grad():
            outputs = self.encoder(input_values)
            embeddings = outputs.embeddings  # (batch, 512)
        return embeddings

    def _extract_ecapa(self, wav: torch.Tensor) -> torch.Tensor:
        """ECAPA-TDNN extraction via SpeechBrain."""
        embeddings = self.encoder.encode_batch(wav)  # (batch, 1, 192)
        return embeddings.squeeze(1)

    def _extract_resemblyzer(self, wav: torch.Tensor) -> torch.Tensor:
        """Resemblyzer GE2E extraction."""
        import numpy as np
        embs = []
        for i in range(wav.shape[0]):
            emb = self.encoder.embed_utterance(wav[i].cpu().numpy())
            embs.append(torch.from_numpy(emb))
        return torch.stack(embs).to(wav.device)

    # ------------------------------------------------------------------
    # Forward: raw wav -> projected embedding
    # ------------------------------------------------------------------

    def forward(self, wav: torch.Tensor, sr: int = 24000) -> torch.Tensor:
        """
        Full pipeline: raw waveform -> projected speaker embedding.
        
        Args:
            wav: (batch, samples) raw waveform
            sr:  sample rate
            
        Returns:
            spk_emb: (batch, output_dim) projected, normalized speaker embedding
        """
        raw_emb = self.extract_raw_embedding(wav, sr)
        raw_emb = raw_emb.to(self.proj[0].weight.dtype)
        
        spk_emb = self.proj(raw_emb)

        if self.normalize:
            spk_emb = F.normalize(spk_emb, p=2, dim=-1)

        return spk_emb

    # ------------------------------------------------------------------
    # Utility: precompute and cache
    # ------------------------------------------------------------------

    @torch.no_grad()
    def precompute_embedding(self, wav: torch.Tensor, sr: int = 24000) -> torch.Tensor:
        """
        Precompute and return the raw embedding (before projection).
        Useful for dataset preprocessing to avoid recomputing every epoch.
        """
        return self.extract_raw_embedding(wav, sr).cpu()

    def project_cached_embedding(self, raw_emb: torch.Tensor) -> torch.Tensor:
        """
        Project a precomputed raw embedding through the trainable projection.
        """
        raw_emb = raw_emb.to(self.proj[0].weight.device, dtype=self.proj[0].weight.dtype)
        spk_emb = self.proj(raw_emb)
        if self.normalize:
            spk_emb = F.normalize(spk_emb, p=2, dim=-1)
        return spk_emb
