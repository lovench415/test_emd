"""
Speaker Encoder — extracts language-agnostic speaker identity embeddings.

Uses pretrained models (WavLM-SV / ECAPA-TDNN / Resemblyzer) to capture
voice timbre, pitch range, formant structure independently of spoken language.
The encoder is always frozen; only the projection MLP is trainable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from f5_tts.model.encoder_utils import BaseEncoder, SPEAKER_RAW_DIMS


class SpeakerEncoder(BaseEncoder):
    """
    Raw waveform → fixed-dim speaker embedding.

    Backends:
        "wavlm_sv"   — microsoft/wavlm-base-plus-sv  (512-d, best quality)
        "ecapa_tdnn"  — speechbrain/spkrec-ecapa-voxceleb (192-d, fast)
        "resemblyzer" — GE2E encoder (256-d, CPU-friendly)
    """

    def __init__(
        self,
        backend: str = "wavlm_sv",
        output_dim: int = 512,
        normalize: bool = True,
        device: str = "cpu",
    ):
        super().__init__(output_dim=output_dim, normalize=normalize)
        self.backend = backend
        self._load_backend(backend, device)
        self.proj = self.make_projection(self.raw_dim, output_dim)

    # ── Backend loading ───────────────────────────────────────────────

    def _load_backend(self, backend: str, device: str):
        if backend == "wavlm_sv":
            self._load_wavlm(device)
        elif backend == "ecapa_tdnn":
            self._load_ecapa(device)
        elif backend == "resemblyzer":
            self._load_resemblyzer(device)
        else:
            raise ValueError(
                f"Unknown speaker backend: {backend!r}. "
                f"Choose from: {list(SPEAKER_RAW_DIMS)}"
            )

    def _load_wavlm(self, device: str):
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        model_id = "microsoft/wavlm-base-plus-sv"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self.encoder = WavLMForXVector.from_pretrained(model_id).to(device).eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.raw_dim = SPEAKER_RAW_DIMS["wavlm_sv"]

    def _load_ecapa(self, device: str):
        from speechbrain.inference.speaker import EncoderClassifier

        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        self.raw_dim = SPEAKER_RAW_DIMS["ecapa_tdnn"]

    def _load_resemblyzer(self, device: str):
        from resemblyzer import VoiceEncoder

        self.encoder = VoiceEncoder(device=device)
        self.raw_dim = SPEAKER_RAW_DIMS["resemblyzer"]

    # ── Raw extraction ────────────────────────────────────────────────

    @torch.no_grad()
    def extract_raw(self, wav: torch.Tensor, sr: int = 24000) -> torch.Tensor:
        """Extract raw speaker embedding: (B, raw_dim)."""
        wav = self._prepare_wav(wav, sr)

        if self.backend == "wavlm_sv":
            inputs = self.feature_extractor(
                self._wav_to_list(wav),
                sampling_rate=self.sample_rate, return_tensors="pt", padding=True,
            )
            outputs = self.encoder(inputs.input_values.to(wav.device))
            return outputs.embeddings

        if self.backend == "ecapa_tdnn":
            return self.encoder.encode_batch(wav).squeeze(1)

        # resemblyzer — per-utterance, returns numpy
        import numpy as np
        embs = [
            torch.from_numpy(self.encoder.embed_utterance(wav[i].cpu().numpy()))
            for i in range(wav.shape[0])
        ]
        return torch.stack(embs).to(wav.device)

    # ── Public interface ──────────────────────────────────────────────

    def forward(self, wav: torch.Tensor, sr: int = 24000, **_) -> torch.Tensor:
        """Raw waveform → projected, normalized speaker embedding (B, output_dim)."""
        raw = self.extract_raw(wav, sr)
        return self._project_and_normalize(raw, self.proj)

    @torch.no_grad()
    def precompute(self, wav: torch.Tensor, sr: int = 24000) -> torch.Tensor:
        """Precompute raw embedding for dataset caching."""
        return self.extract_raw(wav, sr).cpu()

    def project_cached(self, raw_emb: torch.Tensor, **_) -> torch.Tensor:
        """Project a precomputed raw embedding through the trainable head."""
        return self._project_and_normalize(raw_emb, self.proj)
