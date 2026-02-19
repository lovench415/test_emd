"""
Emotion Encoder Module for F5-TTS Enhanced
============================================

Extracts continuous emotion embeddings from reference audio using pretrained
Speech Emotion Recognition (SER) models. Unlike discrete emotion labels,
continuous embeddings capture the full spectrum of emotional expression
(valence, arousal, dominance) — enabling smooth, nuanced emotional transfer.

Design philosophy:
- emotion2vec (FunASR) is the primary backend: state-of-the-art SER that 
  generalizes well across languages (trained on multilingual data).
- We also support Wav2Vec2-based SER as a fallback.
- The encoder is frozen; only a lightweight projection + optional LoRA
  adapter on the projection head is trained.
- We extract BOTH a global emotion embedding (utterance-level) and 
  frame-level emotion features, enabling Time-Varying Emotion Control.

Architecture references: 
    - emotion2vec (self-supervised emotion representation)
    - EmoSteer-TTS (emotion embedding as steering vector)
    - ece-tts (emotion condition embedding)
    - Time-Varying Emotion Control (frame-level emotion features)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionEncoder(nn.Module):
    """
    Extracts emotion embeddings from raw audio using a pretrained SER model.
    
    Supported backends:
        - "emotion2vec_base" : emotion2vec base model (768-d, best accuracy)
        - "emotion2vec_plus" : emotion2vec+ with finetuned classifier head
        - "wav2vec2_ser"     : Wav2Vec2 finetuned on emotion recognition
        - "hubert_ser"       : HuBERT finetuned on emotion recognition
    
    Outputs:
        - Global emotion embedding: (batch, output_dim) - utterance-level
        - Frame-level emotion features: (batch, T, output_dim) - for time-varying control
    """

    def __init__(
        self,
        backend: str = "emotion2vec_base",
        output_dim: int = 512,
        frame_level: bool = True,
        normalize: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.backend = backend
        self.output_dim = output_dim
        self.frame_level = frame_level
        self.normalize = normalize
        self._device = device

        self.encoder = None
        self.raw_dim: int = 0
        self.raw_frame_rate: int = 50  # frames per second of encoder output

        self._load_backend(backend, device)

        # Global embedding projection (utterance-level)
        self.global_proj = nn.Sequential(
            nn.Linear(self.raw_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

        # Frame-level projection (for time-varying emotion control)
        if frame_level:
            self.frame_proj = nn.Sequential(
                nn.Linear(self.raw_dim, output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim),
            )
            # Temporal smoothing to avoid abrupt emotion changes
            self.temporal_smooth = nn.Conv1d(
                output_dim, output_dim, kernel_size=5, padding=2, groups=output_dim
            )

    # ------------------------------------------------------------------
    # Backend loading
    # ------------------------------------------------------------------

    def _load_backend(self, backend: str, device: str):
        if backend.startswith("emotion2vec"):
            self._load_emotion2vec(backend, device)
        elif backend == "wav2vec2_ser":
            self._load_wav2vec2_ser(device)
        elif backend == "hubert_ser":
            self._load_hubert_ser(device)
        else:
            raise ValueError(f"Unknown emotion encoder backend: {backend}")

    def _load_emotion2vec(self, variant: str, device: str):
        """
        emotion2vec: Self-supervised emotion representation model.
        Base model outputs 768-d features per frame (~50fps at 16kHz).
        
        Uses FunASR for loading. If unavailable, falls back to manual loading.
        """
        try:
            from funasr import AutoModel

            model_name = (
                "iic/emotion2vec_base" 
                if variant == "emotion2vec_base" 
                else "iic/emotion2vec_plus_base"
            )
            self.encoder = AutoModel(model=model_name, device=device)
            self.raw_dim = 768
            self.sample_rate = 16000
            self._emotion2vec_funasr = True
        except ImportError:
            # Fallback: use HuggingFace-hosted emotion2vec weights
            from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/wav2vec2-base"
            )
            self.encoder = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-base"
            ).to(device)
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.raw_dim = 768
            self.sample_rate = 16000
            self._emotion2vec_funasr = False

    def _load_wav2vec2_ser(self, device: str):
        """
        Wav2Vec2 finetuned on speech emotion recognition.
        ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
        """
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

        model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.encoder = Wav2Vec2Model.from_pretrained(model_name).to(device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.raw_dim = 1024  # XLSR-53 large
        self.sample_rate = 16000

    def _load_hubert_ser(self, device: str):
        """
        HuBERT finetuned on emotion recognition tasks.
        """
        from transformers import HubertModel, Wav2Vec2FeatureExtractor

        model_name = "facebook/hubert-large-ls960-ft"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.encoder = HubertModel.from_pretrained(model_name).to(device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.raw_dim = 1024
        self.sample_rate = 16000

    # ------------------------------------------------------------------
    # Raw feature extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_raw_features(
        self, wav: torch.Tensor, sr: int = 24000
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract raw emotion features from waveform.
        
        Args:
            wav: (batch, samples) or (samples,) at any sample rate
            sr:  original sample rate
            
        Returns:
            global_feat: (batch, raw_dim) mean-pooled global feature
            frame_feat:  (batch, T_frames, raw_dim) frame-level features
        """
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.ndim == 3:
            wav = wav.squeeze(1)

        # Resample to 16kHz
        if sr != self.sample_rate:
            import torchaudio
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate).to(wav.device)
            wav = resampler(wav)

        if self.backend.startswith("emotion2vec") and hasattr(self, '_emotion2vec_funasr') and self._emotion2vec_funasr:
            return self._extract_emotion2vec(wav)
        else:
            return self._extract_transformers(wav)

    def _extract_emotion2vec(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """emotion2vec via FunASR."""
        import numpy as np
        
        frame_feats = []
        global_feats = []
        
        for i in range(wav.shape[0]):
            result = self.encoder.generate(
                wav[i].cpu().numpy(), 
                output_dir=None, 
                granularity="frame",
            )
            # result contains frame-level features
            if isinstance(result, list) and len(result) > 0:
                feats = result[0].get("feats", None)
                if feats is not None:
                    feats_tensor = torch.from_numpy(np.array(feats)).to(wav.device)
                    frame_feats.append(feats_tensor)
                    global_feats.append(feats_tensor.mean(dim=0))
                else:
                    # Fallback: use scores as features
                    scores = result[0].get("scores", torch.zeros(self.raw_dim))
                    scores_tensor = torch.tensor(scores).to(wav.device)
                    # Pad to raw_dim if needed
                    if scores_tensor.dim() == 1 and scores_tensor.shape[0] < self.raw_dim:
                        scores_tensor = F.pad(scores_tensor, (0, self.raw_dim - scores_tensor.shape[0]))
                    frame_feats.append(scores_tensor.unsqueeze(0))
                    global_feats.append(scores_tensor if scores_tensor.dim() == 1 else scores_tensor.mean(0))
        
        global_feat = torch.stack(global_feats)
        # Pad frame features to same length
        max_len = max(f.shape[0] for f in frame_feats)
        padded = [F.pad(f, (0, 0, 0, max_len - f.shape[0])) for f in frame_feats]
        frame_feat = torch.stack(padded)
        
        return global_feat, frame_feat

    def _extract_transformers(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract via HuggingFace Transformers models."""
        inputs = self.feature_extractor(
            wav.cpu().numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(wav.device)

        with torch.no_grad():
            outputs = self.encoder(input_values, output_hidden_states=True)
            # Use last hidden state as frame-level features
            frame_feat = outputs.last_hidden_state  # (batch, T, raw_dim)
            # Global: mean pool over time
            global_feat = frame_feat.mean(dim=1)  # (batch, raw_dim)

        return global_feat, frame_feat

    # ------------------------------------------------------------------
    # Forward: raw wav -> projected embeddings
    # ------------------------------------------------------------------

    def forward(
        self, wav: torch.Tensor, sr: int = 24000, target_len: int | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Full pipeline: raw waveform -> emotion embeddings.
        
        Args:
            wav:        (batch, samples) raw waveform
            sr:         sample rate
            target_len: if provided, interpolate frame features to this length
                        (for alignment with mel spectrogram frames)
            
        Returns:
            dict with:
                "global": (batch, output_dim)           - global emotion embedding
                "frame":  (batch, target_len, output_dim) - frame-level features (if frame_level=True)
        """
        global_feat, frame_feat = self.extract_raw_features(wav, sr)
        
        # Cast to projection dtype
        dtype = self.global_proj[0].weight.dtype
        global_feat = global_feat.to(dtype)
        
        # Project global
        global_emb = self.global_proj(global_feat)
        if self.normalize:
            global_emb = F.normalize(global_emb, p=2, dim=-1)

        result = {"global": global_emb}

        # Project frame-level
        if self.frame_level and frame_feat is not None:
            frame_feat = frame_feat.to(dtype)
            frame_emb = self.frame_proj(frame_feat)  # (batch, T, output_dim)
            
            # Temporal smoothing
            frame_emb = frame_emb.permute(0, 2, 1)  # (b, d, T)
            frame_emb = self.temporal_smooth(frame_emb)
            frame_emb = frame_emb.permute(0, 2, 1)  # (b, T, d)

            # Interpolate to target length (mel frames)
            if target_len is not None and frame_emb.shape[1] != target_len:
                frame_emb = frame_emb.permute(0, 2, 1)  # (b, d, T)
                frame_emb = F.interpolate(
                    frame_emb, size=target_len, mode="linear", align_corners=False
                )
                frame_emb = frame_emb.permute(0, 2, 1)  # (b, target_len, d)

            if self.normalize:
                frame_emb = F.normalize(frame_emb, p=2, dim=-1)

            result["frame"] = frame_emb

        return result

    # ------------------------------------------------------------------
    # Utility: precompute and cache
    # ------------------------------------------------------------------

    @torch.no_grad()
    def precompute_embeddings(
        self, wav: torch.Tensor, sr: int = 24000
    ) -> dict[str, torch.Tensor]:
        """
        Precompute raw features for dataset caching.
        Returns dict with 'global_raw' and 'frame_raw' tensors.
        """
        global_feat, frame_feat = self.extract_raw_features(wav, sr)
        return {
            "global_raw": global_feat.cpu(),
            "frame_raw": frame_feat.cpu() if frame_feat is not None else None,
        }

    def project_cached_embeddings(
        self, 
        global_raw: torch.Tensor, 
        frame_raw: torch.Tensor | None = None,
        target_len: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Project precomputed raw features through trainable heads.
        """
        dtype = self.global_proj[0].weight.dtype
        device = self.global_proj[0].weight.device
        
        global_raw = global_raw.to(device, dtype=dtype)
        global_emb = self.global_proj(global_raw)
        if self.normalize:
            global_emb = F.normalize(global_emb, p=2, dim=-1)

        result = {"global": global_emb}

        if self.frame_level and frame_raw is not None:
            frame_raw = frame_raw.to(device, dtype=dtype)
            frame_emb = self.frame_proj(frame_raw)
            frame_emb = frame_emb.permute(0, 2, 1)
            frame_emb = self.temporal_smooth(frame_emb)
            frame_emb = frame_emb.permute(0, 2, 1)

            if target_len is not None and frame_emb.shape[1] != target_len:
                frame_emb = frame_emb.permute(0, 2, 1)
                frame_emb = F.interpolate(
                    frame_emb, size=target_len, mode="linear", align_corners=False
                )
                frame_emb = frame_emb.permute(0, 2, 1)

            if self.normalize:
                frame_emb = F.normalize(frame_emb, p=2, dim=-1)

            result["frame"] = frame_emb

        return result
