"""
Emotion Encoder — extracts continuous emotion embeddings from reference audio.

Uses pretrained SER models (emotion2vec / Wav2Vec2 / HuBERT) to capture the
full spectrum of emotional expression. Outputs both global (utterance-level)
and frame-level features for time-varying emotion control.

The encoder is frozen; only projection heads + temporal smoothing are trained.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from f5_tts.model.encoder_utils import (
    BaseEncoder,
    EMOTION_RAW_DIMS,
    interpolate_temporal,
)


class EmotionEncoder(BaseEncoder):
    """
    Raw waveform → emotion embeddings (global + frame-level).

    Backends:
        "emotion2vec_base" — emotion2vec base (768-d, best accuracy)
        "emotion2vec_plus" — emotion2vec+ finetuned
        "wav2vec2_ser"     — Wav2Vec2 XLSR-53 finetuned on SER (1024-d)
        "hubert_ser"       — HuBERT large finetuned (1024-d)
    """

    def __init__(
        self,
        backend: str = "emotion2vec_base",
        output_dim: int = 512,
        frame_level: bool = True,
        normalize: bool = True,
        device: str = "cpu",
    ):
        super().__init__(output_dim=output_dim, normalize=normalize)
        self.backend = backend
        self.frame_level = frame_level

        self._load_backend(backend, device)

        self.global_proj = self.make_projection(self.raw_dim, output_dim)

        if frame_level:
            self.frame_proj = self.make_projection(self.raw_dim, output_dim)
            self.temporal_smooth = nn.Conv1d(
                output_dim, output_dim, kernel_size=5, padding=2, groups=output_dim,
            )

    # ── Backend loading ───────────────────────────────────────────────

    def _load_backend(self, backend: str, device: str):
        if backend.startswith("emotion2vec"):
            self._load_emotion2vec(backend, device)
        elif backend == "wav2vec2_ser":
            self._load_hf("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", device, 1024)
        elif backend == "hubert_ser":
            self._load_hf("facebook/hubert-large-ls960-ft", device, 1024, model_cls="hubert")
        else:
            raise ValueError(
                f"Unknown emotion backend: {backend!r}. "
                f"Choose from: {list(EMOTION_RAW_DIMS)}"
            )

    def _load_emotion2vec(self, variant: str, device: str):
        try:
            from funasr import AutoModel

            model_name = (
                "iic/emotion2vec_base"
                if variant == "emotion2vec_base"
                else "iic/emotion2vec_plus_base"
            )
            self.encoder = AutoModel(model=model_name, device=device)
            self.raw_dim = 768
            self._use_funasr = True
        except ImportError:
            self._load_hf("facebook/wav2vec2-base", device, 768)
            self._use_funasr = False

    def _load_hf(self, model_id: str, device: str, raw_dim: int, model_cls: str = "wav2vec2"):
        """Load any HuggingFace Wav2Vec2-style model."""
        from transformers import Wav2Vec2FeatureExtractor

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

        if model_cls == "hubert":
            from transformers import HubertModel
            self.encoder = HubertModel.from_pretrained(model_id).to(device).eval()
        else:
            from transformers import Wav2Vec2Model
            self.encoder = Wav2Vec2Model.from_pretrained(model_id).to(device).eval()

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.raw_dim = raw_dim
        self._use_funasr = False

    # ── Raw extraction ────────────────────────────────────────────────

    @torch.no_grad()
    def extract_raw(
        self, wav: torch.Tensor, sr: int = 24000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (global_feat, frame_feat):
            global_feat: (B, raw_dim)
            frame_feat:  (B, T_frames, raw_dim)
        """
        wav = self._prepare_wav(wav, sr)

        if getattr(self, "_use_funasr", False):
            return self._extract_funasr(wav)
        return self._extract_hf(wav)

    def _extract_funasr(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        import numpy as np

        frame_feats, global_feats = [], []
        for i in range(wav.shape[0]):
            result = self.encoder.generate(wav[i].cpu().numpy(), output_dir=None, granularity="frame")
            if isinstance(result, list) and result:
                feats = result[0].get("feats")
                if feats is not None:
                    ft = torch.from_numpy(np.array(feats)).to(wav.device)
                    frame_feats.append(ft)
                    global_feats.append(ft.mean(dim=0))
                    continue

            # Fallback: zero features
            frame_feats.append(torch.zeros(1, self.raw_dim, device=wav.device))
            global_feats.append(torch.zeros(self.raw_dim, device=wav.device))

        global_feat = torch.stack(global_feats)
        max_len = max(f.shape[0] for f in frame_feats)
        padded = [F.pad(f, (0, 0, 0, max_len - f.shape[0])) for f in frame_feats]
        return global_feat, torch.stack(padded)

    def _extract_hf(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.feature_extractor(
            self._wav_to_list(wav),
            sampling_rate=self.sample_rate, return_tensors="pt", padding=True,
        )
        outputs = self.encoder(inputs.input_values.to(wav.device), output_hidden_states=True)
        frame_feat = outputs.last_hidden_state       # (B, T, raw_dim)
        global_feat = frame_feat.mean(dim=1)          # (B, raw_dim)
        return global_feat, frame_feat

    # ── Temporal smoothing (shared logic) ─────────────────────────────

    def _smooth_and_interpolate(
        self, frame_emb: torch.Tensor, target_len: int | None,
    ) -> torch.Tensor:
        """Apply temporal smoothing conv and optional interpolation."""
        frame_emb = frame_emb.permute(0, 2, 1)                    # (B, D, T)
        frame_emb = self.temporal_smooth(frame_emb)
        frame_emb = frame_emb.permute(0, 2, 1)                    # (B, T, D)
        if target_len is not None:
            frame_emb = interpolate_temporal(frame_emb, target_len)
        if self.normalize:
            frame_emb = F.normalize(frame_emb, p=2, dim=-1)
        return frame_emb

    # ── Public interface ──────────────────────────────────────────────

    def forward(
        self, wav: torch.Tensor, sr: int = 24000, target_len: int | None = None, **_,
    ) -> dict[str, torch.Tensor]:
        """
        Full pipeline → {"global": (B, D), "frame": (B, T', D)}.
        """
        global_feat, frame_feat = self.extract_raw(wav, sr)

        result = {"global": self._project_and_normalize(global_feat, self.global_proj)}

        if self.frame_level and frame_feat is not None:
            dtype = self.frame_proj[0].weight.dtype
            frame_emb = self.frame_proj(frame_feat.to(dtype))
            result["frame"] = self._smooth_and_interpolate(frame_emb, target_len)

        return result

    @torch.no_grad()
    def precompute(self, wav: torch.Tensor, sr: int = 24000) -> dict[str, torch.Tensor]:
        """Precompute raw features for dataset caching."""
        g, f = self.extract_raw(wav, sr)
        return {"global_raw": g.cpu(), "frame_raw": f.cpu() if f is not None else None}

    def project_cached(
        self,
        global_raw: torch.Tensor,
        frame_raw: torch.Tensor | None = None,
        target_len: int | None = None,
        **_,
    ) -> dict[str, torch.Tensor]:
        """Project precomputed raw features through trainable heads."""
        result = {"global": self._project_and_normalize(global_raw, self.global_proj)}

        if self.frame_level and frame_raw is not None:
            dtype = self.frame_proj[0].weight.dtype
            device = self.frame_proj[0].weight.device
            frame_emb = self.frame_proj(frame_raw.to(device, dtype=dtype))
            result["frame"] = self._smooth_and_interpolate(frame_emb, target_len)

        return result
