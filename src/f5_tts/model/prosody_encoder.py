"""
Prosody Encoder — extracts F0 contour, energy, and voicing from audio.

Provides frame-level prosodic features for fine-grained intonation control.
Unlike emotion2vec (trained for emotion classification), this directly
captures pitch melody, rhythm, and stress patterns.

Backends:
    "dio"     — pyworld DIO (fast, CPU, good quality)
    "harvest" — pyworld Harvest (slower, most accurate F0)
    "crepe"   — torchcrepe (GPU, neural F0, best for noisy audio)
    "rmvpe"   — RMVPE (GPU, robust neural F0, best overall quality)

This module is extraction-only — no trainable parameters.
All projection and smoothing happens in ConditioningAggregator
(prosody_raw_proj + prosody_direct_proj + temporal_smooth).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np


PROSODY_RAW_DIM = 7  # log_f0, voicing, log_energy, delta_f0, delta_energy, log_f0_absolute, local_rate


class ProsodyEncoder:
    """
    Raw waveform → frame-level prosody features.

    Output: (B, T_frames, 7) — log_f0, voicing, log_energy, Δf0, Δenergy, log_f0_absolute.
    First 5 features are normalized to ~zero mean, unit variance.
    Channel 5 (log_f0_absolute) preserves absolute pitch for voice cloning.

    No trainable parameters — this is a deterministic feature extractor.
    Projection to model_dim happens in ConditioningAggregator.
    """

    def __init__(
        self,
        backend: str = "dio",
        hop_length: int = 256,
        sample_rate: int = 24000,
        f0_floor: float = 50.0,
        f0_ceil: float = 800.0,
        device: str = "cpu",
        output_dim: int | None = None,  # accepted for API compat, not used (no trainable params)
        rmvpe_model_path: str | None = None,  # path to rmvpe.pt (auto-downloads if None)
    ):
        self.backend = backend
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.raw_dim = PROSODY_RAW_DIM
        self.device = device
        self.rmvpe_model_path = rmvpe_model_path
        self._rmvpe_model = None
        self._validate_backend(backend)

    def _validate_backend(self, backend: str):
        if backend in ("dio", "harvest"):
            try:
                import pyworld
            except ImportError:
                raise ImportError(f"pyworld required for '{backend}' backend: pip install pyworld")
        elif backend == "crepe":
            try:
                import torchcrepe
            except ImportError:
                raise ImportError("torchcrepe required for 'crepe' backend: pip install torchcrepe")
        elif backend == "rmvpe":
            pass  # RMVPE model loaded lazily on first use
        else:
            raise ValueError(f"Unknown prosody backend: {backend!r}. Choose from: dio, harvest, crepe, rmvpe")

    # Stub methods so existing code that calls .to()/.eval() doesn't crash
    def to(self, *args, **kwargs):
        # Update self.device if a device argument is provided
        if args:
            dev = args[0]
            if isinstance(dev, (str, torch.device)):
                self.device = str(dev)
        if "device" in kwargs:
            self.device = str(kwargs["device"])
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter([])

    # ── Raw feature extraction ────────────────────────────────────────

    @torch.no_grad()
    def extract_raw(
        self, wav: torch.Tensor, sr: int = 24000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract raw prosody features.

        Args:
            wav: (B, T) or (1, T) waveform
            sr: sample rate

        Returns:
            features: (B, T_frames, 6) — log_f0, voicing, log_energy, Δf0, Δenergy, log_f0_absolute
            mask:     (B, T_frames) bool — True where features are valid
        """
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.ndim == 3:
            wav = wav.squeeze(1)

        device = wav.device
        batch_features = []
        batch_masks = []

        for i in range(wav.shape[0]):
            audio_np = wav[i].cpu().float().numpy()

            if self.backend in ("dio", "harvest"):
                f0, energy = self._extract_pyworld(audio_np, sr)
            elif self.backend == "crepe":
                f0, energy = self._extract_crepe(wav[i:i+1], sr, device)
            elif self.backend == "rmvpe":
                f0, energy = self._extract_rmvpe(wav[i:i+1], sr, device)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")

            features, mask = self._build_features(f0, energy)
            batch_features.append(features)
            batch_masks.append(mask)

        # Pad to same length
        max_len = max(f.shape[0] for f in batch_features)
        padded_features = []
        padded_masks = []
        for f, m in zip(batch_features, batch_masks):
            pad_t = max_len - f.shape[0]
            padded_features.append(F.pad(f, (0, 0, 0, pad_t)))
            padded_masks.append(F.pad(m, (0, pad_t), value=False))

        features = torch.stack(padded_features).to(device)  # (B, T, 5)
        masks = torch.stack(padded_masks).to(device)         # (B, T)
        return features, masks

    def _extract_pyworld(self, audio_np: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """Extract F0 and energy using pyworld."""
        import pyworld as pw

        audio_f64 = audio_np.astype(np.float64)
        frame_period = self.hop_length / sr * 1000  # ms

        if self.backend == "dio":
            f0, t = pw.dio(audio_f64, sr,
                           f0_floor=self.f0_floor, f0_ceil=self.f0_ceil,
                           frame_period=frame_period)
            f0 = pw.stonemask(audio_f64, f0, t, sr)
        else:
            f0, t = pw.harvest(audio_f64, sr,
                               f0_floor=self.f0_floor, f0_ceil=self.f0_ceil,
                               frame_period=frame_period)

        # Energy: RMS per frame
        n_frames = len(f0)
        energy = np.zeros(n_frames, dtype=np.float64)
        for j in range(n_frames):
            start = j * self.hop_length
            end = min(start + self.hop_length, len(audio_np))
            if start < len(audio_np):
                frame = audio_np[start:end]
                energy[j] = np.sqrt(np.mean(frame ** 2) + 1e-8)

        return f0, energy

    def _extract_crepe(self, wav_tensor: torch.Tensor, sr: int, device) -> tuple[np.ndarray, np.ndarray]:
        """Extract F0 using torchcrepe, energy from waveform."""
        import torchcrepe

        hop = self.hop_length
        if sr != 16000:
            import torchaudio
            wav_16k = torchaudio.transforms.Resample(sr, 16000).to(device)(wav_tensor)
            hop_16k = int(hop * 16000 / sr)
        else:
            wav_16k = wav_tensor
            hop_16k = hop

        f0 = torchcrepe.predict(
            wav_16k.to(device), 16000,
            hop_length=hop_16k,
            fmin=self.f0_floor, fmax=self.f0_ceil,
            model='tiny', device=device,
            batch_size=1,
        )
        f0 = f0.squeeze().cpu().numpy()

        # Energy from original audio
        audio_np = wav_tensor.squeeze().cpu().float().numpy()
        n_frames = len(f0)
        energy = np.zeros(n_frames, dtype=np.float64)
        for j in range(n_frames):
            start = j * hop
            end = min(start + hop, len(audio_np))
            if start < len(audio_np):
                frame = audio_np[start:end]
                energy[j] = np.sqrt(np.mean(frame ** 2) + 1e-8)

        return f0, energy

    def _load_rmvpe(self, device):
        """Lazily load RMVPE model."""
        if self._rmvpe_model is not None:
            return self._rmvpe_model

        from f5_tts.model.RMVPE import RMVPE0Predictor as RMVPE

        model_path = self.rmvpe_model_path
        if model_path is None:
            from huggingface_hub import hf_hub_download

            REPO_ID = "lj1995/VoiceConversionWebUI"
            FILENAME = "rmvpe.pt"

            model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
            if model_path is None:
                raise FileNotFoundError(
                    "RMVPE model not found. Download rmvpe.pt from "
                    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt "
                    "and place in current directory, models/, or ~/.cache/rmvpe/"
                )

        self._rmvpe_model = RMVPE(model_path, device=device)
        return self._rmvpe_model

    def _extract_rmvpe(self, wav_tensor: torch.Tensor, sr: int, device) -> tuple[np.ndarray, np.ndarray]:
        """Extract F0 using RMVPE (robust neural pitch), energy from waveform."""
        import torchaudio

        hop = self.hop_length

        # RMVPE expects 16kHz mono
        wav_16k = wav_tensor.to(device)
        if sr != 16000:
            wav_16k = torchaudio.transforms.Resample(sr, 16000).to(device)(wav_16k)

        model = self._load_rmvpe(device)

        # RMVPE returns F0 at its own hop (160 samples at 16kHz = 10ms)
        # We need to resample to our hop_length
        wav_16k_np = wav_16k.squeeze().cpu().float().numpy()
        f0_rmvpe = model.infer_from_audio(wav_16k_np, thred=0.03)
        # f0_rmvpe: numpy array, (T_rmvpe,)

        # Resample F0 to match our hop_length
        audio_np = wav_tensor.squeeze().cpu().float().numpy()
        n_frames_target = len(audio_np) // hop + 1
        if len(f0_rmvpe) != n_frames_target:
            # Interpolate to target frame count
            f0_interp = np.interp(
                np.linspace(0, 1, n_frames_target),
                np.linspace(0, 1, len(f0_rmvpe)),
                f0_rmvpe,
            )
            f0 = f0_interp
        else:
            f0 = f0_rmvpe

        # Energy: RMS per frame (same as pyworld path)
        n_frames = len(f0)
        energy = np.zeros(n_frames, dtype=np.float64)
        for j in range(n_frames):
            start = j * hop
            end = min(start + hop, len(audio_np))
            if start < len(audio_np):
                frame = audio_np[start:end]
                energy[j] = np.sqrt(np.mean(frame ** 2) + 1e-8)

        return f0.astype(np.float64), energy

    def _build_features(self, f0: np.ndarray, energy: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build 7-dim feature vector from raw F0 and energy.

        Features:
            0: log_f0           — log-scaled F0, z-normalized per utterance (relative contour)
            1: voicing          — 1.0 voiced, 0.0 unvoiced
            2: log_energy       — log-scaled RMS energy, z-normalized
            3: delta_f0         — F0 velocity (pitch change rate)
            4: delta_energy     — energy velocity
            5: log_f0_absolute  — log-scaled F0, NOT normalized (absolute pitch level)
            6: local_rate       — local speaking rate (voicing density in ~200ms window)

        Channel 6 captures rhythm and pacing: high = fast speech, low = pause/slow.
        Computed as smoothed voicing density over a sliding window.
        Together with channels 1-2, the model can learn pause positions,
        acceleration/deceleration patterns, and phrase-level rhythm.
        """
        n = len(f0)
        voiced = f0 > 0
        log_f0_abs = np.zeros(n, dtype=np.float32)
        log_f0_abs[voiced] = np.log(f0[voiced] + 1e-8)

        # z-normalized version (relative contour)
        log_f0 = log_f0_abs.copy()
        if voiced.sum() > 1:
            mean_f0 = log_f0[voiced].mean()
            std_f0 = log_f0[voiced].std() + 1e-8
            log_f0[voiced] = (log_f0[voiced] - mean_f0) / std_f0

        voicing = voiced.astype(np.float32)
        log_energy = np.log(energy + 1e-8).astype(np.float32)

        # Normalize energy
        if n > 1:
            mean_e = log_energy.mean()
            std_e = log_energy.std() + 1e-8
            log_energy = (log_energy - mean_e) / std_e

        # Delta features (first-order difference)
        delta_f0 = np.zeros(n, dtype=np.float32)
        delta_energy = np.zeros(n, dtype=np.float32)
        if n > 1:
            delta_f0[1:] = np.diff(log_f0)
            delta_energy[1:] = np.diff(log_energy)

        # Local speaking rate: voicing density in sliding window.
        # Window ~200ms at 24kHz/256hop ≈ 19 frames. Use 15 for efficiency.
        # High = dense speech (fast), low = pause/silence (slow).
        # Z-normalized so model sees relative speed changes.
        win = min(15, max(3, n // 4))
        if n >= win:
            kernel = np.ones(win, dtype=np.float32) / win
            local_rate = np.convolve(voicing, kernel, mode='same').astype(np.float32)
        else:
            local_rate = np.full(n, voicing.mean(), dtype=np.float32)
        # Z-normalize
        if n > 1:
            lr_mean = local_rate.mean()
            lr_std = local_rate.std() + 1e-8
            local_rate = (local_rate - lr_mean) / lr_std

        features = np.stack([log_f0, voicing, log_energy, delta_f0, delta_energy, log_f0_abs, local_rate], axis=-1)
        mask = np.ones(n, dtype=bool)

        return torch.from_numpy(features), torch.from_numpy(mask)

    def precompute(self, wav: torch.Tensor, sr: int = 24000) -> dict[str, torch.Tensor]:
        """Precompute raw features for dataset caching."""
        features, mask = self.extract_raw(wav, sr)
        return {"prosody_raw": features.cpu(), "prosody_mask": mask.cpu()}
