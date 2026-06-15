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
        # Fixed global normalization stats for log_f0 (ch 0) and log_energy
        # (ch 2). Using FIXED stats — rather than per-utterance mean/std — makes
        # the normalization identical at train time (full utterance) and
        # inference (reference clip), so the same physical contour maps to the
        # same feature values in both. Per-utterance z-norm made them disagree
        # (the classic prosody train/inference mismatch). Defaults are reasonable
        # for speech; override with corpus-measured statistics for best results.
        f0_norm_mean: float = 5.0,    # ~log(150 Hz)
        f0_norm_std: float = 0.5,
        energy_norm_mean: float = -4.0,
        energy_norm_std: float = 2.0,
        norm_mode: str = "per_utterance",  # DEFAULT — matches the original
                                            # working model (per-utterance z-norm
                                            # of log_f0/log_energy against the
                                            # reference itself). "fixed" uses
                                            # global stats and is ONLY correct if
                                            # the model was trained that way; using
                                            # it on a per-utterance-trained model
                                            # corrupts prosody → poor cloning +
                                            # reference leakage.
        center_log_f0_abs: bool = False,   # recenter channel 5 (abs log-F0) to ~0
                                           # using FIXED constants, fixing the scale
                                           # imbalance while preserving absolute
                                           # pitch. Default off = matches __58__.
        f0_abs_center: float = 5.3,        # ~log(200 Hz); subtracted from abs log-F0
        f0_abs_scale: float = 0.5,         # divides abs log-F0 to ~unit scale
    ):
        self.center_log_f0_abs = center_log_f0_abs
        self.f0_abs_center = f0_abs_center
        self.f0_abs_scale = max(f0_abs_scale, 1e-6)
        self.backend = backend
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.raw_dim = PROSODY_RAW_DIM
        self.device = device
        self.rmvpe_model_path = rmvpe_model_path
        self.f0_norm_mean = f0_norm_mean
        self.f0_norm_std = max(f0_norm_std, 1e-6)
        self.energy_norm_mean = energy_norm_mean
        self.energy_norm_std = max(energy_norm_std, 1e-6)
        # Floor applied to RMS energy before log, used IDENTICALLY in stat
        # estimation (raw_logf0_energy) and feature building (_build_features).
        # The old 1e-8 floor mapped silent frames to log≈-18, which dominated
        # the corpus energy mean/std (std blew up ~30×), crushing real speech
        # energy dynamics after normalization. 1e-3 keeps silence bounded while
        # preserving speech-region dynamics.
        self.energy_floor = 1e-3
        self.norm_mode = norm_mode
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
        lengths: torch.Tensor | list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract raw prosody features.

        Args:
            wav: (B, T) or (1, T) waveform
            sr: sample rate

        Returns:
            features: (B, T_frames, 7) — log_f0(norm), voicing, log_energy(norm),
                      Δf0, Δenergy, log_f0_absolute, local_rate
            mask:     (B, T_frames) bool — True where features are valid
        """
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.ndim == 3:
            wav = wav.squeeze(1)

        # GPU backends (rmvpe/crepe) must run on the encoder's configured device
        # (self.device), NOT the input tensor's device. In prepare_data the audio
        # is loaded on CPU (torchaudio.load) and passed here as a CPU tensor — using
        # wav.device would force RMVPE/crepe onto CPU (very slow). Audio is moved
        # to the target device inside the backend extractors as needed.
        device = self.device if self.backend in ("rmvpe", "crepe") else wav.device

        # Move audio to target device once (GPU backends need it there;
        # CPU backends keep it on CPU since device==wav.device for them).
        wav = wav.to(device)

        # Capture lengths + the ORIGINAL input sr BEFORE the resample below, so
        # we can rescale lengths into the post-resample rate. (Lengths arrive in
        # input-sr samples.)
        orig_sr_for_len = sr
        sample_lengths = None
        if lengths is not None:
            ll = lengths.tolist() if hasattr(lengths, "tolist") else list(lengths)
            sample_lengths = [int(x) for x in ll]

        # Resample to the encoder's target sample_rate (matches mel: 24kHz).
        # Frame timing uses hop_length at self.sample_rate; if input sr differs,
        # prosody frames would not align with mel frames (mel is at sample_rate).
        # Resample once here so all downstream frame counts match the mel.
        if sr != self.sample_rate:
            import torchaudio
            if not hasattr(self, "_resamplers"):
                self._resamplers = {}
            key = (sr, self.sample_rate)
            if key not in self._resamplers:
                self._resamplers[key] = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav = self._resamplers[key].to(device)(wav)
            sr = self.sample_rate

        batch_features = []
        batch_masks = []

        # Rescale lengths from the original input rate to the current rate.
        cur_lengths = None
        if sample_lengths is not None and len(sample_lengths) == wav.shape[0]:
            ratio = self.sample_rate / float(orig_sr_for_len)
            T_cur = wav.shape[-1]
            cur_lengths = [max(1, min(int(round(l * ratio)), T_cur)) for l in sample_lengths]

        # ── #2: batched RMVPE path ──
        # Run the RMVPE network once over the whole padded batch instead of
        # one forward per sample. F0 comes back as a list (one contour per row,
        # trimmed to valid frames); energy is still computed per sample on CPU
        # (cheap RMS). pyworld/crepe stay per-sample below.
        rmvpe_f0_batch = None
        if self.backend == "rmvpe" and wav.shape[0] > 1:
            try:
                import torchaudio
                model = self._load_rmvpe(device)
                wav_16k = wav.to(device).float()
                cur_sr = sr
                if cur_sr != 16000:
                    if not hasattr(self, "_resamplers"):
                        self._resamplers = {}
                    key = (cur_sr, 16000)
                    if key not in self._resamplers:
                        self._resamplers[key] = torchaudio.transforms.Resample(cur_sr, 16000).to(device)
                    wav_16k = self._resamplers[key](wav_16k)
                # Real per-sample lengths at 16k (so padding is trimmed before decode).
                if cur_lengths is not None:
                    r16 = 16000 / float(sr)
                    T16 = wav_16k.shape[-1]
                    lengths_16k = [max(1, min(int(round(l * r16)), T16)) for l in cur_lengths]
                else:
                    lengths_16k = [wav_16k.shape[-1]] * wav_16k.shape[0]
                rmvpe_f0_batch = model.infer_from_audio_tensor_batch(
                    wav_16k, lengths=lengths_16k, thred=0.03,
                )
            except Exception as e:
                rmvpe_f0_batch = None  # fall back to per-sample below
                # Warn once: the fallback keeps preprocessing alive, but if the
                # batch path fails on EVERY batch (e.g. a bug in batch infer, OOM,
                # or a version mismatch) the run silently drops to the much slower
                # per-sample path. Surface it so that systematic failure is noticed
                # instead of just being slow for no apparent reason.
                if not getattr(self, "_rmvpe_batch_warned", False):
                    self._rmvpe_batch_warned = True
                    import warnings
                    warnings.warn(
                        f"RMVPE batched F0 extraction failed ({type(e).__name__}: {e}); "
                        f"falling back to per-sample extraction (slower). This warning "
                        f"is shown once; if it recurs every batch, batching is disabled "
                        f"for the whole run.",
                        RuntimeWarning,
                    )

        for i in range(wav.shape[0]):
            # Trim padding for this sample so feature/frame counts match the
            # single-sample path (energy + frame count derive from audio length).
            if cur_lengths is not None:
                audio_np = wav[i, :cur_lengths[i]].cpu().float().numpy()
            else:
                audio_np = wav[i].cpu().float().numpy()

            if self.backend in ("dio", "harvest"):
                f0, energy = self._extract_pyworld(audio_np, sr)
            elif self.backend == "crepe":
                wav_i = wav[i:i+1, :cur_lengths[i]] if cur_lengths is not None else wav[i:i+1]
                f0, energy = self._extract_crepe(wav_i, sr, device)
            elif self.backend == "rmvpe":
                if rmvpe_f0_batch is not None:
                    f0, energy = self._rmvpe_f0_to_features(rmvpe_f0_batch[i], audio_np)
                else:
                    wav_i = wav[i:i+1, :cur_lengths[i]] if cur_lengths is not None else wav[i:i+1]
                    f0, energy = self._extract_rmvpe(wav_i, sr, device)
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

        features = torch.stack(padded_features).to(device)  # (B, T, 7)
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

        # RMVPE expects 16kHz mono
        wav_16k = wav_tensor.to(device)
        if sr != 16000:
            wav_16k = torchaudio.transforms.Resample(sr, 16000).to(device)(wav_16k)

        model = self._load_rmvpe(device)

        # RMVPE returns F0 at its own hop (160 samples at 16kHz = 10ms).
        # Use the GPU-native path: pass the 16kHz tensor directly, avoiding
        # the GPU→numpy→GPU round-trip (a host sync per call).
        wav_16k_1d = wav_16k.squeeze()
        if wav_16k_1d.dim() == 0:
            wav_16k_1d = wav_16k_1d.unsqueeze(0)
        f0_rmvpe = model.infer_from_audio_tensor(wav_16k_1d, thred=0.03)
        # f0_rmvpe: numpy array, (T_rmvpe,)

        audio_np = wav_tensor.squeeze().cpu().float().numpy()
        return self._rmvpe_f0_to_features(f0_rmvpe, audio_np)

    def _rmvpe_f0_to_features(self, f0_rmvpe: np.ndarray, audio_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Resample an RMVPE F0 contour to our hop grid and compute frame energy.

        Shared by the single-sample (_extract_rmvpe) and batched
        (infer_from_audio_tensor_batch) paths so both produce identical features.
        """
        hop = self.hop_length
        # Resample F0 to match our hop_length
        n_frames_target = len(audio_np) // hop + 1
        if len(f0_rmvpe) != n_frames_target:
            # Interpolate to target frame count
            f0 = np.interp(
                np.linspace(0, 1, n_frames_target),
                np.linspace(0, 1, len(f0_rmvpe)),
                f0_rmvpe,
            )
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

    def raw_logf0_energy(self, audio_np: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (log_f0_voiced, log_energy) UN-normalized, for corpus-stat
        estimation. log_f0_voiced contains only voiced frames (unvoiced excluded,
        matching how channel 0 is normalized). Used by prepare_data to measure the
        fixed normalization stats; not used in the normal feature path."""
        if self.backend in ("dio", "harvest"):
            f0, energy = self._extract_pyworld(audio_np, sr)
        elif self.backend == "crepe":
            wav_i = torch.from_numpy(audio_np).unsqueeze(0)
            f0, energy = self._extract_crepe(wav_i, sr, self.device)
        elif self.backend == "rmvpe":
            wav_i = torch.from_numpy(audio_np).unsqueeze(0)
            f0, energy = self._extract_rmvpe(wav_i, sr, self.device)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        f0 = np.asarray(f0)
        voiced = f0 > 0
        log_f0_voiced = np.log(f0[voiced] + 1e-8).astype(np.float64)
        log_energy = np.log(np.maximum(np.asarray(energy), self.energy_floor)).astype(np.float64)
        return log_f0_voiced, log_energy

    @staticmethod
    def update_log_stats(acc: dict, log_f0_voiced: np.ndarray, log_energy: np.ndarray) -> dict:
        """Streaming accumulation of count/sum/sumsq for log_f0 and log_energy.
        `acc` is a dict initialised to zeros; call per sample, then finalize_log_stats."""
        acc["f0_n"] += log_f0_voiced.size
        acc["f0_sum"] += float(log_f0_voiced.sum())
        acc["f0_sumsq"] += float(np.square(log_f0_voiced).sum())
        acc["e_n"] += log_energy.size
        acc["e_sum"] += float(log_energy.sum())
        acc["e_sumsq"] += float(np.square(log_energy).sum())
        return acc

    @staticmethod
    def finalize_log_stats(acc: dict) -> dict:
        """Turn accumulated sums into mean/std for f0 and energy."""
        def _ms(n, s, ss):
            if n < 2:
                return None, None
            mean = s / n
            var = max(ss / n - mean * mean, 0.0)
            return mean, var ** 0.5
        f0_mean, f0_std = _ms(acc["f0_n"], acc["f0_sum"], acc["f0_sumsq"])
        e_mean, e_std = _ms(acc["e_n"], acc["e_sum"], acc["e_sumsq"])
        return {
            "f0_norm_mean": f0_mean, "f0_norm_std": f0_std,
            "energy_norm_mean": e_mean, "energy_norm_std": e_std,
            "f0_frames": acc["f0_n"], "energy_frames": acc["e_n"],
        }

    @staticmethod
    def empty_log_stats() -> dict:
        return {"f0_n": 0, "f0_sum": 0.0, "f0_sumsq": 0.0,
                "e_n": 0, "e_sum": 0.0, "e_sumsq": 0.0}

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

        log_f0 = log_f0_abs.copy()
        voicing = voiced.astype(np.float32)

        if self.norm_mode == "fixed":
            # Fixed global stats (fix A): same normalization at train and
            # inference regardless of segment. Requires the model to be TRAINED
            # with the same stats. Using this on a model trained with
            # per-utterance norm corrupts prosody → poor cloning + ref leakage.
            log_energy = np.log(np.maximum(energy, self.energy_floor)).astype(np.float32)
            if voiced.sum() > 0:
                log_f0[voiced] = (log_f0[voiced] - self.f0_norm_mean) / self.f0_norm_std
            log_energy = (log_energy - self.energy_norm_mean) / self.energy_norm_std
        else:
            # Per-utterance z-norm (DEFAULT — matches the original/working model).
            # The reference is normalized against itself. Energy uses the
            # (energy + 1e-8) floor exactly as the trained model expects.
            log_energy = np.log(energy + 1e-8).astype(np.float32)
            if voiced.sum() > 1:
                m = log_f0[voiced].mean(); s = log_f0[voiced].std() + 1e-8
                log_f0[voiced] = (log_f0[voiced] - m) / s
            if n > 1:
                em = log_energy.mean(); es = log_energy.std() + 1e-8
                log_energy = (log_energy - em) / es

        # Delta features (first-order difference)
        delta_f0 = np.zeros(n, dtype=np.float32)
        delta_energy = np.zeros(n, dtype=np.float32)
        if n > 1:
            delta_f0[1:] = np.diff(log_f0)
            delta_energy[1:] = np.diff(log_energy)
            # Mask delta_f0 at voicing boundaries: where current OR previous
            # frame is unvoiced, log_f0 jumps 0↔value, producing a spurious
            # "pitch velocity" that is not real pitch motion. Zero those out so
            # delta_f0 reflects actual pitch change within voiced regions only.
            boundary = ~(voiced[1:] & voiced[:-1])
            delta_f0[1:][boundary] = 0.0

        # Local speaking rate: voicing density in sliding window (~200ms).
        # Z-normalized exactly as the trained (__58__) model expects.
        win = min(15, max(3, n // 4))
        if n >= win:
            kernel = np.ones(win, dtype=np.float32) / win
            local_rate = np.convolve(voicing, kernel, mode='same').astype(np.float32)
        else:
            local_rate = np.full(n, voicing.mean(), dtype=np.float32)
        if n > 1:
            lr_mean = local_rate.mean()
            lr_std = local_rate.std() + 1e-8
            local_rate = (local_rate - lr_mean) / lr_std

        # Channel 5: absolute log-F0. Raw value ~5-6 (log of 150-250 Hz) — an
        # order of magnitude larger than the other ~0±1 channels, so it dominates
        # prosody_raw_proj's input at init and slows learning. Optionally subtract
        # a FIXED constant (NOT per-utterance mean) to recenter it to ~0 while
        # PRESERVING absolute pitch level (bass vs tenor) — the whole point of
        # this channel for cloning. Division by a fixed scale equalizes magnitude.
        # Fixed (not per-utterance) so the same physical pitch maps identically
        # for a reference clip and a full utterance (no train/inference drift).
        log_f0_abs_out = log_f0_abs
        if self.center_log_f0_abs:
            log_f0_abs_out = log_f0_abs.copy()
            log_f0_abs_out[voiced] = (log_f0_abs[voiced] - self.f0_abs_center) / self.f0_abs_scale

        features = np.stack([log_f0, voicing, log_energy, delta_f0, delta_energy, log_f0_abs_out, local_rate], axis=-1)
        mask = np.ones(n, dtype=bool)

        return torch.from_numpy(features), torch.from_numpy(mask)

    def precompute(self, wav: torch.Tensor, sr: int = 24000) -> dict[str, torch.Tensor]:
        """Precompute raw features for dataset caching."""
        features, mask = self.extract_raw(wav, sr)
        return {"prosody_raw": features.cpu(), "prosody_mask": mask.cpu()}
