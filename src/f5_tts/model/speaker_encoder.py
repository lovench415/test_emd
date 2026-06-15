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
        "wavlm_sv"      — microsoft/wavlm-base-plus-sv  (512-d, best quality)
        "wavlm_sv_onnx" — ONNX export of the same model (512-d, no transformers
                          WavLMForXVector dependency, fast cold start). Auto-
                          downloads from HF repo stik168/wavlm_base_plus_sv_onnx.
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
        self._use_onnx = False
        self._load_backend(backend, device)
        self.proj = self.make_projection(self.raw_dim, output_dim)

    # ── Backend loading ───────────────────────────────────────────────

    def _load_backend(self, backend: str, device: str):
        if backend == "wavlm_sv":
            self._load_wavlm(device)
        elif backend == "wavlm_sv_onnx":
            self._load_wavlm_onnx(device)
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

    def _load_wavlm_onnx(self, device: str):
        """Load microsoft/wavlm-base-plus-sv exported to ONNX.

        Model: stik168/wavlm_base_plus_sv_onnx
          - wavlm_base_plus_sv.onnx
          - wavlm_base_plus_sv.onnx.data   (external weights sibling)
        onnxruntime resolves the .data file relative to the .onnx path at
        session-creation time, so the graph must be loaded from the on-disk
        path (never from bytes) and the two files must sit together.

        The graph takes a 16kHz waveform input and returns the 512-d x-vector
        speaker embedding. We still apply the learnable projection + L2 norm
        downstream, identical to the torch backend, so cached embeddings from
        either backend are interchangeable (both 512-d raw).

        Model path resolution (first hit wins):
          1. self.onnx_model_path if set before construction
          2. env var WAVLM_SV_ONNX_PATH (file path or a directory)
          3. "wavlm_base_plus_sv.onnx" in CWD
          4. auto-download from HF repo stik168/wavlm_base_plus_sv_onnx
             (both the .onnx and its .onnx.data are fetched into the HF cache)

        Direct download URLs (if fetching manually):
          https://huggingface.co/stik168/wavlm_base_plus_sv_onnx/resolve/main/wavlm_base_plus_sv.onnx
          https://huggingface.co/stik168/wavlm_base_plus_sv_onnx/resolve/main/wavlm_base_plus_sv.onnx.data
        """
        import os

        import onnxruntime as ort
        from transformers import Wav2Vec2FeatureExtractor

        repo_id = getattr(self, "onnx_repo_id", None) or os.environ.get(
            "WAVLM_SV_ONNX_REPO", "stik168/wavlm_base_plus_sv_onnx"
        )
        onnx_file = "wavlm_base_plus_sv.onnx"
        data_file = "wavlm_base_plus_sv.onnx.data"

        def _resolve(p):
            if not p:
                return None
            if os.path.isdir(p):
                cand = os.path.join(p, onnx_file)
                return cand if os.path.exists(cand) else None
            return p if os.path.exists(p) else None

        candidates = [
            getattr(self, "onnx_model_path", None),
            os.environ.get("WAVLM_SV_ONNX_PATH"),
            onnx_file,
            os.path.join(os.path.dirname(__file__), "..", "models", onnx_file),
        ]
        model_path = next((r for r in (_resolve(c) for c in candidates) if r), None)

        # A local .onnx with EXTERNAL weights is useless without its .data
        # sibling. If the graph is present but .data is missing next to it,
        # fetch the .data into that same dir so onnxruntime can resolve it.
        if model_path is not None:
            sibling = os.path.join(os.path.dirname(os.path.abspath(model_path)), data_file)
            if not os.path.exists(sibling):
                try:
                    from huggingface_hub import hf_hub_download
                    fetched = hf_hub_download(repo_id=repo_id, filename=data_file)
                    # symlink/copy the fetched .data next to the local .onnx
                    try:
                        os.symlink(fetched, sibling)
                    except Exception:
                        import shutil
                        shutil.copyfile(fetched, sibling)
                except Exception:
                    # No .data obtainable; if weights are external the session
                    # will fail below with a clear onnxruntime error.
                    pass

        # Last resort: pull both files from HF hub into the cache.
        if model_path is None:
            try:
                from huggingface_hub import hf_hub_download
                try:
                    hf_hub_download(repo_id=repo_id, filename=data_file)
                except Exception:
                    pass  # some exports inline weights → no .data file
                model_path = hf_hub_download(repo_id=repo_id, filename=onnx_file)
            except Exception as e:
                raise FileNotFoundError(
                    "wavlm_sv_onnx backend: ONNX model not found locally and "
                    f"auto-download from '{repo_id}' failed ({e}). Set "
                    "WAVLM_SV_ONNX_PATH to wavlm_base_plus_sv.onnx and keep its "
                    "sibling wavlm_base_plus_sv.onnx.data in the same folder. "
                    "Download: https://huggingface.co/stik168/wavlm_base_plus_sv_onnx"
                )

        providers = ["CPUExecutionProvider"]
        if str(device).startswith("cuda"):
            avail = ort.get_available_providers()
            if "CUDAExecutionProvider" in avail:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.onnx_session = ort.InferenceSession(model_path, providers=providers)
        graph_inputs = self.onnx_session.get_inputs()
        input_names = [i.name for i in graph_inputs]
        # WavLM ONNX exports typically name the input "input_values"; fall back
        # to the first declared input defensively.
        self.onnx_input_name = (
            "input_values" if "input_values" in input_names else input_names[0]
        )
        sel = next((i for i in graph_inputs if i.name == self.onnx_input_name), graph_inputs[0])
        self.onnx_input_rank = len(sel.shape) if getattr(sel, "shape", None) else 2
        self.onnx_output_name = self.onnx_session.get_outputs()[0].name

        # Same feature extractor as the torch model (zero-mean/unit-var at 16kHz).
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        )
        self.raw_dim = SPEAKER_RAW_DIMS["wavlm_sv_onnx"]
        self._use_onnx = True

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
    def extract_raw(self, wav: torch.Tensor, sr: int = 24000,
                    lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Extract raw speaker embedding: (B, raw_dim).

        lengths: optional (B,) true sample lengths in the INPUT sample rate `sr`
            (i.e. measured on `wav` as passed in), used when the input is ALREADY
            padded to equal length. Without it, WavLM-SV pools over padding
            silence → contaminated speaker embedding. Lengths are rescaled to
            self.sample_rate internally to match the (possibly resampled) audio.
        """
        wav = self._prepare_wav(wav, sr)
        # lengths are given at the INPUT rate `sr`; _prepare_wav resampled the
        # audio to self.sample_rate. Rescale ONCE here so slices/masks in every
        # backend branch index the resampled waveform correctly (no-op when
        # sr == self.sample_rate, e.g. prepare_data passes 16k).
        if lengths is not None and sr != self.sample_rate:
            lengths = (lengths.float() * (self.sample_rate / sr)).long().clamp_min(1)

        if self.backend == "wavlm_sv":
            inputs = self.feature_extractor(
                self._wav_to_list(wav),
                sampling_rate=self.sample_rate, return_tensors="pt", padding=True,
            )
            input_values = inputs.input_values.to(wav.device)
            attention_mask = getattr(inputs, "attention_mask", None)
            attention_mask = attention_mask.to(wav.device) if attention_mask is not None else None
            # Rebuild a correct mask from true lengths when input was pre-padded
            # (feature extractor's mask is all-ones for equal-length input).
            # NOTE: lengths were already rescaled to self.sample_rate at the top
            # of extract_raw — do NOT rescale again here (double-rescale would
            # shrink the mask and cut off real speech for non-16k inputs).
            if lengths is not None:
                lengths = lengths.to(wav.device).float()
                T_axis = input_values.shape[-1]
                lengths = lengths.round().clamp(min=1, max=T_axis).long()
                sample_mask = torch.arange(T_axis, device=wav.device).unsqueeze(0) < lengths.unsqueeze(1)
                attention_mask = sample_mask.long()
            outputs = self.encoder(input_values, attention_mask=attention_mask)
            return outputs.embeddings

        if self.backend == "wavlm_sv_onnx":
            import numpy as np
            # Same preprocessing as torch WavLM (16kHz, zero-mean/unit-var).
            # PADDING SAFETY: when `lengths` is given the input is pre-padded to
            # equal length; pooling over padding contaminates the x-vector (same
            # bug fixed for the torch backend). If the ONNX graph exposes an
            # attention_mask input — feed a mask built from true lengths.
            # Otherwise fall back to per-sample inference on TRIMMED audio
            # (cheap: length-bucketed batches are near-homogeneous anyway).
            graph_input_names = [i.name for i in self.onnx_session.get_inputs()]
            has_mask_input = "attention_mask" in graph_input_names

            if lengths is not None and not has_mask_input:
                embs = []
                for j in range(wav.shape[0]):
                    L = int(lengths[j].item())
                    a = wav[j:j + 1, :max(L, 1)]
                    inputs = self.feature_extractor(
                        self._wav_to_list(a),
                        sampling_rate=self.sample_rate, return_tensors="np", padding=True,
                    )
                    iv = inputs.input_values.astype(np.float32)
                    feed = iv[0] if (self.onnx_input_rank == 1) else iv
                    out = self.onnx_session.run(
                        [self.onnx_output_name], {self.onnx_input_name: feed}
                    )[0]
                    e = torch.from_numpy(np.asarray(out)).float()
                    embs.append(e.reshape(-1))  # (D,) from either (D,) or (1, D)
                return torch.stack(embs).to(wav.device)

            inputs = self.feature_extractor(
                self._wav_to_list(wav),
                sampling_rate=self.sample_rate, return_tensors="np", padding=True,
            )
            input_values = inputs.input_values.astype(np.float32)  # (B, T)
            feed_dict = {self.onnx_input_name: input_values}
            if has_mask_input:
                if lengths is not None:
                    am = (
                        np.arange(input_values.shape[-1])[None, :]
                        < lengths.cpu().numpy()[:, None]
                    ).astype(np.int64)
                else:
                    am = getattr(inputs, "attention_mask", None)
                    am = am.astype(np.int64) if am is not None else np.ones_like(input_values, dtype=np.int64)
                feed_dict["attention_mask"] = am
            if self.onnx_input_rank == 1 and input_values.shape[0] == 1:
                feed_dict[self.onnx_input_name] = input_values[0]  # (T,) for 1-D graphs
                if "attention_mask" in feed_dict:
                    feed_dict["attention_mask"] = feed_dict["attention_mask"][0]  # keep rank consistent
            out = self.onnx_session.run([self.onnx_output_name], feed_dict)[0]
            emb = torch.from_numpy(np.asarray(out)).to(wav.device).float()
            if emb.ndim == 1:
                emb = emb.unsqueeze(0)
            return emb  # (B, 512), projected + normalized downstream

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
