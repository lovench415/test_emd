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
        "emotion2vec_onnx" — emotion2vec+ base exported to ONNX (768-d, no funasr)
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
        if backend == "emotion2vec_onnx":
            self._load_emotion2vec_onnx(device)
        elif backend.startswith("emotion2vec"):
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

            model_name = {
                "emotion2vec_base":       "iic/emotion2vec_base",
                "emotion2vec_plus":       "iic/emotion2vec_plus_base",
                "emotion2vec_plus_large": "iic/emotion2vec_plus_large",
            }.get(variant, "iic/emotion2vec_plus_base")
            self.encoder = AutoModel(model=model_name, device=device)
            self._use_funasr = True
            # Feature dim from the registry (base/plus = 768, plus_large = 1024,
            # confirmed). The registry is the single source of truth that the
            # cache manifest and the model projection also use, so they stay
            # consistent.
            self.raw_dim = EMOTION_RAW_DIMS.get(variant, 768)
        except ImportError:
            self._load_hf("facebook/wav2vec2-base", device, 768)
            self._use_funasr = False

    def _load_emotion2vec_onnx(self, device: str):
        """Load emotion2vec_plus_base exported to ONNX.

        Model: stik168/emotion2vec_plus_base_onnx — file "emotion2vec_emb.onnx"
        with weights in the sibling "emotion2vec_emb.onnx.data" (ONNX external
        data). onnxruntime loads the .data automatically as long as it sits next
        to the .onnx file and the session is created from that file path, so we
        must NOT load the graph from bytes — always pass the on-disk path.

        Runs via onnxruntime instead of funasr/torch — no funasr dependency,
        faster cold start, easy CPU/GPU switch. The graph takes a single input
        named "source" = a 1D float32 16kHz waveform and returns a list whose
        FIRST element is the frame-level embedding (T_frames, 768).

        Model path resolution (first hit wins):
          1. self.onnx_model_path if set before construction
          2. env var EMOTION2VEC_ONNX_PATH (file path or a directory)
          3. "emotion2vec_emb.onnx" / "emotion2vec_plus_base.onnx" in CWD
          4. auto-download from HF repo stik168/emotion2vec_plus_base_onnx
             (both the .onnx and its .onnx.data are fetched into the HF cache)
        """
        import os

        import onnxruntime as ort

        repo_id = getattr(self, "onnx_repo_id", None) or os.environ.get(
            "EMOTION2VEC_ONNX_REPO", "stik168/emotion2vec_plus_base_onnx"
        )
        onnx_file = "emotion2vec_emb.onnx"
        data_file = "emotion2vec_emb.onnx.data"

        def _resolve(p):
            # Accept either a direct .onnx path or a directory containing it.
            if not p:
                return None
            if os.path.isdir(p):
                for fn in (onnx_file, "emotion2vec_plus_base.onnx"):
                    cand = os.path.join(p, fn)
                    if os.path.exists(cand):
                        return cand
                return None
            return p if os.path.exists(p) else None

        candidates = [
            getattr(self, "onnx_model_path", None),
            os.environ.get("EMOTION2VEC_ONNX_PATH"),
            onnx_file,
            "emotion2vec_plus_base.onnx",
        ]
        model_path = next((r for r in (_resolve(c) for c in candidates) if r), None)

        # Last resort: pull from the HF hub. The .onnx uses external weights, so
        # fetch the .data sibling into the SAME cache dir first — onnxruntime
        # resolves it relative to the .onnx path at session-creation time.
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
                    "emotion2vec_onnx backend: ONNX model not found locally and "
                    f"auto-download from '{repo_id}' failed ({e}). Set "
                    "EMOTION2VEC_ONNX_PATH to emotion2vec_emb.onnx and keep its "
                    "sibling emotion2vec_emb.onnx.data in the same folder."
                )

        # Prefer CUDA EP when the encoder is on GPU and it is available.
        providers = ["CPUExecutionProvider"]
        if str(device).startswith("cuda"):
            avail = ort.get_available_providers()
            if "CUDAExecutionProvider" in avail:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.onnx_session = ort.InferenceSession(model_path, providers=providers)
        # Resolve the input name once (graph exports it as "source"; fall back
        # to the first declared input defensively).
        graph_inputs = self.onnx_session.get_inputs()
        input_names = [i.name for i in graph_inputs]
        self.onnx_input_name = "source" if "source" in input_names else input_names[0]
        # Record the declared rank of the chosen input so we feed the right
        # shape: (T,) for 1-D graphs, (1, T) for 2-D (batched) graphs.
        sel = next((i for i in graph_inputs if i.name == self.onnx_input_name), graph_inputs[0])
        self.onnx_input_rank = len(sel.shape) if getattr(sel, "shape", None) else 1

        # Resolve the frame-level output. This graph exposes two outputs:
        # "frames" (1, T, 768) and "utterance" (1, 768); we take the frame one
        # and derive the global vector from it (masked mean) for consistency
        # with the other backends. Fall back to the first output by position.
        out_names = [o.name for o in self.onnx_session.get_outputs()]
        self.onnx_frames_output = "frames" if "frames" in out_names else out_names[0]

        self.raw_dim = 768
        self._use_funasr = False
        self._use_onnx = True

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
        global_feat, frame_feat, _ = self.extract_raw_with_mask(wav, sr)
        return global_feat, frame_feat

    @torch.no_grad()
    def extract_raw_with_mask(
        self, wav: torch.Tensor, sr: int = 24000,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (global_feat, frame_feat, frame_mask):
            global_feat: (B, raw_dim)
            frame_feat:  (B, T_frames, raw_dim)
            frame_mask:  (B, T_frames) bool

        lengths: optional (B,) true sample lengths (at self.sample_rate) for
            inputs that are ALREADY padded to equal length. Without this, a
            pre-padded batch yields an all-ones attention_mask (the feature
            extractor can't tell padding from signal), contaminating both
            frame_feat (padding frames) and global_feat (mean over padding).
        """
        # Capture true (input-sr) lengths, then resample. _prepare_wav may change
        # the sample count, so rescale lengths to post-resample units once here —
        # both the funasr and HF paths below interpret `lengths` against the
        # already-resampled tensor. In the common case (input sr == encoder sr,
        # both 16k) the ratio is 1.0 and this is a no-op.
        rescaled_lengths = lengths
        if lengths is not None:
            if not torch.is_tensor(lengths):
                lengths = torch.tensor(lengths, dtype=torch.long)
            in_T = wav.shape[-1]
            wav = self._prepare_wav(wav, sr)
            out_T = wav.shape[-1]
            if out_T != in_T:
                ratio = out_T / float(in_T) if in_T > 0 else 1.0
                rescaled_lengths = (lengths.float() * ratio).round().long().clamp(min=1, max=out_T)
            else:
                rescaled_lengths = lengths.long().clamp(min=1, max=out_T)
        else:
            wav = self._prepare_wav(wav, sr)

        if getattr(self, "_use_onnx", False):
            return self._extract_onnx(wav, lengths=rescaled_lengths)
        if getattr(self, "_use_funasr", False):
            return self._extract_funasr(wav, lengths=rescaled_lengths)
        return self._extract_hf(wav, lengths=rescaled_lengths)

    def _extract_funasr(self, wav: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import numpy as np

        # Trim each row to its true length before handing to funasr. The batch
        # arrives pre-padded to equal length; without trimming, the trailing
        # zeros would be treated as signal and contaminate the per-utterance
        # emotion features (funasr has no attention-mask input here).
        n = wav.shape[0]
        if lengths is not None:
            T = wav.shape[-1]
            # `lengths` are already in post-resample sample units (rescaled by the
            # caller). Clamp to T as a safety guard against rounding/off-by-one.
            lens = [max(1, min(int(l), T)) for l in lengths.tolist()]
            wav_list = [wav[i, :lens[i]].cpu().numpy() for i in range(n)]
        else:
            wav_list = [wav[i].cpu().numpy() for i in range(n)]

        # ── #1: batched funasr call ──
        # funasr's AutoModel.generate accepts a list of inputs and returns a
        # list of per-item results in order (same as wav.scp / file-list mode).
        # One call lets funasr batch internally instead of a Python loop with
        # one GPU dispatch per utterance. Fall back to per-sample on any error
        # (older funasr signatures, or a backend that rejects list input).
        results = None
        try:
            results = self.encoder.generate(
                wav_list, output_dir=None, granularity="frame",
                batch_size=len(wav_list),
            )
            if not (isinstance(results, list) and len(results) == len(wav_list)):
                results = None  # unexpected shape → use fallback
        except Exception:
            results = None

        def _one(idx):
            if results is not None:
                return results[idx]
            r = self.encoder.generate(wav_list[idx], output_dir=None, granularity="frame")
            return r[0] if isinstance(r, list) and r else None

        frame_feats, global_feats, valid_lens = [], [], []
        for i in range(wav.shape[0]):
            item = _one(i)
            feats = item.get("feats") if isinstance(item, dict) else None
            if feats is not None and np.asarray(feats).size > 0:
                # np.array(feats) may be float64; cast to float32 so it stacks
                # cleanly with the float32 zero-fallback rows below.
                ft = torch.from_numpy(np.array(feats, dtype=np.float32)).to(wav.device)
                frame_feats.append(ft)
                global_feats.append(ft.mean(dim=0))
                valid_lens.append(ft.shape[0])
            else:
                # Fallback: zero features. Length-1 tensor for padding, but
                # valid_len=0 so the frame mask is all-False (the zero frame must
                # NOT be treated as a real emotion frame downstream).
                frame_feats.append(torch.zeros(1, self.raw_dim, device=wav.device))
                global_feats.append(torch.zeros(self.raw_dim, device=wav.device))
                valid_lens.append(0)

        return self._stack_frame_lists(frame_feats, global_feats, wav.device, valid_lens)

    @staticmethod
    def _stack_frame_lists(frame_feats: list, global_feats: list, device, valid_lens: list | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad a list of per-sample (T_i, D) frame tensors into a padded batch
        plus an exact frame mask. Shared by the funasr and ONNX paths so both
        return identical (global_feat, frame_feat, frame_mask) structure.

        valid_lens: optional per-sample count of GENUINELY valid frames. Defaults
        to each tensor's own length. Zero-fallback rows (failed extraction) pass a
        length-1 zero tensor for padding but valid_len=0, so the mask is all-False
        for them — otherwise the mask would mark the zero frame valid and
        downstream cross-attention/direct-add would treat zeros as real emotion.
        """
        global_feat = torch.stack(global_feats)
        max_len = max(f.shape[0] for f in frame_feats)
        padded = [F.pad(f, (0, 0, 0, max_len - f.shape[0])) for f in frame_feats]
        frame_feat = torch.stack(padded)
        if valid_lens is None:
            valid_lens = [f.shape[0] for f in frame_feats]
        lens = torch.tensor(valid_lens, device=device, dtype=torch.long).clamp(min=0)
        frame_mask = torch.arange(max_len, device=device).unsqueeze(0) < lens.unsqueeze(1)
        return global_feat, frame_feat, frame_mask

    def _extract_onnx(self, wav: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Frame-level emotion features via the emotion2vec ONNX graph.

        The graph (stik168/emotion2vec_plus_base_onnx) takes input "source" of
        shape (1, T) float32 16kHz and returns "frames" (1, T_frames, 768) plus
        "utterance" (1, 768). We run one sample at a time, trimming padding via
        `lengths`, take the "frames" output, and derive the global vector as the
        masked mean of frames (consistent with the other backends).

        NOTE: this graph normalizes the waveform internally (a LayerNormalization
        over the time axis is the first op on "source"), so we do NOT normalize
        in Python by default — doing so would double-normalize. Override with
        EMOTION2VEC_ONNX_NORMALIZE=1 (or self.onnx_normalize=True) only for an
        export that lacks the built-in norm.
        """
        import numpy as np
        import os as _os

        # Default OFF: the shipped graph already layer-norms the input.
        normalize_input = getattr(self, "onnx_normalize", None)
        if normalize_input is None:
            normalize_input = _os.environ.get("EMOTION2VEC_ONNX_NORMALIZE", "0") == "1"

        n = wav.shape[0]
        T = wav.shape[-1]
        if lengths is not None:
            if not torch.is_tensor(lengths):
                lengths = torch.tensor(lengths, dtype=torch.long)
            lens = [max(1, min(int(l), T)) for l in lengths.tolist()]
        else:
            lens = [T] * n

        out_name = getattr(self, "onnx_frames_output", None)
        # The emotion2vec conv stack needs a minimum input length to emit >=1
        # frame: frames = ((T//80 - 3)//2 - 1)//2 + 1, so T < ~400 samples
        # (<25ms @16k) yields 0 frames and an empty/invalid output. Pad such
        # clips with zeros up to the minimum (after the graph's internal
        # layer-norm a short zero tail is harmless) so we still get features
        # instead of a silent zero-fallback.
        MIN_SAMPLES = 480  # comfortably above the 400-sample threshold
        frame_feats, global_feats, valid_lens = [], [], []
        for i in range(n):
            x = wav[i, :lens[i]].detach().cpu().float().numpy()
            if x.shape[0] < MIN_SAMPLES:
                x = np.pad(x, (0, MIN_SAMPLES - x.shape[0]))
            # Optional waveform normalization (off by default; graph does it).
            if normalize_input and x.size > 1 and x.std() > 1e-6:
                x = (x - x.mean()) / (x.std() + 1e-8)
            # Match the graph's declared input rank: (1, T) for batched 2-D
            # graphs, (T,) for 1-D. This export is 2-D.
            if getattr(self, "onnx_input_rank", 1) >= 2:
                x = x.reshape(1, -1)
            # Graph input is float32 (elem_type=1); ensure dtype after any
            # normalization/pad so onnxruntime doesn't reject the feed.
            x = np.ascontiguousarray(x, dtype=np.float32)
            feats = None
            try:
                fetch = [out_name] if out_name else None
                out = self.onnx_session.run(fetch, {self.onnx_input_name: x})
                if isinstance(out, (list, tuple)) and len(out) > 0:
                    feats = np.asarray(out[0], dtype=np.float32)
                    # Drop the leading batch axis: (1, T, D) → (T, D).
                    if feats.ndim == 3 and feats.shape[0] == 1:
                        feats = feats[0]
            except Exception:
                feats = None

            if feats is not None and feats.ndim == 2 and feats.shape[0] > 0:
                ft = torch.from_numpy(np.ascontiguousarray(feats)).to(wav.device)
                frame_feats.append(ft)
                global_feats.append(ft.mean(dim=0))
                valid_lens.append(ft.shape[0])
            else:
                # Zero-fallback: length-1 zero tensor for padding, valid_len=0 so
                # the frame mask is all-False (zeros must not read as real emotion).
                frame_feats.append(torch.zeros(1, self.raw_dim, device=wav.device))
                global_feats.append(torch.zeros(self.raw_dim, device=wav.device))
                valid_lens.append(0)

        return self._stack_frame_lists(frame_feats, global_feats, wav.device, valid_lens)

    def _extract_hf(self, wav: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.feature_extractor(
            self._wav_to_list(wav),
            sampling_rate=self.sample_rate, return_tensors="pt", padding=True,
        )
        input_values = inputs.input_values.to(wav.device)
        attention_mask = getattr(inputs, "attention_mask", None)
        attention_mask = attention_mask.to(wav.device) if attention_mask is not None else None

        # If explicit lengths given (input was pre-padded to equal length), the
        # feature extractor's attention_mask is all-ones and useless. Rebuild a
        # correct sample-level mask from the true lengths so padding is ignored.
        if lengths is not None:
            lengths = lengths.to(wav.device)
            sample_mask = torch.arange(input_values.shape[-1], device=wav.device).unsqueeze(0) < lengths.unsqueeze(1)
            attention_mask = sample_mask.long()

        outputs = self.encoder(input_values, attention_mask=attention_mask, output_hidden_states=True)
        frame_feat = outputs.last_hidden_state       # (B, T, raw_dim)

        if attention_mask is not None and hasattr(self.encoder, "_get_feat_extract_output_lengths"):
            input_lengths = attention_mask.sum(dim=-1)
            frame_lengths = self.encoder._get_feat_extract_output_lengths(input_lengths).to(wav.device)
            frame_lengths = frame_lengths.clamp(max=frame_feat.shape[1])
            frame_mask = torch.arange(frame_feat.shape[1], device=wav.device).unsqueeze(0) < frame_lengths.unsqueeze(1)
        else:
            frame_mask = torch.ones(frame_feat.shape[:2], dtype=torch.bool, device=wav.device)

        denom = frame_mask.sum(dim=1, keepdim=True).clamp_min(1).to(frame_feat.dtype)
        global_feat = (frame_feat * frame_mask.unsqueeze(-1).to(frame_feat.dtype)).sum(dim=1) / denom
        return global_feat, frame_feat, frame_mask

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
        """DEPRECATED projection path — see note below.

        Like SpeakerEncoder, the pipeline caches RAW emotion features
        (extract_raw / extract_raw_with_mask) and projects them inside the model
        via cond_aggregator.emotion_raw_proj / frame_raw_proj (the canonical,
        trained heads, which also apply the emotion_bottleneck). The encoder-side
        global_proj / frame_proj used here are NOT trained by the pipeline and
        always L2-normalize → embeddings incompatible with the model. Do not use
        forward()/project_cached() for the training/inference pipeline; use
        extract_raw and let the model project.
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
