"""
F5-TTS Enhanced — FastAPI inference server.

Start:
    python -m f5_tts.infer.server \
        --ckpt_path ckpts/model/model_last.pt \
        --host 0.0.0.0 \
        --port 8000

C# client calls POST /synthesize with JSON body,
receives WAV audio bytes in response.
"""

from __future__ import annotations

import argparse
import io
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

# ── Globals (loaded once at startup) ────────────────────────────────────────

_model = None
_vocoder = None
_spk_enc = None
_emo_enc = None
_prosody_enc = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_args = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup, release on shutdown."""
    global _model, _vocoder, _spk_enc, _emo_enc, _prosody_enc

    from f5_tts.infer.enhanced_infer import (
        load_embedding_extractors,
        load_enhanced_model,
        load_prosody_encoder,
        load_vocoder,
    )

    print(f"Loading models on {_device}...")

    _vocoder = load_vocoder(device=_device)
    _model = load_enhanced_model(_args.ckpt_path, device_str=_device)
    _spk_enc, _emo_enc = load_embedding_extractors(device_str=_device)

    has_prosody = (
        hasattr(_model.transformer, "cond_aggregator")
        and hasattr(_model.transformer.cond_aggregator, "prosody_cross_attns")
    )
    _prosody_enc = (
        load_prosody_encoder(backend=_args.prosody_backend, device_str=_device)
        if has_prosody
        else None
    )

    print("✓ Models loaded. Server ready.")
    yield

    # Cleanup
    del _model, _vocoder, _spk_enc, _emo_enc, _prosody_enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="F5-TTS Enhanced", lifespan=lifespan)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — C# can poll this before sending requests."""
    return {
        "status": "ok",
        "device": _device,
        "model_loaded": _model is not None,
    }


@app.post("/synthesize")
async def synthesize(
    ref_audio: UploadFile = File(..., description="Reference audio file (WAV/MP3)"),
    gen_text: str = Form(..., description="Text to synthesize"),
    ref_text: str = Form("", description="Transcript of reference audio (optional)"),
    emotion_cfg_strength: float = Form(0.5),
    prosody_cfg_strength: float = Form(0.0),
    output_sample_rate: Optional[int] = Form(None, description="Output sample rate (e.g. 32000)"),
):
    """
    Synthesize speech and return WAV bytes.

    C# usage:
        var content = new MultipartFormDataContent();
        content.Add(new ByteArrayContent(refAudioBytes), "ref_audio", "ref.wav");
        content.Add(new StringContent(text), "gen_text");
        var response = await client.PostAsync("/synthesize", content);
        var wavBytes = await response.Content.ReadAsByteArrayAsync();
    """
    from f5_tts.infer.enhanced_infer import infer_enhanced
    import torchaudio.functional as AF

    if _model is None:
        raise HTTPException(503, "Models not loaded")

    # Save uploaded reference audio to temp file
    suffix = os.path.splitext(ref_audio.filename or "ref.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await ref_audio.read())
        tmp_path = tmp.name

    try:
        wave, sr = infer_enhanced(
            ref_audio=tmp_path,
            ref_text=ref_text,
            gen_text=gen_text,
            model=_model,
            vocoder=_vocoder,
            speaker_encoder=_spk_enc,
            emotion_encoder=_emo_enc,
            prosody_encoder=_prosody_enc,
            emotion_cfg_strength=emotion_cfg_strength,
            prosody_cfg_strength=prosody_cfg_strength,
            device_str=_device,
        )
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")
    finally:
        os.unlink(tmp_path)

    # Optional resample
    if output_sample_rate and output_sample_rate != sr:
        import numpy as np
        wave_t = torch.from_numpy(wave).unsqueeze(0)
        wave_t = AF.resample(wave_t, orig_freq=sr, new_freq=output_sample_rate)
        wave = wave_t.squeeze(0).numpy()
        sr = output_sample_rate

    # Encode as WAV bytes
    buf = io.BytesIO()
    sf.write(buf, wave, sr, format="WAV")
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="audio/wav",
        headers={"X-Sample-Rate": str(sr)},
    )


@app.post("/synthesize_batch")
async def synthesize_batch(
    ref_audio: UploadFile = File(...),
    texts: str = Form(..., description="Newline-separated texts to synthesize"),
    ref_text: str = Form(""),
    emotion_cfg_strength: float = Form(0.5),
    prosody_cfg_strength: float = Form(0.0),
):
    """
    Synthesize multiple texts with the same reference.
    Returns a ZIP file containing 0.wav, 1.wav, 2.wav...
    """
    import zipfile
    from f5_tts.infer.enhanced_infer import infer_enhanced

    if _model is None:
        raise HTTPException(503, "Models not loaded")

    text_list = [t.strip() for t in texts.splitlines() if t.strip()]
    if not text_list:
        raise HTTPException(400, "No texts provided")

    suffix = os.path.splitext(ref_audio.filename or "ref.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await ref_audio.read())
        tmp_path = tmp.name

    zip_buf = io.BytesIO()
    try:
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, text in enumerate(text_list):
                wave, sr = infer_enhanced(
                    ref_audio=tmp_path,
                    ref_text=ref_text,
                    gen_text=text,
                    model=_model,
                    vocoder=_vocoder,
                    speaker_encoder=_spk_enc,
                    emotion_encoder=_emo_enc,
                    prosody_encoder=_prosody_enc,
                    emotion_cfg_strength=emotion_cfg_strength,
                    prosody_cfg_strength=prosody_cfg_strength,
                    device_str=_device,
                )
                wav_buf = io.BytesIO()
                sf.write(wav_buf, wave, sr, format="WAV")
                zf.writestr(f"{i}.wav", wav_buf.getvalue())
    except Exception as e:
        raise HTTPException(500, f"Batch inference failed: {e}")
    finally:
        os.unlink(tmp_path)

    zip_buf.seek(0)
    return Response(
        content=zip_buf.read(),
        media_type="application/zip",
    )


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    global _args, _device

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default=None)
    p.add_argument("--prosody_backend", default="dio",
                   choices=["dio", "harvest", "rmvpe", "crepe"])
    p.add_argument("--workers", type=int, default=1,
                   help="Number of uvicorn workers (1 recommended — GPU is shared)")
    _args = p.parse_args()

    if _args.device:
        _device = _args.device

    uvicorn.run(
        "f5_tts.infer.server:app",
        host=_args.host,
        port=_args.port,
        workers=_args.workers,
    )


if __name__ == "__main__":
    main()
