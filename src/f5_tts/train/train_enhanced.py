"""
F5-TTS Enhanced — Training Entry Point
=========================================

Usage:
    python train_enhanced.py --dataset_dir /data --embedding_dir /emb
    accelerate launch --multi_gpu train_enhanced.py --dataset_dir /data --embedding_dir /emb
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
from collections import defaultdict
from importlib.resources import files
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from f5_tts.model.backbones.enhanced_dit import EnhancedDiT
from f5_tts.model.enhanced_cfm import EnhancedCFM
from f5_tts.model.enhanced_dataset import EnhancedDataset, collate_fn
from f5_tts.model.speaker_encoder import SpeakerEncoder
from f5_tts.model.emotion_encoder import EmotionEncoder
from f5_tts.model.encoder_utils import SPEAKER_RAW_DIMS, EMOTION_RAW_DIMS
from f5_tts.model.prosody_encoder import ProsodyEncoder, PROSODY_RAW_DIM
from f5_tts.model.utils import get_tokenizer, exists, lens_to_mask, mask_from_frac_lengths, convert_char_to_pinyin
from f5_tts.model.samplers import (
    DynamicBatchSampler,
    BucketDynamicBatchSampler,
    SpeakerAwareBucketDynamicBatchSampler,
    SpeakerBalancedDynamicBatchSampler,
)

# ── Audio constants (match F5-TTS base) ──
TARGET_SR = 24000
N_MEL = 100
HOP = 256
WIN = 1024
N_FFT = 1024
MEL_TYPE = "vocos"


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="F5-TTS Enhanced Training")

    g = p.add_argument_group("data")
    g.add_argument("--dataset_dir", required=True)
    g.add_argument("--embedding_dir", default=None)
    g.add_argument("--vocab_file", default=None)

    g = p.add_argument_group("pretrained")
    g.add_argument("--pretrain_ckpt", default=None)
    g.add_argument("--exp_name", default="F5TTS_Base",
                   choices=["F5TTS_v1_Base", "F5TTS_Base"])

    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=5)
    g.add_argument("--learning_rate", type=float, default=3e-4)
    g.add_argument("--batch_size_per_gpu", type=int, default=19200)
    g.add_argument("--batch_size_type", default="frame", choices=["frame", "sample"])
    g.add_argument("--bucket_batching", action="store_true")
    g.add_argument("--speaker_aware_batching", action="store_true")
    g.add_argument("--bucket_size", type=int, default=512)
    g.add_argument("--max_speakers_per_batch", type=int, default=8)
    g.add_argument("--max_samples_per_speaker", type=int, default=8)
    g.add_argument("--speaker_balanced_batching", action="store_true")
    g.add_argument("--speakers_per_batch", type=int, default=8)
    g.add_argument("--samples_per_speaker", type=int, default=4)
    g.add_argument("--max_samples", type=int, default=32)
    g.add_argument("--grad_accumulation_steps", type=int, default=1)
    g.add_argument("--max_grad_norm", type=float, default=1.0)
    g.add_argument("--num_warmup_updates", type=int, default=2000)
    g.add_argument("--lr_base_mult", type=float, default=0.1,
                   help="LR multiplier for unfrozen DiT blocks (× base LR). "
                        "Lower = safer for pretrained weights.")
    g.add_argument("--lr_prosody_mult", type=float, default=2.0,
                   help="LR multiplier for prosody modules (× base LR). "
                        "Higher = faster cold-start for new prosody pathway.")
    g.add_argument("--lr_duration_mult", type=float, default=1.0,
                   help="LR multiplier for duration predictor (× base LR).")

    g = p.add_argument_group("freeze strategy")
    g.add_argument("--freeze_base", action="store_true", default=True)
    g.add_argument("--no_freeze_base", dest="freeze_base", action="store_false")
    g.add_argument("--unfreeze_top_k", type=int, default=0)

    g = p.add_argument_group("conditioning")
    g.add_argument("--speaker_emb_dim", type=int, default=512)
    g.add_argument("--emotion_emb_dim", type=int, default=512)
    g.add_argument("--speaker_backend", default="wavlm_sv", choices=list(SPEAKER_RAW_DIMS))
    g.add_argument("--emotion_backend", default="emotion2vec_base", choices=list(EMOTION_RAW_DIMS))
    g.add_argument("--no_cross_attn", action="store_true")
    g.add_argument("--no_adaln", action="store_true")
    g.add_argument("--no_input_add", action="store_true")

    g = p.add_argument_group("prosody")
    g.add_argument("--prosody_backend", default="dio", choices=["dio", "harvest", "crepe", "rmvpe"])
    g.add_argument("--prosody_dim", type=int, default=256)
    g.add_argument("--no_prosody", action="store_true", help="Disable prosody conditioning")

    g = p.add_argument_group("duration predictor")
    g.add_argument("--use_duration_predictor", action="store_true", help="Train prosody-conditioned duration predictor")
    g.add_argument("--dur_loss_weight", type=float, default=0.1, help="Weight for duration prediction loss")

    g = p.add_argument_group("checkpoints")
    g.add_argument("--checkpoint_dir", default="ckpts/f5tts_enhanced")
    g.add_argument("--save_per_updates", type=int, default=5000)
    g.add_argument("--last_per_updates", type=int, default=1000)
    g.add_argument("--keep_last_n_checkpoints", type=int, default=5)

    g = p.add_argument_group("logging")
    g.add_argument("--logger", default=None, choices=[None, "wandb", "tensorboard"])
    g.add_argument("--wandb_project", default="f5tts-enhanced")
    g.add_argument("--wandb_run_name", default="enhanced_finetune")

    g = p.add_argument_group("misc")
    g.add_argument("--seed", type=int, default=666)
    g.add_argument("--num_workers", type=int, default=4)

    g = p.add_argument_group("validation")
    g.add_argument("--val_split", type=float, default=0.05,
                   help="Fraction of dataset for validation (0 to disable)")
    g.add_argument("--val_every", type=int, default=1000,
                   help="Run validation every N training updates")
    g.add_argument("--val_batches", type=int, default=0,
                   help="Max batches per validation (0 = all)")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
#  Model construction
# ══════════════════════════════════════════════════════════════════════

def download_pretrained(args) -> str:
    """Download or locate pretrained F5-TTS checkpoint."""
    if args.pretrain_ckpt and os.path.exists(args.pretrain_ckpt):
        return args.pretrain_ckpt

    from cached_path import cached_path

    url = (
        "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
        if args.exp_name == "F5TTS_v1_Base"
        else "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt"
    )
    ckpt = str(cached_path(url))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    dest = os.path.join(args.checkpoint_dir, f"pretrained_{os.path.basename(ckpt)}")
    if not os.path.exists(dest):
        shutil.copy2(ckpt, dest)
    return dest


def build_model(args, vocab_size: int) -> EnhancedCFM:
    """Build EnhancedCFM with conditioning modules."""
    cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    if args.exp_name != "F5TTS_v1_Base":
        cfg.update(text_mask_padding=False, pe_attn_head=1)

    cfg.update(
        speaker_emb_dim=args.speaker_emb_dim,
        emotion_emb_dim=args.emotion_emb_dim,
        use_adaln_cond=not args.no_adaln,
        use_input_add_cond=not args.no_input_add,
        use_cross_attn_cond=not args.no_cross_attn,
        speaker_raw_dim=SPEAKER_RAW_DIMS.get(args.speaker_backend, 512),
        emotion_raw_dim=EMOTION_RAW_DIMS.get(args.emotion_backend, 768),
        prosody_dim=args.prosody_dim,
        prosody_raw_dim=PROSODY_RAW_DIM if not args.no_prosody else None,
        use_prosody_cross_attn=not args.no_prosody,
    )

    return EnhancedCFM(
        transformer=EnhancedDiT(**cfg, text_num_embeds=vocab_size, mel_dim=N_MEL),
        mel_spec_kwargs=dict(
            n_fft=N_FFT, hop_length=HOP, win_length=WIN,
            n_mel_channels=N_MEL, target_sample_rate=TARGET_SR, mel_spec_type=MEL_TYPE,
        ),
        use_duration_predictor=getattr(args, "use_duration_predictor", False),
        speaker_emb_dim=args.speaker_emb_dim,
    )


def load_pretrained_weights(model: EnhancedCFM, ckpt_path: str):
    """Load original F5-TTS weights (strict=False)."""
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(ckpt_path, device="cpu")
    else:
        raw = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        if isinstance(raw, dict) and "ema_model_state_dict" in raw:
            state = {
                k.replace("ema_model.", ""): v
                for k, v in raw["ema_model_state_dict"].items()
                if k not in ("initted", "step")
            }
        elif isinstance(raw, dict) and "model_state_dict" in raw:
            state = raw["model_state_dict"]
        else:
            state = raw

    for k in ("mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"):
        state.pop(k, None)

    missing, unexpected = model.load_state_dict(state, strict=False)
    new = [k for k in missing if "cond_aggregator" in k]
    print(f"  Loaded {len(state) - len(unexpected)} base params | "
          f"New conditioning: {len(new)} | Skipped: {len(unexpected)}")
    del state; gc.collect(); torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════
#  Freeze strategy
# ══════════════════════════════════════════════════════════════════════

def apply_freeze(model: EnhancedCFM, freeze_base: bool, unfreeze_top_k: int):
    """Freeze base model weights; keep conditioning + duration predictor trainable."""
    if not freeze_base:
        return

    for p in model.parameters():
        p.requires_grad = False
    for p in model.transformer.cond_aggregator.parameters():
        p.requires_grad = True
    # Duration predictor is a separate module — must be explicitly unfrozen
    if model.duration_predictor is not None:
        for p in model.duration_predictor.parameters():
            p.requires_grad = True

    if unfreeze_top_k > 0:
        blocks = model.transformer.transformer_blocks
        for i in range(max(0, len(blocks) - unfreeze_top_k), len(blocks)):
            for p in blocks[i].parameters():
                p.requires_grad = True
        for p in model.transformer.norm_out.parameters():
            p.requires_grad = True
        for p in model.transformer.proj_out.parameters():
            p.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.1f}%)")


def build_param_groups(model: EnhancedCFM, args) -> list[dict]:
    """Build optimizer parameter groups with differential learning rates.

    Groups (from highest to lowest LR):
        prosody:      prosody-specific modules (cold start, needs fast LR)
        conditioning: cond_aggregator excluding prosody (speaker/emotion, near-converged)
        duration:     duration predictor (separate task)
        base:         unfrozen DiT blocks (pretrained, needs conservative LR)

    All LRs are multiples of args.learning_rate.
    """
    lr = args.learning_rate

    # Collect parameter id sets for each group
    prosody_ids = set()
    base_ids = set()
    duration_ids = set()

    agg = model.transformer.cond_aggregator

    # Prosody-specific parameters inside cond_aggregator
    prosody_module_names = [
        "prosody_raw_proj", "prosody_direct_proj", "prosody_temporal_smooth",
        "prosody_cross_attns", "prosody_global_proj", "emo_prosody_fusion",
    ]
    for name in prosody_module_names:
        mod = getattr(agg, name, None)
        if mod is not None:
            if isinstance(mod, nn.Parameter):
                prosody_ids.add(id(mod))
            else:
                for p in mod.parameters():
                    prosody_ids.add(id(p))
    # Scalar gates
    for attr in ("prosody_block_gates", "prosody_global_gate"):
        p = getattr(agg, attr, None)
        if p is not None and isinstance(p, nn.Parameter):
            prosody_ids.add(id(p))

    # Duration predictor
    if model.duration_predictor is not None:
        for p in model.duration_predictor.parameters():
            duration_ids.add(id(p))

    # Unfrozen DiT blocks (top-K + norm_out + proj_out)
    if args.freeze_base and args.unfreeze_top_k > 0:
        blocks = model.transformer.transformer_blocks
        for i in range(max(0, len(blocks) - args.unfreeze_top_k), len(blocks)):
            for p in blocks[i].parameters():
                if p.requires_grad:
                    base_ids.add(id(p))
        for mod in (model.transformer.norm_out, model.transformer.proj_out):
            for p in mod.parameters():
                if p.requires_grad:
                    base_ids.add(id(p))

    # Build groups — each trainable param in exactly one group
    prosody_params, cond_params, dur_params, base_params = [], [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in prosody_ids:
            prosody_params.append(p)
        elif pid in duration_ids:
            dur_params.append(p)
        elif pid in base_ids:
            base_params.append(p)
        else:
            cond_params.append(p)  # everything else (speaker/emotion conditioning, mel_spec if trainable)

    groups = []
    if cond_params:
        groups.append({"params": cond_params, "lr": lr, "name": "conditioning"})
    if prosody_params:
        groups.append({"params": prosody_params, "lr": lr * args.lr_prosody_mult, "name": "prosody"})
    if dur_params:
        groups.append({"params": dur_params, "lr": lr * args.lr_duration_mult, "name": "duration"})
    if base_params:
        groups.append({"params": base_params, "lr": lr * args.lr_base_mult, "name": "base_blocks"})

    # Summary
    for g in groups:
        n = sum(p.numel() for p in g["params"])
        print(f"  Param group '{g['name']}': {n:,} params, lr={g['lr']:.2e}")

    return groups


# ══════════════════════════════════════════════════════════════════════
#  Encoder loading (online extraction fallback)
# ══════════════════════════════════════════════════════════════════════

def maybe_load_encoders(args, dev: str):
    """Load encoders only if no precomputed embeddings exist."""
    if args.embedding_dir and os.path.exists(os.path.join(args.embedding_dir, "0.pt")):
        print("Using precomputed embeddings — encoders NOT loaded")
        return None, None, None

    print("Loading encoders for online extraction (slower; run prepare_data.py for speed)")
    spk = SpeakerEncoder(backend=args.speaker_backend, output_dim=args.speaker_emb_dim,
                         device=dev).to(dev).eval()
    emo = EmotionEncoder(backend=args.emotion_backend, output_dim=args.emotion_emb_dim,
                         frame_level=True, device=dev).to(dev).eval()
    for enc in (spk, emo):
        for p in enc.parameters():
            p.requires_grad = False

    prosody_enc = None
    if not args.no_prosody:
        prosody_enc = ProsodyEncoder(
            backend=args.prosody_backend, output_dim=args.prosody_dim, device=dev,
        ).to(dev).eval()
        for p in prosody_enc.parameters():
            p.requires_grad = False

    return spk, emo, prosody_enc


# ══════════════════════════════════════════════════════════════════════
#  Dataset & DataLoader
# ══════════════════════════════════════════════════════════════════════

def build_dataloaders(args):
    from datasets import Dataset as HFD, load_from_disk

    raw_path = os.path.join(args.dataset_dir, "raw")
    try:
        ds = load_from_disk(raw_path)
    except Exception:
        ds = HFD.from_file(os.path.join(args.dataset_dir, "raw.arrow"))

    with open(os.path.join(args.dataset_dir, "duration.json")) as f:
        durations = json.load(f)["duration"]
    
    
    
    # ── Train / Val split ──
    n = len(ds)
    val_loader = None
    if args.val_split > 0 and n > 20:
        n_val = max(int(n * args.val_split), 1)
        n_train = n - n_val

        # Deterministic split by index (not random — keeps embedding cache aligned)
        rng = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(n, generator=rng).tolist()
        train_idx = sorted(indices[:n_train])
        val_idx = sorted(indices[n_train:])

        train_ds = ds.select(train_idx)
        val_ds = ds.select(val_idx)
        train_durs = [durations[i] for i in train_idx]
        val_durs = [durations[i] for i in val_idx]

        # Remap embedding cache indices:
        # original index i → cache file "{i}.pt"
        # We pass the original indices so dataset can load correct cache files
        train_emb_map = train_idx  # original indices for cache lookup
        val_emb_map = val_idx
    else:
        train_ds, train_durs, train_emb_map = ds, durations, None
        val_ds = None

    def _make_dataset(hf_ds, durs, emb_index_map=None):
        return EnhancedDataset(
            hf_dataset=hf_ds, durations=durs,
            embedding_cache_dir=args.embedding_dir,
            embedding_index_map=emb_index_map,
            target_sample_rate=TARGET_SR, hop_length=HOP,
            n_mel_channels=N_MEL, n_fft=N_FFT, win_length=WIN, mel_spec_type=MEL_TYPE,
        )

    train_dataset = _make_dataset(train_ds, train_durs, train_emb_map)

    if args.batch_size_type == "frame":
        sampler = SequentialSampler(train_dataset)
        
        if args.speaker_balanced_batching:
            bs = SpeakerBalancedDynamicBatchSampler(
                sampler,
                args.batch_size_per_gpu,
                speakers_per_batch=args.speakers_per_batch,
                samples_per_speaker=args.samples_per_speaker,
                max_samples=args.max_samples,
                random_seed=args.seed,
                drop_residual=False,
            )
        elif args.speaker_aware_batching:
            bs = SpeakerAwareBucketDynamicBatchSampler(
                sampler,
                args.batch_size_per_gpu,
                max_samples=args.max_samples,
                bucket_size=args.bucket_size,
                max_speakers_per_batch=args.max_speakers_per_batch,
                max_samples_per_speaker=args.max_samples_per_speaker,
                random_seed=args.seed,
                drop_residual=False,
            )
        elif args.bucket_batching:
            bs = BucketDynamicBatchSampler(
                sampler,
                args.batch_size_per_gpu,
                max_samples=args.max_samples,
                bucket_size=args.bucket_size,
                random_seed=args.seed,
                drop_residual=False,
            )
        else:
            bs = DynamicBatchSampler(
                sampler,
                args.batch_size_per_gpu,
                max_samples=args.max_samples,
                random_seed=args.seed,
                drop_residual=False,
            )
        
        #bs = DynamicBatchSampler(sampler, args.batch_size_per_gpu, max_samples=args.max_samples, random_seed=args.seed)
        train_loader = DataLoader(train_dataset, collate_fn=collate_fn,
                                  num_workers=args.num_workers, pin_memory=True,
                                  persistent_workers=args.num_workers > 0,
                                  batch_sampler=bs)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size_per_gpu,
                                  collate_fn=collate_fn, num_workers=args.num_workers,
                                  pin_memory=True, shuffle=True)

    if val_ds is not None:
        val_dataset = _make_dataset(val_ds, val_durs, val_emb_map)
        if args.batch_size_type == "frame":
            val_sampler = SequentialSampler(val_dataset)
            val_bs = DynamicBatchSampler(val_sampler, args.batch_size_per_gpu,
                                         max_samples=args.max_samples, random_seed=args.seed)
            val_loader = DataLoader(val_dataset, collate_fn=collate_fn,
                                    num_workers=min(args.num_workers, 2), pin_memory=True,
                                    batch_sampler=val_bs)
        else:
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size_per_gpu,
                                    collate_fn=collate_fn,
                                    num_workers=min(args.num_workers, 2), pin_memory=True)

    n_train = len(train_dataset)
    n_val = len(val_loader.dataset) if val_loader else 0
    print(f"Dataset: {n_train} train, {n_val} val")

    return train_dataset, train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════
#  Validation
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, val_loader, accelerator, spk_enc, emo_enc, max_batches=0, prosody_enc=None):
    """
    Deterministic validation pass — returns average loss comparable across runs.

    Fixes ALL sources of randomness from EnhancedCFM.forward():
      - timestep:      fixed t=0.5 (midpoint — hardest point for flow matching)
      - noise x0:      seeded per-batch via isolated RNG (does NOT affect training)
      - mask fraction:  fixed frac=0.85 (85% of frames masked)
      - mask position:  seeded per-batch via isolated RNG
      - dropout mode:   disabled (full conditioning, no drops)

    Multi-GPU: val_loader is sharded by Accelerate — each GPU sees 1/N of the
    data.  We all_reduce (total_loss, total_frames) before averaging so every
    rank returns the same globally-correct loss.
    """
    model.eval()
    dev = accelerator.device
    unwrapped = accelerator.unwrap_model(model)

    total_loss = 0.0
    total_frames = 0
    n_batches = 0

    FIXED_TIME = 0.5
    FIXED_FRAC = 0.85

    # Save global RNG state — restored after validation so training is unaffected.
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if dev.type == "cuda" else None

    for batch in val_loader:
        mel = batch["mel"].permute(0, 2, 1)
        text = convert_char_to_pinyin(batch["text"])
        lens = batch["mel_lengths"].long()
        condition_lifecycle = getattr(unwrapped, "condition_lifecycle", None)
        conditions = (
            condition_lifecycle.prepare(
                batch=batch,
                target_len=mel.shape[1],
                device=dev,
                speaker_encoder=spk_enc,
                emotion_encoder=emo_enc,
                prosody_encoder=prosody_enc,
            )
            if condition_lifecycle is not None else None
        )

        # ── Deterministic forward (mirrors EnhancedCFM.forward) ──
        inp = unwrapped._to_mel(mel)
        B, seq_len, dtype = inp.shape[0], inp.shape[1], inp.dtype
        text_ids = unwrapped._to_text_ids(text, B, dev)

        mask = lens_to_mask(lens, length=seq_len)

        # Seed per batch index → same mask positions every run
        torch.manual_seed(n_batches)
        if dev.type == "cuda":
            torch.cuda.manual_seed(n_batches)
        frac_lengths = torch.full((B,), FIXED_FRAC, device=dev)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)
        if exists(mask):
            rand_span_mask &= mask

        # Same seed offset → same noise every run
        torch.manual_seed(n_batches + 100_000)
        if dev.type == "cuda":
            torch.cuda.manual_seed(n_batches + 100_000)
        x0 = torch.randn_like(inp)

        time = torch.full((B,), FIXED_TIME, dtype=dtype, device=dev)
        t = time.unsqueeze(-1).unsqueeze(-1)

        phi = (1 - t) * x0 + t * inp
        flow = inp - x0
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(inp), inp)

        model_conditions = conditions

        val_runtime = None
        if condition_lifecycle is not None and model_conditions is not None:
            val_runtime = condition_lifecycle.build_runtime(model_conditions)

        pred = unwrapped.transformer(
            x=phi, cond=cond, text=text_ids, time=time, mask=mask,
            drop_audio_cond=False, drop_text=False,
            conditioning_runtime=val_runtime,
        )

        loss = F.mse_loss(pred, flow, reduction="none")
        masked = loss[rand_span_mask]
        if masked.numel() == 0:
            n_batches += 1
            continue
        batch_loss = masked.mean()

        # Skip NaN batches (corrupted weights shouldn't poison val metric)
        if not torch.isfinite(batch_loss):
            n_batches += 1
            continue

        batch_frames = rand_span_mask.sum().item()
        total_loss += batch_loss.item() * batch_frames
        total_frames += batch_frames
        n_batches += 1

        if max_batches > 0 and n_batches >= max_batches:
            break

    # ── Aggregate across GPUs ──
    # Each rank saw 1/N of the val set.  Sum weighted losses and frame counts
    # so every rank computes the same global average.
    if accelerator.num_processes > 1:
        stats = torch.tensor([total_loss, total_frames], device=dev, dtype=torch.float64)
        stats = accelerator.reduce(stats, reduction="sum")
        total_loss = stats[0].item()
        total_frames = int(stats[1].item())
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng)

    model.train()

    if total_frames == 0:
        return float("nan")

    return total_loss / total_frames


# ══════════════════════════════════════════════════════════════════════
#  Checkpoint management
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: torch.nn.Module,
    ema_model: object | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    epoch: int,
    path: str,
    is_last: bool = False,
):
    """Save a resumable checkpoint (online model + EMA + optimizer + scheduler)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }

    if ema_model is not None:
        payload["ema_model_state_dict"] = ema_model.state_dict()

    torch.save(payload, path)


def clean_old_checkpoints(ckpt_dir: str, keep: int):
    """Remove old checkpoints robustly.

    Keeps:
      - model_last.pt always
      - newest `keep` checkpoints among files that look like model checkpoints

    Robustness:
      - supports names like model_123.pt, model_step_123.pt, model-123.pt, etc.
      - ignores files it can't parse as a checkpoint (falls back to mtime ordering)
    """
    if keep <= 0:
        return
    if not os.path.isdir(ckpt_dir):
        return

    files = []
    for name in os.listdir(ckpt_dir):
        if name in ("model_last.pt", "model_best.pt"):
            continue
        if name.startswith("pretrained_"):
            continue
        if not name.endswith(".pt"):
            continue
        if "model" not in name:
            continue

        path = os.path.join(ckpt_dir, name)

        step = None
        m = re.search(r"(\d+)", name)
        if m:
            try:
                step = int(m.group(1))
            except Exception:
                step = None

        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0

        files.append((name, step, mtime))

    def sort_key(item):
        name, step, mtime = item
        has_step = 0 if step is not None else 1
        step_val = step if step is not None else -1
        return (has_step, step_val, mtime)

    files.sort(key=sort_key)

    while len(files) > keep:
        name, _, _ = files.pop(0)
        try:
            os.remove(os.path.join(ckpt_dir, name))
        except FileNotFoundError:
            pass

# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    writer = None
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
   
    accelerator = Accelerator(
        log_with=args.logger if args.logger == "wandb" else None,
        kwargs_handlers=[ddp],
        gradient_accumulation_steps=args.grad_accumulation_steps,
    )
    dev = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        print("=" * 60)
        print("  F5-TTS Enhanced — Training")
        print("=" * 60)

    # 1. Tokenizer
    vocab_path = (
        args.vocab_file
        or (lambda p: p if os.path.exists(p) else str(files("f5_tts").joinpath("infer/examples/vocab.txt")))(
            os.path.join(args.dataset_dir, "vocab.txt")
        )
    )
    vocab_map, vocab_size = get_tokenizer(vocab_path, "custom")
    if is_main:
        print(f"Vocab: {vocab_size} tokens")

    # 2. Model
    model = build_model(args, vocab_size)
    model.vocab_char_map = vocab_map

    # 3. Pretrained weights
    ckpt_path = download_pretrained(args)
    load_pretrained_weights(model, ckpt_path)

    # 4. Freeze
    if is_main:
        print(f"Freeze: base={args.freeze_base}, unfreeze_top_k={args.unfreeze_top_k}")
    apply_freeze(model, args.freeze_base, args.unfreeze_top_k)

    # 5. EMA (main process)
    ema = EMA(model, include_online_model=False) if is_main else None
    if ema:
        ema.to(dev)

    # 6. Optimizer with differential learning rates
    param_groups = build_param_groups(model, args)
    optimizer = AdamW(param_groups, lr=args.learning_rate)  # base lr as fallback
    trainable = [p for g in param_groups for p in g["params"]]

    # 7. Data
    dataset, dataloader, val_loader = build_dataloaders(args)

    # 8. Encoders
    spk_enc, emo_enc, prosody_enc = maybe_load_encoders(args, str(dev))

    # 9. Scheduler
    warmup = args.num_warmup_updates * accelerator.num_processes
    total = math.ceil(len(dataloader) / args.grad_accumulation_steps) * args.epochs
    decay = max(total - warmup, 1)
    scheduler = SequentialLR(optimizer, [
        LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup),
        LinearLR(optimizer, start_factor=1.0, end_factor=1e-2, total_iters=decay),
    ], milestones=[warmup])

    # 10. Prepare
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler,
    )
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    # 11. Logger
    if args.logger == "wandb" and is_main:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.wandb_run_name}},
        )
    elif args.logger == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter

        writer = None
        if is_main:
            writer = SummaryWriter(log_dir=f"{args.checkpoint_dir}")

    # 12. Resume
    step = 0
    start_epoch = 0
    last_ckpt = os.path.join(args.checkpoint_dir, "model_last.pt")
    if os.path.exists(last_ckpt):
        ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=True)
        # Restore online model weights
        
        model_sd = ckpt.get("model_state_dict")
        if model_sd is not None and len(model_sd) > 0:
            accelerator.unwrap_model(model).load_state_dict(model_sd, strict=False)
            if is_main:
                print(f"  Restored {len(model_sd)} model params")
        else:
            if is_main:
                print("  ⚠️ No model_state_dict in checkpoint — model weights NOT restored."
                      " Training continues from pretrained weights.")
        
        if is_main and ema and "ema_model_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (ValueError, KeyError) as e:
                if is_main:
                    print(f"  ⚠ Optimizer state mismatch (param groups changed?): {e}")
                    print("    Optimizer reset — LR restarts from warmup.")
        if "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except (ValueError, KeyError) as e:
                if is_main:
                    print(f"  ⚠ Scheduler state mismatch: {e}")
        step = ckpt.get("step", ckpt.get("update", 0))
        start_epoch = ckpt.get("epoch", 0)
        if is_main:
            print(f"Resumed at step {step}, epoch {start_epoch}")
        del ckpt

    # ── Training loop ─────────────────────────────────────────────
    if is_main:
        print(f"\n{'='*60}\n  {args.epochs} epochs | ~{total} updates | "
              f"warmup {args.num_warmup_updates} | {accelerator.num_processes} GPU(s)")
        if val_loader is not None:
            print(f"  Validation every {args.val_every} steps"
                  f"{f' ({args.val_batches} batches)' if args.val_batches else ''}")
        print(f"{'='*60}\n")

    best_val_loss = float("inf")
    raw_model = accelerator.unwrap_model(model)  # unwrapped for checkpoint saving
    
    mode_loss_sum = defaultdict(float)
    mode_count = defaultdict(int)
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if hasattr(dataloader, "batch_sampler") and hasattr(dataloader.batch_sampler, "set_epoch"):
            dataloader.batch_sampler.set_epoch(epoch)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}",
                    disable=not is_main, unit="batch")
        epoch_loss, epoch_steps = 0.0, 0
        nan_consecutive = 0

        for batch in pbar:
            with accelerator.accumulate(model):
                mel = batch["mel"].permute(0, 2, 1)       # (B, D, T) → (B, T, D)
                text = convert_char_to_pinyin(batch["text"])
                lens = batch["mel_lengths"].long()  # arange requires integer dtype

                condition_lifecycle = getattr(raw_model, "condition_lifecycle", None)
                conditions = (
                    condition_lifecycle.prepare_for_train(
                        batch,
                        target_len=mel.shape[1],
                        device=dev,
                        speaker_encoder=spk_enc,
                        emotion_encoder=emo_enc,
                        prosody_encoder=prosody_enc,
                    )
                    if condition_lifecycle is not None else None
                )

                loss, _, _, mode_names, dur_loss = model(
                    inp=mel, text=text, lens=lens,
                    conditions=conditions,
                    # Duration predictor inputs
                    prosody_raw=batch.get("prosody_raw"),
                    prosody_mask=batch.get("prosody_mask"),
                    text_byte_lens=torch.tensor(
                        [len(t.encode("utf-8")) for t in batch["text"]],
                        dtype=torch.long,
                    ) if getattr(raw_model, "duration_predictor", None) is not None else None,
                    speaker_emb_for_dur=batch.get("speaker_raw"),
                )

                # Combined loss (dur_loss is 0.0 when predictor disabled — free to add)
                total_loss = loss + args.dur_loss_weight * dur_loss
                
                
                # ── NaN/Inf guard: skip corrupted batches ──
                if not torch.isfinite(total_loss):
                    optimizer.zero_grad()
                    nan_consecutive += 1
                    if is_main:
                        tqdm.write(f"  ⚠ step ~{step}: NaN/Inf loss, skipping batch "
                                   f"({nan_consecutive} consecutive)")
                    if nan_consecutive >= 5:
                        if is_main:
                            tqdm.write("  ✖ 5 consecutive NaN batches — model weights likely "
                                       "corrupted. Stopping epoch.")
                        break
                    continue
                
                # Per-mode loss tracking: distribute batch loss across all modes present.
                # This is approximate (batch loss ≠ per-sample loss), but avoids
                # the previous bug of attributing the entire loss to the most common mode.
                loss_val = float(loss.detach().item())
                for m in mode_names:
                    mode_loss_sum[m] += loss_val
                    mode_count[m] += 1
                
                nan_consecutive = 0
                accelerator.backward(total_loss)

                # Only step optimizer on sync steps (i.e., after gradient accumulation)
                if accelerator.sync_gradients:
                    if args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                if is_main and ema:
                    ema.update()
                step += 1
                epoch_loss += loss.item()
                epoch_steps += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 avg=f"{epoch_loss/epoch_steps:.4f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}", s=step)
                
                if is_main and args.logger:
                    accelerator.log({"train/loss": loss.item(),
                                     "train/lr": scheduler.get_last_lr()[0]}, step=step)

                if args.logger == "tensorboard" and accelerator.is_main_process:
                    writer.add_scalar("train_loss", loss.item(), step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                
                if step % 100 == 0 and any(mode_count.values()):
                
                    for m in ("full", "uncond", "voice_only", "no_emotion", "no_prosody", "no_speaker", "textless"):
                        ls = torch.tensor(mode_loss_sum.get(m, 0.0), device=accelerator.device, dtype=torch.float32)
                        ct = torch.tensor(mode_count.get(m, 0), device=accelerator.device, dtype=torch.float32)

                        ls = accelerator.reduce(ls, reduction="sum")
                        ct = accelerator.reduce(ct, reduction="sum")

                        if is_main and ct.item() > 0:
                            accelerator.log({f"train/loss_{m}": (ls / ct).item()}, step=step)
                        
                            if args.logger == "tensorboard":
                                writer.add_scalar(f"train/loss_{m}", (ls / ct).item(), step)
                    
                    mode_loss_sum.clear()
                    mode_count.clear()
                
            if step % args.last_per_updates == 0 and accelerator.sync_gradients:
                accelerator.wait_for_everyone()
                if is_main:
                    save_checkpoint(raw_model, ema, optimizer, scheduler, step, epoch,
                                    os.path.join(args.checkpoint_dir, "model_last.pt"))

            if step % args.save_per_updates == 0 and accelerator.sync_gradients:
                accelerator.wait_for_everyone()
                if is_main:
                    save_checkpoint(raw_model, ema, optimizer, scheduler, step, epoch,
                                    os.path.join(args.checkpoint_dir, f"model_{step}.pt"))
                    clean_old_checkpoints(args.checkpoint_dir, args.keep_last_n_checkpoints)

            # ── Validation ──
            if (val_loader is not None
                    and step > 0
                    and step % args.val_every == 0
                    and accelerator.sync_gradients):
                val_loss = validate(model, val_loader, accelerator,
                                    spk_enc, emo_enc, args.val_batches, prosody_enc=prosody_enc)

                if is_main:
                    improved = val_loss < best_val_loss
                    marker = " ★ best" if improved else ""
                    print(f"\n  ⟐ Val step {step}: loss={val_loss:.5f}{marker}")

                    if args.logger:
                        accelerator.log({"val/loss": val_loss}, step=step)

                    if args.logger == "tensorboard" and accelerator.is_main_process:
                        writer.add_scalar("val_loss", val_loss, step)

                    if improved:
                        best_val_loss = val_loss
                        save_checkpoint(raw_model, ema, optimizer, scheduler, step, epoch,
                                        os.path.join(args.checkpoint_dir, "model_best.pt"))

        # ── End-of-epoch validation ──
        if val_loader is not None:
            val_loss = validate(model, val_loader, accelerator,
                                spk_enc, emo_enc, args.val_batches, prosody_enc=prosody_enc)
            if is_main:
                improved = val_loss < best_val_loss
                marker = " ★ best" if improved else ""
                print(f"  Epoch {epoch+1} — train: {epoch_loss/max(epoch_steps,1):.4f}"
                      f" | val: {val_loss:.5f}{marker}")
                if args.logger:
                    accelerator.log({"val/loss": val_loss,
                                     "val/best_loss": best_val_loss}, step=step)



                if improved:
                    best_val_loss = val_loss
                    save_checkpoint(raw_model, ema, optimizer, scheduler, step, epoch,
                                    os.path.join(args.checkpoint_dir, "model_best.pt"))
        elif is_main:
            print(f"  Epoch {epoch+1} done — avg loss: {epoch_loss/max(epoch_steps,1):.4f}")

    # Final save
    accelerator.wait_for_everyone()
    if is_main:
        save_checkpoint(raw_model, ema, optimizer, scheduler, step, args.epochs,
                        os.path.join(args.checkpoint_dir, "model_last.pt"))
        print(f"\nDone! {step} updates → {args.checkpoint_dir}/model_last.pt")
        if val_loader is not None and best_val_loss < float("inf"):
            print(f"Best val loss: {best_val_loss:.5f} → {args.checkpoint_dir}/model_best.pt")
    accelerator.end_training()
    if writer and hasattr(writer, 'close'):
        writer.close()


if __name__ == "__main__":
    main()
