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
    g.add_argument("--rebuild_every_n_steps", type=int, default=0,
                   help="Re-shuffle remaining batches every N steps within an epoch. "
                        "0 = only rebuild at epoch boundaries (default). "
                        "E.g. 500 = reshuffle every 500 steps.")
    g.add_argument("--speaker_balanced_batching", action="store_true")
    g.add_argument("--speakers_per_batch", type=int, default=8)
    g.add_argument("--samples_per_speaker", type=int, default=4)
    g.add_argument("--max_samples", type=int, default=32)
    g.add_argument("--grad_accumulation_steps", type=int, default=1)
    g.add_argument("--max_grad_norm", type=float, default=1.0)
    g.add_argument("--num_warmup_updates", type=int, default=2000)
    g.add_argument("--num_warmup_updates_cross_attn", type=int, default=0,
                   help="Separate longer warmup for cross-attention group. "
                        "0 = use --num_warmup_updates. "
                        "Recommended: 3×-5× num_warmup_updates when starting cross-attn cold.")
    g.add_argument("--scheduler", default="cosine",
                   choices=["cosine", "linear"],
                   help="LR decay schedule after warmup. cosine recommended.")
    g.add_argument("--weight_decay", type=float, default=0.1,
                   help="AdamW weight decay. 0.1 is standard for transformers.")
    g.add_argument("--adam_betas", type=float, nargs=2, default=[0.9, 0.98],
                   metavar=("BETA1", "BETA2"),
                   help="AdamW betas. (0.9, 0.98) more stable than default (0.9, 0.999) for speech.")
    g.add_argument("--adam_eps", type=float, default=1e-8,
                   help="AdamW epsilon.")
    g.add_argument("--speed_perturb", action="store_true",
                   help="Randomly perturb reference audio speed (0.9-1.1x) during training. "
                        "Improves generalisation to different speaking rates.")
    g.add_argument("--ema_decay", type=float, default=0.9999,
                   help="EMA decay rate. 0.9999 for large datasets/long runs; "
                        "0.999 for smaller datasets or short staged curriculum.")
    g.add_argument("--ema_update_after", type=int, default=100,
                   help="Start EMA updates after this many steps (skip noisy early steps).")
    g.add_argument("--ema_update_every", type=int, default=1,
                   help="Update EMA every N steps.")
    g.add_argument("--lr_base_mult", type=float, default=0.1,
                   help="LR multiplier for unfrozen DiT blocks (× base LR).")
    g.add_argument("--lr_speaker_mult", type=float, default=1.0,
                   help="LR multiplier for speaker/global conditioning: "
                        "adaln_cond, fusion, speaker_raw_proj, emotion_raw_proj, input_add.")
    g.add_argument("--lr_prosody_mult", type=float, default=2.0,
                   help="LR multiplier for prosody direct/global modules (× base LR).")
    g.add_argument("--lr_cross_attn_mult", type=float, default=2.0,
                   help="LR multiplier for cross-attention modules (emotion + prosody "
                        "frame-level paths). Higher = faster learning for the slowest pathway.")
    g.add_argument("--lr_duration_mult", type=float, default=1.0,
                   help="LR multiplier for duration predictor (× base LR).")

    g = p.add_argument_group("freeze strategy")
    g.add_argument("--freeze_base", action="store_true", default=True)
    g.add_argument("--no_freeze_base", dest="freeze_base", action="store_false")
    g.add_argument("--unfreeze_top_k", type=int, default=0)
    g.add_argument(
        "--train_stage", type=int, default=0,
        help="""Curriculum training stage. Each stage adds NEW modules on top of previous.
  0 = all conditioning (default, no curriculum)
  1 = Speaker+Global  speaker_raw_proj + emotion_raw_proj + fusion + adaln + input_add + duration
  2 = Prosody direct  prosody_direct_proj + temporal_smooth + global_proj + gates
  3 = Prosody attn    prosody_cross_attns + prosody_raw_proj
  4 = Emotion attn    frame_raw_proj + cross_attns
  5 = Fusion          emo_prosody_fusion
  6 = DiT blocks      top-K transformer blocks + norm_out + proj_out
Recommended flow: 1 → 2 → 3 → 4 → 5 → (6) → 0 (joint finetune, low LR).
        """
    )
    g.add_argument(
        "--freeze_adaln", action="store_true", default=False,
        help="On stages 2-6: freeze AdaLN (keep only the new module trainable). "
             "Use only after stage 1 has fully converged. "
             "Default: AdaLN stays active on stages 2-6 to stabilise DiT blocks."
    )

    g = p.add_argument_group("conditioning")
    g.add_argument("--speaker_emb_dim", type=int, default=512)
    g.add_argument("--emotion_emb_dim", type=int, default=512)
    g.add_argument("--speaker_backend", default="wavlm_sv", choices=list(SPEAKER_RAW_DIMS))
    g.add_argument("--emotion_backend", default="emotion2vec_base", choices=list(EMOTION_RAW_DIMS))
    g.add_argument("--no_cross_attn", action="store_true")
    g.add_argument("--no_adaln", action="store_true")
    g.add_argument("--adaln_bottleneck_dim", type=int, default=256,
                   help="AdaLN bottleneck width. fused_global is compressed to this "
                        "dim before producing scale/shift in each block. 256 (default) "
                        "saves params but can blur fine speaker detail (4:1 compression "
                        "of 1024). Try 512 for better voice fidelity at +~25M params. "
                        "NOTE: changing this changes the architecture → retrain required.")
    g.add_argument("--no_input_add", action="store_true")
    g.add_argument("--speaker_emb_dropout", type=float, default=0.0,
                   help="Dropout on raw speaker embedding (train only). "
                        "Anti-overfitting for few-speaker datasets. Try 0.1-0.2 for 5-10 speakers.")
    g.add_argument("--speaker_emb_noise", type=float, default=0.0,
                   help="Gaussian noise std added to speaker embedding (train only). "
                        "Forces generalisation to unseen voices. Try 0.05-0.1 for few speakers.")

    g = p.add_argument_group("prosody")
    g.add_argument("--prosody_backend", default="dio", choices=["dio", "harvest", "crepe", "rmvpe"])
    g.add_argument("--prosody_dim", type=int, default=256)
    g.add_argument("--fusion_mode", default="concat", choices=["concat", "residual"],
                   help="How global speaker+emotion combine into fused_global. "
                        "'concat' (default): shared Linear([speaker|emotion]) — emotion "
                        "can dominate and zero out speaker → voice stops cloning. "
                        "'residual': speaker has its own path (always reaches output), "
                        "emotion is a gated correction — speaker can't be zeroed. "
                        "NOTE: changes architecture → retrain required.")
    g.add_argument("--cross_attn_gate_floor", type=float, default=0.0,
                   help="Floor on the EFFECTIVE cross-attn gate to prevent the death spiral (gate→0 → to_out frozen → path dies). Keeps the path alive with live gradient "
                        "without freezing the gate parameter (unlike clamp). Try 0.05 for small data where cross-attn tends to die. 0.0 = off (original).")
    g.add_argument("--no_prosody", action="store_true", help="Disable prosody conditioning")
    g.add_argument("--no_prosody_direct", action="store_true",
                   help="Disable ONLY the prosody direct path, keeping prosody cross-attention. "
                        "Removes direct↔cross-attn competition on the same layers. "
                        "Recommended with enough data (15h+) so cross-attn can carry prosody.")
    g.add_argument("--no_prosody_cross_attn", action="store_true",
                   help="Disable ONLY prosody cross-attention, keeping the direct path. "
                        "Recommended for small datasets (<10h) where cross-attn would die.")

    g = p.add_argument_group("duration predictor")
    g.add_argument("--use_duration_predictor", action="store_true", help="Train prosody-conditioned duration predictor")
    g.add_argument("--duration_speaker_dropout", type=float, default=0.0,
                   help="Dropout on the speaker vector inside the duration predictor. "
                        "Speaker is ~98%% of the predictor input, so it tends to memorise "
                        "per-speaker rates and over-predict duration on unseen/cross-language "
                        "speakers (→ padding/leakage). Try 0.3-0.5.")
    g.add_argument("--duration_speaker_bottleneck", type=int, default=0,
                   help="Compress the speaker vector to this width before the duration "
                        "predictor (e.g. 32) to limit per-speaker memorisation. 0 = off.")
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
        adaln_bottleneck_dim=args.adaln_bottleneck_dim,
        use_input_add_cond=not args.no_input_add,
        use_cross_attn_cond=not args.no_cross_attn,
        speaker_raw_dim=SPEAKER_RAW_DIMS.get(args.speaker_backend, 512),
        emotion_raw_dim=EMOTION_RAW_DIMS.get(args.emotion_backend, 768),
        prosody_dim=args.prosody_dim,
        prosody_raw_dim=PROSODY_RAW_DIM if not args.no_prosody else None,
        use_prosody_cross_attn=(not args.no_prosody) and (not args.no_prosody_cross_attn),
        use_prosody_direct=(not args.no_prosody) and (not args.no_prosody_direct),
        fusion_mode=args.fusion_mode,
        cross_attn_gate_floor=args.cross_attn_gate_floor,
        speaker_emb_dropout=args.speaker_emb_dropout,
        speaker_emb_noise=args.speaker_emb_noise,
    )

    return EnhancedCFM(
        transformer=EnhancedDiT(**cfg, text_num_embeds=vocab_size, mel_dim=N_MEL),
        mel_spec_kwargs=dict(
            n_fft=N_FFT, hop_length=HOP, win_length=WIN,
            n_mel_channels=N_MEL, target_sample_rate=TARGET_SR, mel_spec_type=MEL_TYPE,
        ),
        use_duration_predictor=getattr(args, "use_duration_predictor", False),
        duration_speaker_dropout=getattr(args, "duration_speaker_dropout", 0.0),
        duration_speaker_bottleneck=getattr(args, "duration_speaker_bottleneck", 0),
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

def zero_future_stage_gates(model, stage: int):
    """Zero the gates of gated-residual modules belonging to stages AFTER `stage`.

    Must be re-applied after loading a resume checkpoint: apply_curriculum_stage
    sets these gate values, but a subsequent load_state_dict(model_last.pt)
    overwrites them with the checkpoint's values (which are the non-zero init
    0.02/0.15 for never-trained future modules). Without re-zeroing, untrained
    future modules would inject noise into the residual stream. requires_grad is
    untouched (load_state_dict doesn't change it), so only values need fixing.
    """
    if stage <= 0:
        return
    agg = model.transformer.cond_aggregator

    def _zero_gates_in(mod, value=0.0):
        if mod is None:
            return
        items = list(mod.values()) if hasattr(mod, "values") else [mod]
        for m in items:
            if hasattr(m, "gate") and isinstance(m.gate, torch.nn.Parameter):
                with torch.no_grad():
                    m.gate.fill_(value)
            elif isinstance(m, torch.nn.Module):
                # Gate may live on nested submodules (e.g. EmoProsodyFusion holds
                # two _FusionCrossAttn each with its own gate). Recurse to find
                # every gate so the whole module is silenced.
                for sub in m.modules():
                    if sub is m:
                        continue
                    if hasattr(sub, "gate") and isinstance(sub.gate, torch.nn.Parameter):
                        with torch.no_grad():
                            sub.gate.fill_(value)

    def _zero_param(name, value=0.0):
        p = getattr(agg, name, None)
        if p is not None and isinstance(p, torch.nn.Parameter):
            with torch.no_grad():
                p.fill_(value)

    if stage < 2:
        _zero_param("prosody_block_gates", 0.0)
        _zero_param("prosody_global_gate", 0.0)
    if stage < 3:
        _zero_gates_in(getattr(agg, "prosody_cross_attns", None), 0.0)
    if stage < 4:
        _zero_gates_in(getattr(agg, "cross_attns", None), 0.0)
    if stage < 5:
        _zero_gates_in(getattr(agg, "emo_prosody_fusion", None), 0.0)


def apply_curriculum_stage(model, stage: int,
                           unfreeze_top_k: int = 0,
                           freeze_adaln: bool = False):
    """Curriculum: speaker → prosody-direct → prosody-attn → emotion → fusion → joint.

    Each stage unfreezes NEW modules; previously trained modules stay active.
    Gates of untrained gated residuals are zeroed → no noise injection.
    AdaLN stays active on stages 2-5 (stabilises DiT); disable with --freeze_adaln.

    Stage 1: Speaker+Global  speaker_raw_proj + emotion_raw_proj + fusion +
                             adaln_cond + input_add + duration_predictor
    Stage 2: Prosody direct  prosody_direct_proj + prosody_temporal_smooth +
                             prosody_global_proj + prosody gates
    Stage 3: Prosody attn    prosody_cross_attns + prosody_raw_proj
                             (prosody_direct stays active)
    Stage 4: Emotion attn    frame_raw_proj + cross_attns
    Stage 5: Fusion          emo_prosody_fusion
    Stage 6: DiT blocks      top-K transformer blocks + norm_out + proj_out
    Stage 0: (no curriculum — joint, use apply_freeze)

    Recommended flow: 1 → 2 → 3 → 4 → 5 → (6) → 0 (joint finetune, low LR).
    """
    if stage == 0:
        return

    agg = model.transformer.cond_aggregator

    def unfreeze(mod):
        if mod is None:
            return
        if isinstance(mod, torch.nn.Parameter):
            mod.requires_grad = True
        else:
            for p in mod.parameters():
                p.requires_grad = True

    def _collect_gates(mod):
        """All gate Parameters under `mod` — direct or on nested submodules
        (EmoProsodyFusion keeps gates on its two _FusionCrossAttn children).
        Handles ModuleDict (.values()), single modules, and recursion."""
        if mod is None:
            return []
        items = list(mod.values()) if hasattr(mod, "values") else [mod]
        gates = []
        for m in items:
            if hasattr(m, "gate") and isinstance(m.gate, torch.nn.Parameter):
                gates.append(m.gate)
            elif isinstance(m, torch.nn.Module):
                for sub in m.modules():
                    if sub is not m and hasattr(sub, "gate") and isinstance(sub.gate, torch.nn.Parameter):
                        gates.append(sub.gate)
        return gates

    def zero_gates_in(mod, value=0.0):
        for g in _collect_gates(mod):
            with torch.no_grad():
                g.fill_(value)

    def zero_param(name, value=0.0):
        p = getattr(agg, name, None)
        if p is not None and isinstance(p, torch.nn.Parameter):
            with torch.no_grad():
                p.fill_(value)

    def restore_gate(name, value):
        p = getattr(agg, name, None)
        if p is not None and isinstance(p, torch.nn.Parameter):
            with torch.no_grad():
                p.fill_(value)
            p.requires_grad = True

    def gates_all_zero(mod):
        gates = _collect_gates(mod)
        if not gates:
            return False
        return all(g.detach().abs().max().item() < 1e-6 for g in gates)

    # ── Freeze everything ──────────────────────────────────────────────
    for p in model.parameters():
        p.requires_grad = False

    # ── Zero gates of modules not yet trained (future stages only) ─────
    # Stage N zeroes gates belonging to stages > N so frozen-untrained
    # modules inject no noise into the residual stream. (Shared with the
    # resume path via zero_future_stage_gates so the two never drift.)
    zero_future_stage_gates(model, stage)

    # ── AdaLN + input_add: active on stages 1-6 by default ─────────────
    adaln_active = (stage == 1) or (stage in (2, 3, 4, 5, 6) and not freeze_adaln)
    if adaln_active:
        unfreeze(getattr(agg, "adaln_cond", None))
        unfreeze(getattr(agg, "input_add", None))

    # ── fusion + emotion_raw_proj: active stages 1-6 ───────────────────
    # fusion(speaker_emb, emotion_emb) feeds adaln_cond/input_add.
    # Both inputs (speaker_raw_proj, emotion_raw_proj) must be trained for
    # fusion to receive clean signal — so emotion_raw_proj trains from stage 1.
    if stage in (1, 2, 3, 4, 5, 6):
        unfreeze(getattr(agg, "fusion", None))
        unfreeze(getattr(agg, "emotion_raw_proj", None))

    # ── Stage 1: Speaker + global conditioning ─────────────────────────
    if stage >= 1:
        unfreeze(getattr(agg, "speaker_raw_proj", None))
        if model.duration_predictor is not None:
            for p in model.duration_predictor.parameters():
                p.requires_grad = True

    # ── Stage 2: Prosody direct (coarse pitch/energy, no attention) ────
    if stage >= 2:
        unfreeze(getattr(agg, "prosody_direct_proj", None))
        unfreeze(getattr(agg, "prosody_temporal_smooth", None))
        unfreeze(getattr(agg, "prosody_global_proj", None))
        if stage == 2:
            pg = getattr(agg, "prosody_block_gates", None)
            if pg is not None and pg.detach().abs().max().item() < 1e-6:
                # First run: gates are still zeroed → restore to small init.
                restore_gate("prosody_block_gates", 0.02)
                restore_gate("prosody_global_gate", 0.02)
            else:
                # Resuming stage 2 with already-trained gates: keep their learned
                # per-block values as-is (do NOT collapse them to their mean,
                # which would erase the per-block specialisation); just re-enable
                # gradients.
                for attr in ("prosody_block_gates", "prosody_global_gate"):
                    p = getattr(agg, attr, None)
                    if p is not None and isinstance(p, torch.nn.Parameter):
                        p.requires_grad = True
        else:
            # Stage 3+: prosody direct already trained, keep gates as-is
            for attr in ("prosody_block_gates", "prosody_global_gate"):
                p = getattr(agg, attr, None)
                if p is not None and isinstance(p, torch.nn.Parameter):
                    p.requires_grad = True

    # ── Stage 3: Prosody cross-attention (fine frame-level pitch) ──────
    if stage >= 3:
        unfreeze(getattr(agg, "prosody_cross_attns", None))
        unfreeze(getattr(agg, "prosody_raw_proj", None))
        if stage == 3 and gates_all_zero(getattr(agg, "prosody_cross_attns", None)):
            zero_gates_in(getattr(agg, "prosody_cross_attns", None), 0.15)

    # ── Stage 4: Emotion cross-attention (intonation/affect) ───────────
    if stage >= 4:
        unfreeze(getattr(agg, "frame_raw_proj", None))
        unfreeze(getattr(agg, "cross_attns", None))
        if stage == 4 and gates_all_zero(getattr(agg, "cross_attns", None)):
            zero_gates_in(getattr(agg, "cross_attns", None), 0.15)

    # ── Stage 5: Emotion↔Prosody fusion ────────────────────────────────
    if stage >= 5:
        unfreeze(getattr(agg, "emo_prosody_fusion", None))
        if stage == 5 and gates_all_zero(getattr(agg, "emo_prosody_fusion", None)):
            zero_gates_in(getattr(agg, "emo_prosody_fusion", None), 0.15)

    # ── Stage 6: DiT base blocks ───────────────────────────────────────
    if stage >= 6:
        if unfreeze_top_k <= 0:
            print(f"  ⚠ Stage 6 requires --unfreeze_top_k > 0 to unfreeze DiT blocks. "
                  f"Currently no DiT blocks will be unfrozen.")
        else:
            blocks = model.transformer.transformer_blocks
            for i in range(max(0, len(blocks) - unfreeze_top_k), len(blocks)):
                for p in blocks[i].parameters():
                    p.requires_grad = True
            for p in model.transformer.norm_out.parameters():
                p.requires_grad = True
            for p in model.transformer.proj_out.parameters():
                p.requires_grad = True


    stage_names = {
        1: "Speaker+Global (speaker+emotion_raw_proj + fusion + adaln + input_add + duration)",
        2: "Prosody direct (prosody_direct/global_proj + temporal_smooth + gates)",
        3: "Prosody attn   (prosody_cross_attns + prosody_raw_proj)",
        4: "Emotion attn   (frame_raw_proj + cross_attns)",
        5: "Fusion         (emo_prosody_fusion)",
        6: f"DiT blocks     (top-{unfreeze_top_k} blocks + norm_out + proj_out)",
    }
    total    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Stage {stage}: {stage_names.get(stage, 'unknown')}")
    print(f"  Total: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.1f}%)")


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
        cross_attn:   frame-level cross-attention paths (emotion + prosody)
                      Slowest to learn — deep attention chain needs high LR.
        prosody:      prosody direct/global modules (fast learners)
        conditioning: speaker/emotion global projections, AdaLN, fusion, input_add
        duration:     duration predictor (separate task)
        base:         unfrozen DiT blocks (pretrained, conservative LR)

    All LRs are multiples of args.learning_rate.
    """
    lr = args.learning_rate

    # Collect parameter id sets for each group
    speaker_ids = set()
    emotion_ids = set()
    cross_attn_ids = set()
    proj_ids = set()
    prosody_ids = set()
    base_ids = set()
    duration_ids = set()

    agg = model.transformer.cond_aggregator

    # Speaker/global conditioning group (stable identity signal)
    speaker_module_names = [
        "adaln_cond",       # fused_global → scale/shift in every DiT block
        "fusion",           # concat-mode fusion (None in residual mode)
        "fusion_speaker",   # residual-mode: speaker path (always reaches fused)
        "fusion_emotion",   # residual-mode: emotion correction
        "fusion_emotion_gate",
        "speaker_raw_proj", # raw WavLM-SV → speaker_emb (stable per utterance)
        "emotion_raw_proj", # GLOBAL emotion → other input to shared fusion;
                            # kept here (not emotion group) so both fusion inputs
                            # train at the SAME lr → no speaker/emotion competition
        "input_add",        # fused_global → DiT input residual
    ]
    for name in speaker_module_names:
        mod = getattr(agg, name, None)
        if mod is not None:
            if isinstance(mod, nn.Parameter):
                speaker_ids.add(id(mod))
            else:
                for p in mod.parameters():
                    speaker_ids.add(id(p))

    # Emotion group: all emotion-related modules together
    #
    # emotion_raw_proj:  global emotion2vec → emotion_emb → fusion → adaln_cond
    #   gradient path: 22 DiT blocks (aggregated) but EMOTION VARIES per utterance
    #   → gradient is more variable than speaker → beta2=0.95 (faster adaptation)
    #
    # cross_attns: frame-level emotion attention (blocks 0,4,8,12,16,20)
    #   gradient: gate(0.15, decreasing) × 6 blocks → weak
    #   → beta2=0.95
    #
    # emo_prosody_fusion: cross-modal emotion↔prosody frame fusion
    #   gradient: through cross_attns gate → weak
    #   → beta2=0.95
    # emotion FRAME modules only — emotion_raw_proj (GLOBAL) is intentionally
    # in the speaker group instead, because it is the other input to the shared
    # `fusion` module. If emotion_raw_proj trains faster (cross_attn_mult=2x)
    # than speaker_raw_proj/fusion (speaker_mult=1x), fusion adapts to the
    # emotion half first and underuses the speaker half → voice stops cloning
    # while emotion/prosody still respond. Keeping BOTH fusion inputs in the
    # same group at the same LR prevents that competition.
    emotion_module_names = [
        "cross_attns",        # emotion frame cross-attention
        "emo_prosody_fusion", # emotion↔prosody cross-modal fusion
    ]
    for name in emotion_module_names:
        mod = getattr(agg, name, None)
        if mod is not None:
            if isinstance(mod, nn.Parameter):
                emotion_ids.add(id(mod))
            else:
                for p in mod.parameters():
                    emotion_ids.add(id(p))

    # Cross-attention modules: prosody attention only (emotion moved to emotion group)
    cross_attn_module_names = [
        "prosody_cross_attns",   # prosody frame cross-attention blocks
    ]
    for name in cross_attn_module_names:
        mod = getattr(agg, name, None)
        if mod is not None:
            if isinstance(mod, nn.Parameter):
                cross_attn_ids.add(id(mod))
            else:
                for p in mod.parameters():
                    cross_attn_ids.add(id(p))

    # Input projection group: frame_raw_proj + prosody_raw_proj
    # These are DEEPEST in the gradient chain:
    #   loss → 6 DiT blocks → gate(0.15, decreasing) → cross_attn → proj
    # Weakest gradient in the entire model → most aggressive beta2
    for name in ("frame_raw_proj", "prosody_raw_proj"):
        mod = getattr(agg, name, None)
        if mod is not None:
            for p in mod.parameters():
                proj_ids.add(id(p))

    # Prosody group: direct addition + global stats (fast learners, already working)
    prosody_module_names = [
        "prosody_direct_proj", "prosody_temporal_smooth", "prosody_global_proj",
    ]
    for name in prosody_module_names:
        mod = getattr(agg, name, None)
        if mod is not None:
            if isinstance(mod, nn.Parameter):
                prosody_ids.add(id(mod))
            else:
                for p in mod.parameters():
                    prosody_ids.add(id(p))
    # Scalar gates for prosody
    for attr in ("prosody_block_gates", "prosody_global_gate"):
        p = getattr(agg, attr, None)
        if p is not None and isinstance(p, nn.Parameter):
            prosody_ids.add(id(p))

    # Duration predictor
    if model.duration_predictor is not None:
        for p in model.duration_predictor.parameters():
            duration_ids.add(id(p))

    # Unfrozen DiT base params (pretrained) — collected by ACTUAL requires_grad,
    # not gated on args.freeze_base. DiT blocks can be unfrozen by either path:
    #   - apply_freeze(freeze_base=True, unfreeze_top_k>0), OR
    #   - apply_curriculum_stage(stage=6, unfreeze_top_k>0), which does NOT set
    #     freeze_base, OR
    #   - joint stage 0 with freeze_base=False (all blocks trainable).
    # Previously base_ids was filled only under `freeze_base and unfreeze_top_k>0`,
    # so stage-6 / no-freeze runs left these trainable pretrained params out of
    # every conditioning set → they fell through to the conditioning group and
    # trained at full LR instead of the conservative lr*lr_base_mult, risking
    # catastrophic damage to pretrained weights. Keying on requires_grad covers
    # all unfreeze paths and is safe (DiT params never overlap conditioning sets).
    for blk in model.transformer.transformer_blocks:
        for p in blk.parameters():
            if p.requires_grad:
                base_ids.add(id(p))
    for mod in (model.transformer.norm_out, model.transformer.proj_out):
        for p in mod.parameters():
            if p.requires_grad:
                base_ids.add(id(p))

    # Build groups — each trainable param in exactly one group
    proj_params, cross_attn_params, emotion_params, speaker_params, prosody_params, cond_params, dur_params, base_params = [], [], [], [], [], [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in proj_ids:
            proj_params.append(p)
        elif pid in emotion_ids:
            emotion_params.append(p)
        elif pid in cross_attn_ids:
            cross_attn_params.append(p)
        elif pid in speaker_ids:
            speaker_params.append(p)
        elif pid in prosody_ids:
            prosody_params.append(p)
        elif pid in duration_ids:
            dur_params.append(p)
        elif pid in base_ids:
            base_params.append(p)
        else:
            cond_params.append(p)

    groups = []
    if proj_params:
        groups.append({
            "params": proj_params,
            "lr": lr * args.lr_cross_attn_mult,
            "name": "input_proj",
            # beta2=0.9: deepest gradient chain (gate × 6 blocks → proj)
            # Maximum responsiveness to rare/weak gradient signal.
            "betas": (0.9, 0.9),
        })
    if emotion_params:
        groups.append({
            "params": emotion_params,
            "lr": lr * args.lr_cross_attn_mult,
            "name": "emotion",
            # beta2=0.95: emotion varies per utterance → more variable gradient
            # than speaker. Faster adaptation than speaker group.
            # emotion_raw_proj: 22-block aggregated but variable signal
            # cross_attns + emo_fusion: gate-attenuated, weak
            "betas": (0.9, 0.95),
        })
    if cross_attn_params:
        groups.append({
            "params": cross_attn_params,
            "lr": lr * args.lr_cross_attn_mult,
            "name": "prosody_cross_attn",
            # beta2=0.95: prosody cross-attn, gate-attenuated gradient
            "betas": (0.9, 0.95),
        })
    if speaker_params:
        groups.append({
            "params": speaker_params,
            "lr": lr * args.lr_speaker_mult,
            "name": "speaker",
            # beta2=0.98: adaln_cond gets 22× aggregated gradient from DiT blocks.
            # Slightly conservative to avoid overshooting from large gradient.
            "betas": (0.9, 0.98),
        })
    if prosody_params:
        groups.append({
            "params": prosody_params,
            "lr": lr * args.lr_prosody_mult,
            "name": "prosody",
            # beta2=0.95: prosody_block_gates start at 0.02 → weak gradient to proj.
            # Faster adaptation helps when gradient signal is small.
            "betas": (0.9, 0.95),
        })
    if cond_params:
        groups.append({
            "params": cond_params,
            "lr": lr,
            "name": "conditioning",
            "betas": (0.9, 0.98),
        })
    if dur_params:
        groups.append({
            "params": dur_params,
            "lr": lr * args.lr_duration_mult,
            "name": "duration",
            # Standard betas for simple regression task
            "betas": (0.9, 0.98),
        })
    if base_params:
        groups.append({
            "params": base_params,
            "lr": lr * args.lr_base_mult,
            "name": "base_blocks",
            # beta2=0.999: pretrained weights, very conservative updates.
            # Slow second moment decay = resistant to sudden gradient spikes.
            "betas": (0.9, 0.999),
        })

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

    def _make_dataset(hf_ds, durs, emb_index_map=None, is_train=True):
        ds = EnhancedDataset(
            hf_dataset=hf_ds, durations=durs,
            embedding_cache_dir=args.embedding_dir,
            embedding_index_map=emb_index_map,
            target_sample_rate=TARGET_SR, hop_length=HOP,
            n_mel_channels=N_MEL, n_fft=N_FFT, win_length=WIN, mel_spec_type=MEL_TYPE,
            speed_perturb=args.speed_perturb and is_train,
        )
        ds.training = is_train
        return ds

    train_dataset = _make_dataset(train_ds, train_durs, train_emb_map, is_train=True)

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
                rebuild_every_n_steps=args.rebuild_every_n_steps,
            )
        elif args.bucket_batching:
            bs = BucketDynamicBatchSampler(
                sampler,
                args.batch_size_per_gpu,
                max_samples=args.max_samples,
                bucket_size=args.bucket_size,
                random_seed=args.seed,
                drop_residual=False,
                rebuild_every_n_steps=args.rebuild_every_n_steps,
            )
        else:
            bs = DynamicBatchSampler(
                sampler,
                args.batch_size_per_gpu,
                max_samples=args.max_samples,
                random_seed=args.seed,
                drop_residual=False,
                rebuild_every_n_steps=args.rebuild_every_n_steps,
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
        val_dataset = _make_dataset(val_ds, val_durs, val_emb_map, is_train=False)
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
    arch_config: dict | None = None,
):
    """Save a resumable checkpoint (online model + EMA + optimizer + scheduler).

    arch_config: architecture flags (fusion_mode, adaln_bottleneck_dim,
    use_prosody_direct/cross_attn, dims, ...) needed to REBUILD the model
    identically at inference. Persisted so inference doesn't have to guess —
    a mismatch silently drops conditioning weights → broken cloning.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }
    if arch_config is not None:
        payload["arch_config"] = arch_config

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

    # 4. Freeze / curriculum stage
    if is_main:
        print(f"Freeze: base={args.freeze_base}, unfreeze_top_k={args.unfreeze_top_k}, "
              f"train_stage={args.train_stage}")
    if args.train_stage > 0:
        if args.train_stage >= 2:
            last_ckpt = os.path.join(args.checkpoint_dir, "model_last.pt")
            if not os.path.exists(last_ckpt):
                raise ValueError(
                    f"--train_stage {args.train_stage} requires a pre-trained checkpoint "
                    f"(stage 1 must be done first). No checkpoint found at:\n"
                    f"  {last_ckpt}\n"
                    f"Run --train_stage 1 first to train speaker + AdaLN."
                )
            if is_main:
                print(f"  ✓ Checkpoint found for stage {args.train_stage}")
        apply_curriculum_stage(model, args.train_stage, args.unfreeze_top_k,
                               freeze_adaln=args.freeze_adaln)
    else:
        apply_freeze(model, args.freeze_base, args.unfreeze_top_k)

    # 5. EMA (main process)
    # decay controls how fast EMA tracks the online model.
    # 0.9999 (default) suits large datasets / long training.
    # For smaller datasets or staged curriculum (fewer steps per stage),
    # a faster EMA (0.999) tracks rapidly-changing weights better.
    ema = (
        EMA(
            model,
            beta=args.ema_decay,
            update_after_step=args.ema_update_after,
            update_every=args.ema_update_every,
            include_online_model=False,
        )
        if is_main else None
    )
    if ema:
        ema.to(dev)

    # 6. Optimizer with differential learning rates
    param_groups = build_param_groups(model, args)
    optimizer = AdamW(
        param_groups,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        # betas set per-group in build_param_groups
        eps=args.adam_eps,
    )
    trainable = [p for g in param_groups for p in g["params"]]

    # 7. Data
    dataset, dataloader, val_loader = build_dataloaders(args)

    # 8. Encoders
    spk_enc, emo_enc, prosody_enc = maybe_load_encoders(args, str(dev))

    # 9. Scheduler
    # IMPORTANT: scheduler.step() is called once per optimizer sync step, which
    # happens PER RANK. So total/warmup must be in per-rank optimizer steps.
    # At this point `dataloader` has NOT yet been through accelerator.prepare(),
    # so len(dataloader) is the GLOBAL batch count (all ranks). Divide by
    # num_processes to get the per-rank batch count, then by grad accumulation.
    # (Previously: `total` used the global count → scheduler only traversed 1/N
    #  of its schedule on N GPUs; `warmup` was multiplied by num_processes →
    #  warmup phase was N× too long. Both are now in consistent per-rank units.)
    nproc = max(accelerator.num_processes, 1)
    per_rank_batches = math.ceil(len(dataloader) / nproc)
    steps_per_epoch = math.ceil(per_rank_batches / args.grad_accumulation_steps)
    total = steps_per_epoch * args.epochs
    warmup = args.num_warmup_updates
    decay = max(total - warmup, 1)

    warmup_sched = LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup
    )
    if args.scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        decay_sched = CosineAnnealingLR(
            optimizer, T_max=decay, eta_min=args.learning_rate * 1e-2
        )
    else:
        decay_sched = LinearLR(
            optimizer, start_factor=1.0, end_factor=1e-2, total_iters=decay
        )

    # Per-group extended warmup for cross-attention
    # Cross-attn needs longer warmup: random K/V → model learns to close gate
    # immediately without warmup. With 3-5× warmup, gate stabilises before decaying.
    cross_attn_warmup = args.num_warmup_updates_cross_attn  # per-rank steps (no num_processes scaling)
    if cross_attn_warmup > warmup:
        if is_main:
            print(f"  Cross-attn extended warmup: {cross_attn_warmup} steps "
                  f"(global warmup: {warmup} steps)")
            print(f"  → cross-attn LR will reach peak at step {cross_attn_warmup} "
                  f"(other groups peak at step {warmup})")

        # Capture target LRs from scheduler base_lrs (NOT g["lr"] — the LinearLR
        # warmup scheduler already scaled g["lr"] down by start_factor=1e-8 at
        # creation, so g["lr"] no longer holds the true target). base_lrs holds
        # the correct pre-warmup target LR for each group.
        # Then zero the group LR so the global scheduler contributes nothing;
        # per-step override in the training loop drives these groups.
        _cross_attn_init_lrs = {}
        for i, g in enumerate(param_groups):
            if g.get("name") in ("cross_attn", "prosody_cross_attn", "emotion", "input_proj"):
                _cross_attn_init_lrs[i] = warmup_sched.base_lrs[i]
                g["lr"] = 0.0  # start at 0, ramp up separately
    else:
        _cross_attn_init_lrs = {}
        cross_attn_warmup = 0

    scheduler = SequentialLR(
        optimizer, [warmup_sched, decay_sched], milestones=[warmup]
    )

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
            # load_state_dict just overwrote gate VALUES with the checkpoint's;
            # for a staged run that re-introduces non-zero gates on not-yet-trained
            # future modules (they init at 0.02/0.15). Re-zero them so untrained
            # modules stay silent. requires_grad was set by apply_curriculum_stage
            # and is unaffected by load_state_dict.
            if args.train_stage > 0:
                zero_future_stage_gates(accelerator.unwrap_model(model), args.train_stage)
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

    # Pre-compute cross-attn decay_steps once (used in training loop per-step).
    # Reuse the per-rank `total` computed for the global scheduler so the two
    # schedules share the same step basis (recomputing here risked divergence,
    # and post-prepare len(dataloader) is per-rank just like `total`).
    if _cross_attn_init_lrs:
        _cross_attn_decay_steps = max(total - cross_attn_warmup, 1)
    else:
        _cross_attn_decay_steps = 0

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

    # Architecture flags persisted in every checkpoint so inference rebuilds the
    # model identically (a mismatch silently drops conditioning weights → broken
    # cloning). Read by load_enhanced_model via the checkpoint's "arch_config".
    arch_config = {
        "speaker_emb_dim": args.speaker_emb_dim,
        "emotion_emb_dim": args.emotion_emb_dim,
        "adaln_bottleneck_dim": args.adaln_bottleneck_dim,
        "use_adaln_cond": not args.no_adaln,
        "use_input_add_cond": not args.no_input_add,
        "use_cross_attn_cond": not args.no_cross_attn,
        "speaker_raw_dim": SPEAKER_RAW_DIMS.get(args.speaker_backend, 512),
        "emotion_raw_dim": EMOTION_RAW_DIMS.get(args.emotion_backend, 768),
        "prosody_dim": args.prosody_dim,
        "prosody_raw_dim": PROSODY_RAW_DIM if not args.no_prosody else None,
        "use_prosody_cross_attn": (not args.no_prosody) and (not args.no_prosody_cross_attn),
        "use_prosody_direct": (not args.no_prosody) and (not args.no_prosody_direct),
        "fusion_mode": args.fusion_mode,
        "cross_attn_gate_floor": args.cross_attn_gate_floor,
    }

    def _crossattn_gate_report():
        """Effective cross-attn strength per fusion layer = tanh(gate).
        ~0 → that cross-attn is effectively disabled (model ignores it);
        larger → it is actively contributing. Returns {name: mean_abs_tanh_gate}."""
        agg = getattr(raw_model.transformer, "cond_aggregator", None)
        if agg is None:
            return {}
        report = {}
        for attr, label in (("cross_attns", "emo_xattn"), ("prosody_cross_attns", "pros_xattn")):
            md = getattr(agg, attr, None)
            if md is None:
                continue
            vals = []
            for mod in md.values():
                # a fusion module may hold its own gate and/or sub-gates;
                # modules() yields the module itself plus all descendants.
                for sub in mod.modules():
                    g = getattr(sub, "gate", None)
                    if isinstance(g, torch.nn.Parameter):
                        vals.append(torch.tanh(g.detach()).abs().mean().item())
            if vals:
                report[label] = sum(vals) / len(vals)
        return report

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

                loss, _, _, mode_names, dur_loss, per_sample_loss = model(
                    inp=mel, text=text, lens=lens,
                    conditions=conditions,
                    # Duration predictor inputs
                    prosody_raw=batch.get("prosody_raw"),
                    prosody_mask=batch.get("prosody_mask"),
                    text_byte_lens=torch.tensor(
                        [len(t) for t in batch["text"]],  # CHARS not bytes: byte counts skew across scripts (Cyrillic=2B/char) → cross-language duration bias
                        dtype=torch.long,
                    ) if getattr(raw_model, "duration_predictor", None) is not None else None,
                    speaker_emb_for_dur=batch.get("speaker_raw"),
                )

                # Combined loss (dur_loss is 0.0 when predictor disabled — free to add)
                total_loss = loss + args.dur_loss_weight * dur_loss


                # ── NaN/Inf guard: skip corrupted batches (DDP-safe) ──
                # The skip decision MUST be identical on every rank. If one rank
                # saw NaN and bailed out via `continue` (skipping backward) while
                # others called accelerator.backward(), the others would initiate
                # the gradient all-reduce and block forever waiting for the rank
                # that left — a hard DDP deadlock. So we reduce a local NaN flag
                # across all ranks (max = logical OR): if ANY rank has NaN, ALL
                # ranks skip together; otherwise ALL proceed to backward. Same
                # logic makes the consecutive-NaN epoch-break fire in lockstep.
                local_bad = torch.tensor(
                    [0.0 if torch.isfinite(total_loss) else 1.0], device=dev
                )
                if accelerator.num_processes > 1:
                    local_bad = accelerator.reduce(local_bad, reduction="max")
                batch_is_bad = bool(local_bad.item() > 0)

                if batch_is_bad:
                    optimizer.zero_grad()
                    nan_consecutive += 1
                    if is_main:
                        tqdm.write(f"  ⚠ step ~{step}: NaN/Inf loss on ≥1 rank, skipping batch "
                                   f"({nan_consecutive} consecutive)")
                    if nan_consecutive >= 5:
                        if is_main:
                            tqdm.write("  ✖ 5 consecutive NaN batches — model weights likely "
                                       "corrupted. Stopping epoch.")
                        break
                    continue
                
                # Per-mode loss tracking: EXACT per-sample attribution.
                # per_sample_loss[i] is the masked-frame mean for sample i, and
                # mode_names[i] is that sample's dropout mode — so each mode gets
                # only the loss of its own samples (no batch-mean smearing).
                # NaN entries = samples with no masked frames → skipped.
                psl = per_sample_loss.detach()
                for i, m in enumerate(mode_names):
                    li = psl[i].item()
                    if li == li:  # not NaN
                        mode_loss_sum[m] += li
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

                    # Per-group extended warmup for cross-attn groups.
                    # Phase 1 (step < cross_attn_warmup): linear ramp 0 → target_lr
                    # Phase 2 (step >= cross_attn_warmup): cosine decay target_lr → target_lr * 0.01
                    # This gives cross-attn groups their own independent warmup+cosine schedule.
                    # decay_steps pre-computed before loop (see _cross_attn_decay_steps).
                    if _cross_attn_init_lrs:
                        for i, target_lr in _cross_attn_init_lrs.items():
                            if step < cross_attn_warmup:
                                # Linear warmup
                                scale = step / max(cross_attn_warmup, 1)
                                optimizer.param_groups[i]["lr"] = target_lr * scale
                            else:
                                # Cosine decay from target_lr to target_lr * 0.01
                                t = (step - cross_attn_warmup) / _cross_attn_decay_steps
                                t = min(t, 1.0)
                                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * t))
                                min_lr = target_lr * 1e-2
                                optimizer.param_groups[i]["lr"] = min_lr + (target_lr - min_lr) * cosine_factor

            if accelerator.sync_gradients:
                if is_main and ema:
                    ema.update()
                step += 1
                # Track the value actually being minimized (flow + weighted dur),
                # not the bare flow loss — otherwise the epoch average and the
                # progress bar disagree with the optimization target whenever the
                # duration predictor is active.
                total_loss_val = total_loss.item()
                flow_loss_val = loss.item()
                epoch_loss += total_loss_val
                epoch_steps += 1
                _has_dur = getattr(raw_model, "duration_predictor", None) is not None
                postfix = dict(loss=f"{total_loss_val:.4f}",
                               avg=f"{epoch_loss/epoch_steps:.4f}",
                               lr=f"{scheduler.get_last_lr()[0]:.2e}", s=step)
                if _has_dur:
                    # Surface the components so it's clear what drives total.
                    postfix["flow"] = f"{flow_loss_val:.4f}"
                    postfix["dur"] = f"{float(dur_loss.item()) if torch.is_tensor(dur_loss) else float(dur_loss):.4f}"
                pbar.set_postfix(**postfix)

                if is_main and args.logger:
                    log_d = {"train/loss": total_loss_val,
                             "train/flow_loss": flow_loss_val,
                             "train/lr": scheduler.get_last_lr()[0]}
                    if _has_dur:
                        log_d["train/dur_loss"] = (
                            float(dur_loss.item()) if torch.is_tensor(dur_loss) else float(dur_loss)
                        )
                    accelerator.log(log_d, step=step)

                if args.logger == "tensorboard" and accelerator.is_main_process:
                    writer.add_scalar("train_loss", total_loss_val, step)
                    writer.add_scalar("train/flow_loss", flow_loss_val, step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                    if _has_dur:
                        writer.add_scalar("train/dur_loss",
                                          float(dur_loss.item()) if torch.is_tensor(dur_loss) else float(dur_loss),
                                          step)
                
                if step % 100 == 0 and any(mode_count.values()):

                    _cond_report = {}
                    for m in ("full", "uncond", "voice_only", "no_emotion", "no_prosody", "no_speaker", "textless"):
                        ls = torch.tensor(mode_loss_sum.get(m, 0.0), device=accelerator.device, dtype=torch.float32)
                        ct = torch.tensor(mode_count.get(m, 0), device=accelerator.device, dtype=torch.float32)

                        ls = accelerator.reduce(ls, reduction="sum")
                        ct = accelerator.reduce(ct, reduction="sum")

                        if is_main and ct.item() > 0:
                            mval = (ls / ct).item()
                            _cond_report[m] = mval
                            accelerator.log({f"train/loss_{m}": mval}, step=step)

                            if args.logger == "tensorboard":
                                writer.add_scalar(f"train/loss_{m}", mval, step)

                    # Console summary: per-condition loss + contribution of each
                    # signal = (loss without it) - (full). Positive = the model
                    # relies on that condition (removing it hurts). Near-zero for
                    # a signal means it is NOT being used — useful to diagnose
                    # e.g. weak speaker conditioning (poor voice cloning).
                    if is_main and _cond_report:
                        full = _cond_report.get("full")
                        parts = [f"{m}={v:.4f}" for m, v in _cond_report.items()]
                        line = "  [cond] " + "  ".join(parts)
                        if full is not None:
                            contrib = []
                            for m, label in (("no_speaker", "spk"), ("no_prosody", "pros"),
                                             ("no_emotion", "emo"), ("uncond", "all")):
                                if m in _cond_report:
                                    contrib.append(f"{label}+{_cond_report[m]-full:.4f}")
                            if contrib:
                                line += "  | contrib(↑=used): " + " ".join(contrib)
                        tqdm.write(line)

                    mode_loss_sum.clear()
                    mode_count.clear()

                    # Cross-attention diagnostic: effective gate strength per
                    # fusion type. Near-zero → cross-attn effectively off.
                    if is_main:
                        gates = _crossattn_gate_report()
                        if gates:
                            gline = "  [xattn] " + "  ".join(f"{k}={v:.4f}" for k, v in gates.items())
                            tqdm.write(gline)
                            if args.logger:
                                accelerator.log({f"train/gate_{k}": v for k, v in gates.items()}, step=step)
                            if args.logger == "tensorboard":
                                for k, v in gates.items():
                                    writer.add_scalar(f"train/gate_{k}", v, step)
                
            if step % args.last_per_updates == 0 and accelerator.sync_gradients:
                accelerator.wait_for_everyone()
                if is_main:
                    save_checkpoint(raw_model, ema, optimizer, scheduler, step, epoch,
                                    os.path.join(args.checkpoint_dir, "model_last.pt"), arch_config=arch_config)

            if step % args.save_per_updates == 0 and accelerator.sync_gradients:
                accelerator.wait_for_everyone()
                if is_main:
                    save_checkpoint(raw_model, ema, optimizer, scheduler, step, epoch,
                                    os.path.join(args.checkpoint_dir, f"model_{step}.pt"), arch_config=arch_config)
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
                                        os.path.join(args.checkpoint_dir, "model_best.pt"), arch_config=arch_config)

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
                                    os.path.join(args.checkpoint_dir, "model_best.pt"), arch_config=arch_config)
        elif is_main:
            print(f"  Epoch {epoch+1} done — avg loss: {epoch_loss/max(epoch_steps,1):.4f}")

    # Final save
    accelerator.wait_for_everyone()
    if is_main:
        save_checkpoint(raw_model, ema, optimizer, scheduler, step, args.epochs,
                        os.path.join(args.checkpoint_dir, "model_last.pt"), arch_config=arch_config)
        print(f"\nDone! {step} updates → {args.checkpoint_dir}/model_last.pt")
        if val_loader is not None and best_val_loss < float("inf"):
            print(f"Best val loss: {best_val_loss:.5f} → {args.checkpoint_dir}/model_best.pt")
    accelerator.end_training()
    if writer and hasattr(writer, 'close'):
        writer.close()


if __name__ == "__main__":
    main()
