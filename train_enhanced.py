"""
F5-TTS Enhanced — Training Entry Point
=========================================

Complete training pipeline:
    Stage 0: Install dependencies
    Stage 1: Download pretrained F5-TTS base checkpoint
    Stage 2: Prepare dataset + extract embeddings (if not done)
    Stage 3: Launch finetuning of conditioning modules

Usage:
    # Minimal (single GPU, auto-downloads F5-TTS base)
    python train_enhanced.py \
        --dataset_dir /data/f5tts_prepared \
        --embedding_dir /data/f5tts_embeddings

    # Full control
    python train_enhanced.py \
        --dataset_dir /data/f5tts_prepared \
        --embedding_dir /data/f5tts_embeddings \
        --pretrain_ckpt /path/to/F5TTS_Base/model_1200000.pt \
        --epochs 5 \
        --learning_rate 3e-4 \
        --batch_size_per_gpu 19200 \
        --save_per_updates 5000 \
        --unfreeze_top_k 0 \
        --logger wandb

    # Multi-GPU (via accelerate)
    accelerate launch --multi_gpu --num_processes 4 train_enhanced.py \
        --dataset_dir /data/f5tts_prepared \
        --embedding_dir /data/f5tts_embeddings
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import shutil
import sys
from importlib.resources import files

import torch
import torchaudio
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from f5_tts.model.backbones.enhanced_dit import EnhancedDiT
from f5_tts.model.enhanced_cfm import EnhancedCFM
from f5_tts.model.enhanced_dataset import EnhancedCustomDataset, enhanced_collate_fn
from f5_tts.model.dataset import DynamicBatchSampler
from f5_tts.model.speaker_encoder import SpeakerEncoder
from f5_tts.model.emotion_encoder import EmotionEncoder
from f5_tts.model.utils import get_tokenizer, default, exists

# ── Audio config (same as F5-TTS base) ──
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"


def parse_args():
    p = argparse.ArgumentParser(description="F5-TTS Enhanced Training")

    # ── Data ──
    p.add_argument("--dataset_dir", type=str, required=True,
                    help="Path to prepared dataset (with raw.arrow + duration.json)")
    p.add_argument("--embedding_dir", type=str, default=None,
                    help="Path to precomputed embeddings. If None, uses online extraction (slow)")
    p.add_argument("--vocab_file", type=str, default=None,
                    help="Path to vocab.txt. If None, uses default F5-TTS vocab")

    # ── Pretrained base model ──
    p.add_argument("--pretrain_ckpt", type=str, default=None,
                    help="Path to original F5-TTS checkpoint. If None, auto-downloads from HuggingFace")
    p.add_argument("--exp_name", type=str, default="F5TTS_Base",
                    choices=["F5TTS_v1_Base", "F5TTS_Base"],
                    help="Which F5-TTS base architecture to use")

    # ── Training ──
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=3e-4,
                    help="LR for conditioning modules (high because few params)")
    p.add_argument("--batch_size_per_gpu", type=int, default=19200,
                    help="Batch size in frames per GPU")
    p.add_argument("--batch_size_type", type=str, default="frame",
                    choices=["frame", "sample"])
    p.add_argument("--max_samples", type=int, default=32,
                    help="Max sequences per batch (frame mode)")
    p.add_argument("--grad_accumulation_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--num_warmup_updates", type=int, default=2000)

    # ── Freezing strategy ──
    p.add_argument("--freeze_base", action="store_true", default=True,
                    help="Freeze original F5-TTS weights (default: True)")
    p.add_argument("--no_freeze_base", dest="freeze_base", action="store_false",
                    help="Train all parameters (not recommended)")
    p.add_argument("--unfreeze_top_k", type=int, default=0,
                    help="Unfreeze top-K DiT blocks (0=fully frozen, 4=recommended for stage 2)")

    # ── Conditioning ──
    p.add_argument("--speaker_emb_dim", type=int, default=512)
    p.add_argument("--emotion_emb_dim", type=int, default=512)
    p.add_argument("--speaker_backend", type=str, default="wavlm_sv",
                    choices=["wavlm_sv", "ecapa_tdnn", "resemblyzer"])
    p.add_argument("--emotion_backend", type=str, default="emotion2vec_base",
                    choices=["emotion2vec_base", "emotion2vec_plus", "wav2vec2_ser", "hubert_ser"])
    p.add_argument("--no_cross_attn", action="store_true", default=False,
                    help="Disable cross-attention (saves memory)")
    p.add_argument("--no_adaln", action="store_true", default=False,
                    help="Disable AdaLN conditioning")
    p.add_argument("--no_input_add", action="store_true", default=False,
                    help="Disable input embedding addition")

    # ── Checkpointing ──
    p.add_argument("--checkpoint_dir", type=str, default="ckpts/f5tts_enhanced")
    p.add_argument("--save_per_updates", type=int, default=5000)
    p.add_argument("--last_per_updates", type=int, default=1000)
    p.add_argument("--keep_last_n_checkpoints", type=int, default=5)

    # ── Logging ──
    p.add_argument("--logger", type=str, default=None,
                    choices=[None, "wandb", "tensorboard"])
    p.add_argument("--wandb_project", type=str, default="f5tts-enhanced")
    p.add_argument("--wandb_run_name", type=str, default="enhanced_finetune")

    # ── Misc ──
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--num_workers", type=int, default=4)

    return p.parse_args()


# ── Step 1: Get pretrained checkpoint ──────────────────────────────

def get_pretrain_checkpoint(args) -> str:
    """Download or locate the pretrained F5-TTS checkpoint."""
    if args.pretrain_ckpt and os.path.exists(args.pretrain_ckpt):
        print(f"Using pretrained checkpoint: {args.pretrain_ckpt}")
        return args.pretrain_ckpt

    print("Downloading pretrained F5-TTS checkpoint from HuggingFace...")
    from cached_path import cached_path

    if args.exp_name == "F5TTS_v1_Base":
        url = "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    else:
        url = "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt"

    ckpt_path = str(cached_path(url))
    print(f"Downloaded to: {ckpt_path}")

    # Copy to checkpoint dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    dest = os.path.join(args.checkpoint_dir, f"pretrained_{os.path.basename(ckpt_path)}")
    if not os.path.exists(dest):
        shutil.copy2(ckpt_path, dest)
    return dest


# ── Step 2: Build model ────────────────────────────────────────────

def build_model(args, vocab_size: int, device: str) -> EnhancedCFM:
    """Build the EnhancedCFM model with conditioning modules."""

    # Base architecture config (must match the pretrained checkpoint)
    if args.exp_name == "F5TTS_v1_Base":
        model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2,
            text_dim=512, conv_layers=4,
        )
    else:  # F5TTS_Base
        model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2,
            text_dim=512, text_mask_padding=False,
            conv_layers=4, pe_attn_head=1,
        )

    # Add conditioning params
    model_cfg.update(
        speaker_emb_dim=args.speaker_emb_dim,
        emotion_emb_dim=args.emotion_emb_dim,
        use_adaln_cond=not args.no_adaln,
        use_input_add_cond=not args.no_input_add,
        use_cross_attn_cond=not args.no_cross_attn,
    )

    mel_spec_kwargs = dict(
        n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        n_mel_channels=n_mel_channels, target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    model = EnhancedCFM(
        transformer=EnhancedDiT(
            **model_cfg,
            text_num_embeds=vocab_size,
            mel_dim=n_mel_channels,
        ),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=None,  # set below
    )

    return model


def load_pretrained_weights(model: EnhancedCFM, ckpt_path: str, device: str):
    """Load original F5-TTS weights into enhanced model (strict=False)."""
    print(f"\nLoading pretrained weights from {ckpt_path}")

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file
        checkpoint = load_file(ckpt_path, device="cpu")
        state_dict = checkpoint  # safetensors = flat state dict
    else:
        checkpoint = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        if "ema_model_state_dict" in checkpoint:
            state_dict = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "step"]
            }
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

    # Remove known problematic keys
    for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
        state_dict.pop(key, None)

    # Load with strict=False (new conditioning params will be missing → randomly init)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    new_params = [k for k in missing if "cond_aggregator" in k or "enhanced" in k]
    base_missing = [k for k in missing if k not in new_params]

    print(f"  Loaded: {len(state_dict) - len(unexpected)} parameters from base F5-TTS")
    print(f"  New conditioning params (zero/random init): {len(new_params)}")
    if base_missing:
        print(f"  WARNING — missing base params: {base_missing[:5]}...")
    if unexpected:
        print(f"  Skipped unexpected keys: {len(unexpected)}")

    del checkpoint, state_dict
    gc.collect()
    torch.cuda.empty_cache()


# ── Step 3: Freeze / unfreeze ──────────────────────────────────────

def apply_freeze_strategy(model: EnhancedCFM, freeze_base: bool, unfreeze_top_k: int):
    """Freeze base model and optionally unfreeze top-K blocks."""
    if not freeze_base:
        print("WARNING: Training ALL parameters (freeze_base=False)")
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {total:,}")
        return

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze conditioning aggregator (always)
    cond_agg = model.transformer.cond_aggregator
    for param in cond_agg.parameters():
        param.requires_grad = True

    # Optionally unfreeze top-K DiT blocks
    if unfreeze_top_k > 0:
        blocks = model.transformer.transformer_blocks
        n = len(blocks)
        for i in range(max(0, n - unfreeze_top_k), n):
            for param in blocks[i].parameters():
                param.requires_grad = True
        for param in model.transformer.norm_out.parameters():
            param.requires_grad = True
        for param in model.transformer.proj_out.parameters():
            param.requires_grad = True
        print(f"  Unfroze top-{unfreeze_top_k} DiT blocks + output layers")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"\n  Total params:     {total:>12,}")
    print(f"  Frozen (base):    {frozen:>12,}  ({100*frozen/total:.1f}%)")
    print(f"  Trainable (new):  {trainable:>12,}  ({100*trainable/total:.1f}%)")


# ── Step 4: Load embedding encoders ───────────────────────────────

def load_encoders_for_online_extraction(args, device):
    """Load speaker/emotion encoders (only needed if no precomputed embeddings)."""
    if args.embedding_dir and os.path.exists(args.embedding_dir):
        # Check if embeddings exist
        sample_emb = os.path.join(args.embedding_dir, "0.pt")
        if os.path.exists(sample_emb):
            print("Using precomputed embeddings — encoders NOT loaded (saves GPU memory)")
            return None, None

    print("No precomputed embeddings found — loading encoders for online extraction...")
    print("  WARNING: This is ~10-20x slower than precomputed embeddings!")
    print("  Run prepare_data.py --stage embeddings first for faster training.\n")

    speaker_enc = SpeakerEncoder(
        backend=args.speaker_backend,
        output_dim=args.speaker_emb_dim,
        device=device,
    ).to(device).eval()

    emotion_enc = EmotionEncoder(
        backend=args.emotion_backend,
        output_dim=args.emotion_emb_dim,
        frame_level=True,
        device=device,
    ).to(device).eval()

    # Freeze encoders (they are never trained)
    for p in speaker_enc.parameters():
        p.requires_grad = False
    for p in emotion_enc.parameters():
        p.requires_grad = False

    return speaker_enc, emotion_enc


# ── Step 5: Build dataset & dataloader ─────────────────────────────

def build_dataloader(args, mel_spec_kwargs):
    """Build training dataset and dataloader."""
    from datasets import Dataset as HFDataset_
    from datasets import load_from_disk

    # Load raw dataset
    raw_path = os.path.join(args.dataset_dir, "raw")
    try:
        raw_dataset = load_from_disk(raw_path)
    except:
        raw_dataset = HFDataset_.from_file(os.path.join(args.dataset_dir, "raw.arrow"))

    # Load durations
    import json
    with open(os.path.join(args.dataset_dir, "duration.json"), "r") as f:
        durations = json.load(f)["duration"]

    # Create enhanced dataset
    dataset = EnhancedCustomDataset(
        custom_dataset=raw_dataset,
        durations=durations,
        embedding_cache_dir=args.embedding_dir,
        **mel_spec_kwargs,
    )

    print(f"\nDataset: {len(dataset)} samples")
    print(f"Embedding cache: {args.embedding_dir or 'NONE (online extraction)'}")

    # Build dataloader
    if args.batch_size_type == "frame":
        sampler = SequentialSampler(dataset)
        batch_sampler = DynamicBatchSampler(
            sampler, args.batch_size_per_gpu,
            max_samples=args.max_samples,
            random_seed=args.seed,
        )
        dataloader = DataLoader(
            dataset,
            collate_fn=enhanced_collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            batch_sampler=batch_sampler,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size_per_gpu,
            collate_fn=enhanced_collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    return dataset, dataloader


# ══════════════════════════════════════════════════════════════════
#  MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Accelerator ──
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with=args.logger if args.logger == "wandb" else None,
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=args.grad_accumulation_steps,
    )
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        print("=" * 60)
        print("  F5-TTS Enhanced — Training")
        print("=" * 60)

    # ── 1. Tokenizer ──
    if args.vocab_file:
        vocab_path = args.vocab_file
    else:
        # Check dataset dir for vocab
        ds_vocab = os.path.join(args.dataset_dir, "vocab.txt")
        if os.path.exists(ds_vocab):
            vocab_path = ds_vocab
        else:
            vocab_path = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))

    vocab_char_map, vocab_size = get_tokenizer(vocab_path, "custom")
    if is_main:
        print(f"\nVocab: {vocab_size} tokens from {vocab_path}")

    # ── 2. Build model ──
    model = build_model(args, vocab_size, str(device))
    model.vocab_char_map = vocab_char_map

    # ── 3. Load pretrained weights ──
    ckpt_path = get_pretrain_checkpoint(args)
    load_pretrained_weights(model, ckpt_path, str(device))

    # ── 4. Freeze strategy ──
    if is_main:
        print(f"\nFreeze strategy: freeze_base={args.freeze_base}, unfreeze_top_k={args.unfreeze_top_k}")
    apply_freeze_strategy(model, args.freeze_base, args.unfreeze_top_k)

    # ── 5. EMA (main process only) ──
    if is_main:
        ema_model = EMA(model, include_online_model=False)
        ema_model.to(device)

    # ── 6. Optimizer (only trainable params) ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate)

    # ── 7. Dataset & Dataloader ──
    mel_spec_kwargs = dict(
        target_sample_rate=target_sample_rate,
        n_mel_channels=n_mel_channels,
        hop_length=hop_length,
        n_fft=n_fft,
        win_length=win_length,
        mel_spec_type=mel_spec_type,
    )
    dataset, dataloader = build_dataloader(args, mel_spec_kwargs)

    # ── 8. Load encoders (if needed for online extraction) ──
    speaker_enc, emotion_enc = load_encoders_for_online_extraction(args, str(device))

    # ── 9. Scheduler ──
    warmup_updates = args.num_warmup_updates * accelerator.num_processes
    total_updates = math.ceil(len(dataloader) / args.grad_accumulation_steps) * args.epochs
    decay_updates = max(total_updates - warmup_updates, 1)

    warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
    decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1e-2, total_iters=decay_updates)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, decay_scheduler], milestones=[warmup_updates])

    # ── 10. Prepare with accelerator ──
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    # ── 11. Logger init ──
    if args.logger == "wandb" and is_main:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config={
                "epochs": args.epochs,
                "lr": args.learning_rate,
                "batch_size": args.batch_size_per_gpu,
                "freeze_base": args.freeze_base,
                "unfreeze_top_k": args.unfreeze_top_k,
                "speaker_backend": args.speaker_backend,
                "emotion_backend": args.emotion_backend,
                "trainable_params": sum(p.numel() for p in trainable_params),
            },
            init_kwargs={"wandb": {"name": args.wandb_run_name}},
        )

    # ── 12. Try to resume from existing checkpoint ──
    global_update = 0
    last_ckpt = os.path.join(args.checkpoint_dir, "model_last.pt")
    if os.path.exists(last_ckpt):
        if is_main:
            print(f"\nResuming from {last_ckpt}")
        ckpt = torch.load(last_ckpt, weights_only=True, map_location="cpu")
        if is_main and "ema_model_state_dict" in ckpt:
            ema_model.load_state_dict(ckpt["ema_model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_update = ckpt.get("update", ckpt.get("step", 0))
        del ckpt
        if is_main:
            print(f"  Resumed at update {global_update}")

    # ══════════════════════════════════════════════════════════
    #  TRAINING LOOP
    # ══════════════════════════════════════════════════════════
    if is_main:
        print(f"\n{'=' * 60}")
        print(f"  Starting training: {args.epochs} epochs")
        print(f"  Updates per epoch: ~{math.ceil(len(dataloader)/args.grad_accumulation_steps)}")
        print(f"  Total updates: ~{total_updates}")
        print(f"  Warmup: {args.num_warmup_updates} updates")
        print(f"  GPUs: {accelerator.num_processes}")
        print(f"{'=' * 60}\n")

    for epoch in range(args.epochs):
        model.train()

        if hasattr(dataloader, "batch_sampler") and hasattr(dataloader.batch_sampler, "set_epoch"):
            dataloader.batch_sampler.set_epoch(epoch)

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            disable=not is_main,
            unit="batch",
        )

        epoch_loss = 0.0
        epoch_steps = 0

        for batch in pbar:
            with accelerator.accumulate(model):
                # ── Unpack batch ──
                mel_spec = batch["mel"].permute(0, 2, 1)  # (b, d, t) → (b, t, d)
                text_inputs = batch["text"]
                mel_lengths = batch["mel_lengths"]

                # ── Get embeddings ──
                speaker_emb = None
                emotion_global = None
                emotion_frame = None

                # Option A: precomputed embeddings
                if "speaker_raw" in batch:
                    spk_raw = batch["speaker_raw"].to(device)
                    emo_g_raw = batch.get("emotion_global_raw")
                    emo_f_raw = batch.get("emotion_frame_raw")

                    # Project through trainable heads
                    # (During training, projection is part of the model graph)
                    if speaker_enc is not None:
                        speaker_emb = speaker_enc.project_cached_embedding(spk_raw)
                    else:
                        # If no encoder loaded, use raw as-is
                        # (projection is inside ConditioningAggregator)
                        speaker_emb = spk_raw

                    if emo_g_raw is not None:
                        emo_g_raw = emo_g_raw.to(device)
                        if emotion_enc is not None:
                            emo_result = emotion_enc.project_cached_embeddings(
                                emo_g_raw,
                                emo_f_raw.to(device) if emo_f_raw is not None else None,
                                target_len=mel_spec.shape[1],
                            )
                            emotion_global = emo_result["global"]
                            emotion_frame = emo_result.get("frame")
                        else:
                            emotion_global = emo_g_raw

                # Option B: online extraction
                elif "raw_audio" in batch and speaker_enc is not None:
                    raw_audio = batch["raw_audio"].to(device)
                    sr = batch["sample_rate"]
                    with torch.no_grad():
                        speaker_emb = speaker_enc(raw_audio, sr=sr)
                        emo_result = emotion_enc(raw_audio, sr=sr, target_len=mel_spec.shape[1])
                        emotion_global = emo_result["global"]
                        emotion_frame = emo_result.get("frame")

                # ── Forward pass ──
                loss, _, _ = accelerator.unwrap_model(model)(
                    inp=mel_spec,
                    text=text_inputs,
                    lens=mel_lengths,
                    speaker_emb=speaker_emb,
                    emotion_global=emotion_global,
                    emotion_frame=emotion_frame,
                )

                accelerator.backward(loss)

                if args.max_grad_norm > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # ── EMA update ──
            if accelerator.sync_gradients:
                if is_main:
                    ema_model.update()
                global_update += 1

                epoch_loss += loss.item()
                epoch_steps += 1

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    avg=f"{epoch_loss/epoch_steps:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    upd=global_update,
                )

            # ── Logging ──
            if is_main and args.logger:
                accelerator.log({
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch + 1,
                }, step=global_update)

            # ── Save last checkpoint ──
            if global_update % args.last_per_updates == 0 and accelerator.sync_gradients:
                accelerator.wait_for_everyone()
                if is_main:
                    os.makedirs(args.checkpoint_dir, exist_ok=True)
                    torch.save({
                        "ema_model_state_dict": ema_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "update": global_update,
                        "epoch": epoch,
                    }, os.path.join(args.checkpoint_dir, "model_last.pt"))
                    print(f"\n  → Saved model_last.pt at update {global_update}")

            # ── Save numbered checkpoint ──
            if global_update % args.save_per_updates == 0 and accelerator.sync_gradients:
                accelerator.wait_for_everyone()
                if is_main:
                    save_path = os.path.join(args.checkpoint_dir, f"model_{global_update}.pt")
                    torch.save({
                        "ema_model_state_dict": ema_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "update": global_update,
                        "epoch": epoch,
                    }, save_path)
                    print(f"\n  → Saved {save_path}")

                    # Clean old checkpoints
                    if args.keep_last_n_checkpoints > 0:
                        ckpts = sorted([
                            f for f in os.listdir(args.checkpoint_dir)
                            if f.startswith("model_") and f.endswith(".pt")
                            and f != "model_last.pt" and not f.startswith("pretrained_")
                        ], key=lambda x: int(x.split("_")[1].split(".")[0]))
                        while len(ckpts) > args.keep_last_n_checkpoints:
                            old = ckpts.pop(0)
                            os.remove(os.path.join(args.checkpoint_dir, old))
                            print(f"  Removed old: {old}")

        if is_main:
            print(f"\n  Epoch {epoch+1} done — avg loss: {epoch_loss/max(epoch_steps,1):.4f}\n")

    # ── Final save ──
    accelerator.wait_for_everyone()
    if is_main:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save({
            "ema_model_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "update": global_update,
            "epoch": args.epochs,
        }, os.path.join(args.checkpoint_dir, "model_last.pt"))
        print(f"\nTraining complete! Final checkpoint: {args.checkpoint_dir}/model_last.pt")
        print(f"Total updates: {global_update}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
