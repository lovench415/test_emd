"""
Training Launch Script for F5-TTS Enhanced
=============================================

Three training modes:

1. Quick Start (minimal):
   python train_enhanced.py --pretrain_ckpt model.safetensors --dataset_dir data/

2. Full pipeline (recommended):
   python train_enhanced.py \
       --pretrain_ckpt model.safetensors \
       --dataset_dir data/prepared \
       --embedding_dir data/embeddings \
       --epochs 5 --lr 3e-4

3. Stage 2 (unfreeze top blocks):
   python train_enhanced.py \
       --pretrain_ckpt ckpts/model_last.pt \
       --dataset_dir data/prepared \
       --embedding_dir data/embeddings \
       --unfreeze_top_k 4 --lr 1e-5 --epochs 2
"""

import argparse
import os
import shutil
import sys
from importlib.resources import files

import torch
from cached_path import cached_path

from src.f5_tts.model.backbones.enhanced_dit import EnhancedDiT
from src.f5_tts.model.enhanced_cfm import EnhancedCFM
from src.f5_tts.model.enhanced_dataset import EnhancedCustomDataset, enhanced_collate_fn
from src.f5_tts.model.enhanced_trainer import EnhancedTrainer
from src.f5_tts.model.speaker_encoder import SpeakerEncoder
from src.f5_tts.model.emotion_encoder import EmotionEncoder
from f5_tts.model.utils import get_tokenizer


# =====================================================================
# Audio / Mel settings (must match original F5-TTS)
# =====================================================================

TARGET_SAMPLE_RATE = 24000
N_MEL_CHANNELS = 100
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_FFT = 1024
MEL_SPEC_TYPE = "vocos"


def parse_args():
    p = argparse.ArgumentParser(description="F5-TTS Enhanced — Training")

    # ── Data ──
    p.add_argument("--dataset_dir", type=str, required=True,
                    help="Path to prepared dataset (contains raw/ and duration.json)")
    p.add_argument("--embedding_dir", type=str, default=None,
                    help="Path to precomputed embeddings. If None, extracts online (slow)")
    p.add_argument("--vocab_file", type=str, default=None,
                    help="Path to vocab.txt. If None, uses dataset_dir/vocab.txt or default F5-TTS vocab")

    # ── Pretrained checkpoint ──
    p.add_argument("--pretrain_ckpt", type=str, default=None,
                    help="Path to original F5-TTS checkpoint (.pt or .safetensors). "
                         "If None, downloads F5TTS_v1_Base from HuggingFace")

    # ── Model architecture ──
    p.add_argument("--speaker_emb_dim", type=int, default=512)
    p.add_argument("--emotion_emb_dim", type=int, default=512)
    p.add_argument("--speaker_backend", type=str, default="wavlm_sv",
                    choices=["wavlm_sv", "ecapa_tdnn", "resemblyzer"])
    p.add_argument("--emotion_backend", type=str, default="emotion2vec_base",
                    choices=["emotion2vec_base", "emotion2vec_plus", "wav2vec2_ser", "hubert_ser"])
    p.add_argument("--use_adaln", action="store_true", default=True)
    p.add_argument("--use_input_add", action="store_true", default=True)
    p.add_argument("--use_cross_attn", action="store_true", default=True)
    p.add_argument("--no_frame_level", action="store_true", default=False,
                    help="Disable frame-level emotion (saves memory)")

    # ── Freeze strategy ──
    p.add_argument("--freeze_base", action="store_true", default=True,
                    help="Freeze original F5-TTS weights (default: True)")
    p.add_argument("--no_freeze", action="store_true", default=False,
                    help="Don't freeze base (full finetuning, needs more data)")
    p.add_argument("--unfreeze_top_k", type=int, default=0,
                    help="Unfreeze top-K DiT blocks (0 = fully frozen)")

    # ── Training ──
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4,
                    help="Learning rate (3e-4 for frozen base, 1e-5 for unfrozen)")
    p.add_argument("--batch_size_per_gpu", type=int, default=19200,
                    help="Batch size in frames per GPU")
    p.add_argument("--batch_size_type", type=str, default="frame",
                    choices=["frame", "sample"])
    p.add_argument("--max_samples", type=int, default=32)
    p.add_argument("--grad_accumulation", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_updates", type=int, default=2000)

    # ── Checkpointing ──
    p.add_argument("--save_dir", type=str, default="ckpts/f5tts_enhanced")
    p.add_argument("--save_per_updates", type=int, default=5000)
    p.add_argument("--last_per_updates", type=int, default=1000)
    p.add_argument("--keep_last_n", type=int, default=5)

    # ── Logging ──
    p.add_argument("--logger", type=str, default=None,
                    choices=[None, "wandb", "tensorboard"])
    p.add_argument("--wandb_project", type=str, default="F5TTS-Enhanced")
    p.add_argument("--wandb_run_name", type=str, default="enhanced_finetune")

    # ── Device ──
    p.add_argument("--device", type=str, default="cuda")

    return p.parse_args()


def main():
    args = parse_args()

    if args.no_freeze:
        args.freeze_base = False

    print("=" * 60)
    print("  F5-TTS Enhanced — Training")
    print("=" * 60)
    print(f"  Dataset:          {args.dataset_dir}")
    print(f"  Embeddings:       {args.embedding_dir or 'online (slow)'}")
    print(f"  Pretrained:       {args.pretrain_ckpt or 'HuggingFace F5TTS_v1_Base'}")
    print(f"  Freeze base:      {args.freeze_base}")
    print(f"  Unfreeze top-K:   {args.unfreeze_top_k}")
    print(f"  LR:               {args.lr}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Speaker backend:  {args.speaker_backend}")
    print(f"  Emotion backend:  {args.emotion_backend}")
    print("=" * 60)

    # =================================================================
    # 1. Download pretrained checkpoint if not provided
    # =================================================================

    if args.pretrain_ckpt is None:
        print("\nDownloading F5TTS_v1_Base from HuggingFace...")
        args.pretrain_ckpt = str(
            cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors")
        )
        print(f"  → {args.pretrain_ckpt}")

    # Copy pretrained checkpoint to save_dir
    os.makedirs(args.save_dir, exist_ok=True)
    pretrain_name = "pretrained_" + os.path.basename(args.pretrain_ckpt)
    pretrain_dest = os.path.join(args.save_dir, pretrain_name)
    if not os.path.exists(pretrain_dest):
        shutil.copy2(args.pretrain_ckpt, pretrain_dest)
        print(f"  Copied pretrained checkpoint → {pretrain_dest}")

    # =================================================================
    # 2. Set up tokenizer
    # =================================================================

    if args.vocab_file and os.path.exists(args.vocab_file):
        vocab_file = args.vocab_file
        tokenizer_type = "custom"
    elif os.path.exists(os.path.join(args.dataset_dir, "vocab.txt")):
        vocab_file = os.path.join(args.dataset_dir, "vocab.txt")
        tokenizer_type = "custom"
    else:
        vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
        tokenizer_type = "custom"

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer_type)
    print(f"\nVocabulary: {vocab_size} tokens from {vocab_file}")

    # =================================================================
    # 3. Build Enhanced Model
    # =================================================================

    print("\nBuilding EnhancedDiT model...")

    # Base architecture (same as F5TTS_Base for weight compatibility)
    model_cfg = dict(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        text_mask_padding=False,
        conv_layers=4,
        pe_attn_head=1,
        attn_backend="torch",
        attn_mask_enabled=False,
        # Conditioning
        speaker_emb_dim=args.speaker_emb_dim,
        emotion_emb_dim=args.emotion_emb_dim,
        use_adaln_cond=args.use_adaln,
        use_input_add_cond=args.use_input_add,
        use_cross_attn_cond=args.use_cross_attn,
    )

    mel_spec_kwargs = dict(
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mel_channels=N_MEL_CHANNELS,
        target_sample_rate=TARGET_SAMPLE_RATE,
        mel_spec_type=MEL_SPEC_TYPE,
    )

    model = EnhancedCFM(
        transformer=EnhancedDiT(
            **model_cfg,
            text_num_embeds=vocab_size,
            mel_dim=N_MEL_CHANNELS,
        ),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
        speaker_drop_prob=0.1,
        emotion_drop_prob=0.1,
    )

    # =================================================================
    # 4. Load pretrained weights (strict=False)
    # =================================================================

    print(f"\nLoading pretrained weights from {args.pretrain_ckpt}...")

    ckpt_type = args.pretrain_ckpt.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file
        checkpoint = load_file(args.pretrain_ckpt, device="cpu")
        state_dict = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint.items()
            if k not in ["initted", "step"]
        }
    else:
        checkpoint = torch.load(args.pretrain_ckpt, map_location="cpu", weights_only=True)
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

    # Remove potentially problematic keys
    for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
        state_dict.pop(key, None)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    n_loaded = len(state_dict) - len(unexpected)
    print(f"  Loaded:     {n_loaded} parameters (original F5-TTS)")
    print(f"  New (init):  {len(missing)} parameters (conditioning modules)")
    if unexpected:
        print(f"  Unexpected:  {len(unexpected)} (skipped)")

    del checkpoint, state_dict
    torch.cuda.empty_cache()

    # =================================================================
    # 5. Load embedding extractors (for online or projection)
    # =================================================================

    speaker_encoder = None
    emotion_encoder = None

    if args.embedding_dir:
        # Precomputed mode: need projection heads only
        print(f"\nUsing precomputed embeddings from {args.embedding_dir}")
        # Load lightweight encoders just for projection heads
        speaker_encoder = SpeakerEncoder(
            backend=args.speaker_backend,
            output_dim=args.speaker_emb_dim,
            device=args.device,
        )
        emotion_encoder = EmotionEncoder(
            backend=args.emotion_backend,
            output_dim=args.emotion_emb_dim,
            frame_level=not args.no_frame_level,
            device=args.device,
        )
    else:
        # Online mode: full encoders
        print(f"\nLoading full embedding extractors for online extraction...")
        print(f"  WARNING: Online extraction is ~15-20x slower than precomputed.")
        print(f"  Consider running prepare_data.py --stage embeddings first.")
        speaker_encoder = SpeakerEncoder(
            backend=args.speaker_backend,
            output_dim=args.speaker_emb_dim,
            device=args.device,
        )
        emotion_encoder = EmotionEncoder(
            backend=args.emotion_backend,
            output_dim=args.emotion_emb_dim,
            frame_level=not args.no_frame_level,
            device=args.device,
        )

    # =================================================================
    # 6. Load dataset
    # =================================================================

    print(f"\nLoading dataset from {args.dataset_dir}...")

    import json
    from datasets import Dataset as HFDataset
    from datasets import load_from_disk

    try:
        raw_dataset = load_from_disk(os.path.join(args.dataset_dir, "raw"))
    except:
        from datasets import Dataset as D
        raw_dataset = D.from_file(os.path.join(args.dataset_dir, "raw.arrow"))

    duration_path = os.path.join(args.dataset_dir, "duration.json")
    with open(duration_path, "r") as f:
        durations = json.load(f)["duration"]

    train_dataset = EnhancedCustomDataset(
        custom_dataset=raw_dataset,
        durations=durations,
        target_sample_rate=TARGET_SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_mel_channels=N_MEL_CHANNELS,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        mel_spec_type=MEL_SPEC_TYPE,
        embedding_cache_dir=args.embedding_dir,
    )

    print(f"  Samples: {len(train_dataset)}")
    print(f"  Duration: {sum(durations)/3600:.1f} hours")

    # =================================================================
    # 7. Create trainer and start training
    # =================================================================

    print(f"\nInitializing trainer...")

    trainer = EnhancedTrainer(
        model=model,
        epochs=args.epochs,
        learning_rate=args.lr,
        num_warmup_updates=args.warmup_updates,
        save_per_updates=args.save_per_updates,
        keep_last_n_checkpoints=args.keep_last_n,
        checkpoint_path=args.save_dir,
        batch_size_per_gpu=args.batch_size_per_gpu,
        batch_size_type=args.batch_size_type,
        max_samples=args.max_samples,
        grad_accumulation_steps=args.grad_accumulation,
        max_grad_norm=args.max_grad_norm,
        logger=args.logger,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        last_per_updates=args.last_per_updates,
        # PEFT strategy
        freeze_base=args.freeze_base,
        unfreeze_top_k_blocks=args.unfreeze_top_k,
        # Encoders
        speaker_encoder=speaker_encoder,
        emotion_encoder=emotion_encoder,
    )

    print(f"\n{'='*60}")
    print(f"  STARTING TRAINING")
    print(f"{'='*60}\n")

    trainer.train(
        train_dataset=train_dataset,
        resumable_with_seed=666,
    )


if __name__ == "__main__":
    main()
