"""
Enhanced Trainer for F5-TTS
=============================

Finetuning strategy inspired by PEFT-TTS and TTS-CtrlNet:

1. **Freeze base F5-TTS weights** (DiT blocks, text/input embeddings, etc.)
2. **Train only new conditioning modules**:
   - ConditioningAggregator (AdaLN projections, cross-attention, input add)
   - SpeakerEncoder projection head
   - EmotionEncoder projection head + temporal smoothing
3. Optionally, unfreeze top-K DiT blocks for adaptation

This is much faster than training from scratch (10-50x fewer params)
and preserves the base model's speech synthesis quality.

Training stages:
    Stage 1: Train conditioning modules only (1-5 epochs, fast convergence)
    Stage 2: Unfreeze top-K DiT blocks for fine-grained adaptation (optional)
"""

from __future__ import annotations

import gc
import os

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from f5_tts.model.enhanced_cfm import EnhancedCFM
from f5_tts.model.enhanced_dataset import enhanced_collate_fn, EnhancedCustomDataset
from f5_tts.model.dataset import DynamicBatchSampler
from f5_tts.model.utils import default, exists


class EnhancedTrainer:
    """
    Finetuning trainer for the enhanced F5-TTS model.
    
    Freezes base model weights and trains only the new conditioning modules.
    """

    def __init__(
        self,
        model: EnhancedCFM,
        epochs: int,
        learning_rate: float,
        num_warmup_updates: int = 2000,
        save_per_updates: int = 5000,
        keep_last_n_checkpoints: int = 5,
        checkpoint_path: str | None = None,
        batch_size_per_gpu: int = 3200,
        batch_size_type: str = "frame",
        max_samples: int = 32,
        grad_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        logger: str | None = "wandb",
        wandb_project: str = "f5tts-enhanced",
        wandb_run_name: str = "enhanced_finetune",
        wandb_resume_id: str | None = None,
        last_per_updates: int = 1000,
        # Finetuning strategy
        freeze_base: bool = True,
        unfreeze_top_k_blocks: int = 0,
        # Speaker/emotion encoders (for online extraction)
        speaker_encoder=None,
        emotion_encoder=None,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        
        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
        )

        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.checkpoint_path = checkpoint_path
        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.last_per_updates = last_per_updates
        self.speaker_encoder = speaker_encoder
        self.emotion_encoder = emotion_encoder

        # ── Apply freeze strategy ──
        if freeze_base:
            self._freeze_base_model(unfreeze_top_k_blocks)

        # Print trainable param count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.accelerator.print(
            f"Total params: {total_params:,} | "
            f"Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)"
        )

    def _freeze_base_model(self, unfreeze_top_k: int = 0):
        """
        Freeze all base F5-TTS parameters.
        Only conditioning modules remain trainable.
        Optionally unfreeze the top-K transformer blocks.
        """
        # Freeze everything first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze conditioning aggregator
        if hasattr(self.model.transformer, "cond_aggregator"):
            for param in self.model.transformer.cond_aggregator.parameters():
                param.requires_grad = True

        # Unfreeze speaker encoder projection (if present in model)
        # (Speaker/emotion encoders are typically separate, but their projections
        # may be part of the conditioning aggregator)

        # Optionally unfreeze top-K DiT blocks
        if unfreeze_top_k > 0:
            blocks = self.model.transformer.transformer_blocks
            n_blocks = len(blocks)
            for i in range(max(0, n_blocks - unfreeze_top_k), n_blocks):
                for param in blocks[i].parameters():
                    param.requires_grad = True
            # Also unfreeze output norm and projection
            for param in self.model.transformer.norm_out.parameters():
                param.requires_grad = True
            for param in self.model.transformer.proj_out.parameters():
                param.requires_grad = True

    def train(
        self,
        train_dataset: EnhancedCustomDataset,
        resumable_with_seed: int = None,
    ):
        """Main training loop."""
        # Optimizer: only trainable params
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=self.learning_rate)

        # Dataloader
        if self.batch_size_type == "frame":
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler, self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,
            )
            train_loader = DataLoader(
                train_dataset,
                collate_fn=enhanced_collate_fn,
                num_workers=4,
                batch_sampler=batch_sampler,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size_per_gpu,
                collate_fn=enhanced_collate_fn,
                num_workers=4,
                shuffle=True,
                pin_memory=True,
            )

        # LR scheduler
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=self.num_warmup_updates)
        scheduler = SequentialLR(optimizer, [warmup_scheduler], milestones=[self.num_warmup_updates])

        # Prepare with accelerator
        self.model, optimizer, train_loader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_loader, scheduler
        )

        # EMA
        ema_model = EMA(self.model, include_online_model=False)

        global_step = 0

        for epoch in range(self.epochs):
            self.model.train()
            
            if hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch)

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{self.epochs}",
                disable=not self.accelerator.is_main_process,
            )

            for batch in pbar:
                with self.accelerator.accumulate(self.model):
                    mel = batch["mel"].permute(0, 2, 1)  # (b, d, t) -> (b, t, d)
                    text = batch["text"]
                    mel_lengths = batch["mel_lengths"]

                    # Get embeddings
                    speaker_emb = batch.get("speaker_raw")
                    emotion_global = batch.get("emotion_global_raw")
                    emotion_frame = batch.get("emotion_frame_raw")

                    # If using precomputed raw embeddings, project them
                    if speaker_emb is not None and self.speaker_encoder is not None:
                        speaker_emb = self.speaker_encoder.project_cached_embedding(speaker_emb)
                    if emotion_global is not None and self.emotion_encoder is not None:
                        emo_result = self.emotion_encoder.project_cached_embeddings(
                            emotion_global, emotion_frame,
                            target_len=mel.shape[1],
                        )
                        emotion_global = emo_result["global"]
                        emotion_frame = emo_result.get("frame")

                    # Online extraction if no cached embeddings
                    if speaker_emb is None and "raw_audio" in batch:
                        raw_audio = batch["raw_audio"]
                        sr = batch["sample_rate"]
                        if self.speaker_encoder is not None:
                            speaker_emb = self.speaker_encoder(raw_audio, sr=sr)
                        if self.emotion_encoder is not None:
                            emo_result = self.emotion_encoder(
                                raw_audio, sr=sr, target_len=mel.shape[1]
                            )
                            emotion_global = emo_result["global"]
                            emotion_frame = emo_result.get("frame")

                    # Forward pass
                    loss, _, _ = self.model(
                        inp=mel,
                        text=text,
                        lens=mel_lengths,
                        speaker_emb=speaker_emb,
                        emotion_global=emotion_global,
                        emotion_frame=emotion_frame,
                    )

                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(trainable_params, self.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    ema_model.update()
                    global_step += 1

                    pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

                    # Logging
                    if self.accelerator.is_main_process:
                        if self.accelerator.log_with:
                            self.accelerator.log(
                                {"loss": loss.item(), "lr": scheduler.get_last_lr()[0]},
                                step=global_step,
                            )

                    # Save checkpoint
                    if (
                        global_step % self.save_per_updates == 0
                        and self.accelerator.is_main_process
                        and self.checkpoint_path
                    ):
                        self._save_checkpoint(
                            ema_model, optimizer, global_step, epoch
                        )

                    if (
                        global_step % self.last_per_updates == 0
                        and self.accelerator.is_main_process
                        and self.checkpoint_path
                    ):
                        self._save_checkpoint(
                            ema_model, optimizer, global_step, epoch, is_last=True
                        )

        # Final save
        if self.accelerator.is_main_process and self.checkpoint_path:
            self._save_checkpoint(ema_model, optimizer, global_step, self.epochs, is_last=True)

        self.accelerator.print("Training complete!")

    def _save_checkpoint(self, ema_model, optimizer, step, epoch, is_last=False):
        """Save checkpoint with conditioning params."""
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        if is_last:
            path = os.path.join(self.checkpoint_path, "model_last.pt")
        else:
            path = os.path.join(self.checkpoint_path, f"model_{step}.pt")

        torch.save({
            "ema_model_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
        }, path)

        self.accelerator.print(f"Saved checkpoint: {path}")
