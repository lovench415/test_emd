#!/usr/bin/env python3
"""
F5-TTS Enhanced — Full Training Pipeline
==========================================

Полный цикл обучения:
  1. Загрузка базовой F5-TTS + PEFT адаптеры (freeze base, train ~2% params)
  2. Загрузка emotion2vec (pretrained, frozen) + прогрев кеша
  3. Flow matching training loop
  4. Периодическая evaluation + сохранение адаптеров

Flow Matching Loss:
  Дано: x₁ = clean mel, x₀ ~ N(0,1)
  Интерполяция: xₜ = (1-t)·x₀ + t·x₁
  Цель: модель предсказывает v = x₁ - x₀
  Loss: MSE(v_pred, v_target) с маской по реальным фреймам

Запуск:
    # Single GPU
    python scripts/finetune_enhanced.py --config configs/finetune_crosslang_emotion.yaml

    # Multi GPU (accelerate)
    accelerate launch scripts/finetune_enhanced.py --config configs/finetune_crosslang_emotion.yaml

    # Resume
    python scripts/finetune_enhanced.py --config configs/finetune_crosslang_emotion.yaml \
        --resume ckpts/enhanced/step_50000
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import yaml

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from f5_tts_enhanced.model.emotion_extractor import Emotion2vecExtractor
from f5_tts_enhanced.model.adapters import (
    get_trainable_params,
    save_adapters,
    load_adapters,
)
from f5_tts_enhanced.data.dataset import create_dataloader, lang_to_id


# =========================================================================
# Flow Matching Utilities
# =========================================================================

class FlowMatchingScheduler:
    """
    Conditional Flow Matching (OT-CFM) scheduler.
    Реализует интерполяцию и расчёт loss для flow matching.
    """

    def __init__(self, sigma_min: float = 1e-4):
        self.sigma_min = sigma_min

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """t ~ U(0, 1)."""
        return torch.rand(batch_size, device=device)

    def interpolate(
        self,
        x_0: torch.Tensor,    # noise ~ N(0,1)
        x_1: torch.Tensor,    # clean mel
        t: torch.Tensor,      # timestep (B,)
    ) -> tuple:
        """
        OT-CFM interpolation.

        x_t = (1-t)·x_0 + t·x_1
        v_target = x_1 - x_0  (target velocity)

        Args:
            x_0: (B, C, T) — noise
            x_1: (B, C, T) — clean mel
            t: (B,) — timesteps

        Returns:
            x_t: (B, C, T) — noisy input
            v_target: (B, C, T) — target velocity
        """
        t = t[:, None, None]  # (B, 1, 1) for broadcasting
        x_t = (1 - t) * x_0 + t * x_1
        v_target = x_1 - x_0
        return x_t, v_target

    def compute_loss(
        self,
        v_pred: torch.Tensor,     # (B, C, T) — model prediction
        v_target: torch.Tensor,   # (B, C, T) — target velocity
        mask: torch.Tensor,       # (B, T) — True for valid frames
    ) -> torch.Tensor:
        """
        Masked MSE loss.
        Только по реальным фреймам (исключая padding).
        """
        # mask: (B, T) → (B, 1, T)
        mask = mask.unsqueeze(1).float()

        # Per-element MSE
        loss = F.mse_loss(v_pred, v_target, reduction="none")  # (B, C, T)

        # Mask and average
        loss = (loss * mask).sum() / (mask.sum() * v_pred.shape[1] + 1e-8)

        return loss


# =========================================================================
# Trainer
# =========================================================================

class Trainer:
    """
    Полный тренер для F5-TTS Enhanced.

    Этапы каждого шага:
    1. Загрузить батч mel + text + language
    2. Извлечь emotion embeddings (из кеша после warmup)
    3. Sample noise + timestep
    4. Интерполировать x_t
    5. Forward: model(x_t, cond, text, t, emotion, lang) → v_pred
    6. Loss = masked MSE(v_pred, v_target)
    7. Backward + optimizer step
    """

    def __init__(self, config_path: str, resume_from: str = None):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        # Dirs
        ckpt_cfg = self.config.get("checkpoint", {})
        self.save_dir = Path(ckpt_cfg.get("save_dir", "ckpts/enhanced"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        log_cfg = self.config.get("logging", {})
        self.log_dir = Path(log_cfg.get("log_dir", "logs/crosslang_emotion"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Build components
        print("=" * 60)
        print("  F5-TTS Enhanced Training")
        print("=" * 60)

        self.model = self._build_model()
        self.emotion_extractor = self._build_emotion_extractor()
        self.flow = FlowMatchingScheduler()
        self.train_loader, self.val_loader = self._build_dataloaders()
        self.optimizer, self.scheduler = self._build_optimizer()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp())

        # Logging
        self.writer = self._build_logger()

        # Resume
        if resume_from:
            self._resume(resume_from)
        elif self.config.get("checkpoint", {}).get("resume_from"):
            self._resume(self.config["checkpoint"]["resume_from"])

        self._print_summary()

    # ----- Build components -----

    def _build_model(self) -> nn.Module:
        """Загрузить базовую F5-TTS + создать EnhancedF5TTS wrapper."""
        from f5_tts_enhanced.model.enhanced_dit import EnhancedF5TTS

        base_cfg = self.config.get("base_model", {})
        model_cfg = self.config.get("model", {}).get("arch", {})
        adapter_cfg = self.config.get("adapters", {})
        emo_cfg = self.config.get("emotion", {})

        # Загрузка базовой модели F5-TTS
        try:
            from f5_tts.model import DiT
            from f5_tts.model.utils import get_tokenizer

            vocab_path = base_cfg.get("vocab", "")
            tokenizer = "custom" if vocab_path else "pinyin"

            base_model = DiT(
                dim=model_cfg.get("dim", 1024),
                depth=model_cfg.get("depth", 22),
                heads=model_cfg.get("heads", 16),
                ff_mult=model_cfg.get("ff_mult", 2),
                text_dim=model_cfg.get("text_dim", 512),
                conv_layers=model_cfg.get("conv_layers", 4),
            )

            # Загрузить веса
            ckpt_path = base_cfg.get("checkpoint", "")
            if ckpt_path:
                print(f"[Model] Loading base F5-TTS from {ckpt_path}")
                if ckpt_path.startswith("hf://"):
                    from huggingface_hub import hf_hub_download
                    parts = ckpt_path.replace("hf://", "").split("/")
                    repo = "/".join(parts[:2])
                    filename = "/".join(parts[2:])
                    local_path = hf_hub_download(repo_id=repo, filename=filename)
                else:
                    local_path = ckpt_path

                if local_path.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state = load_file(local_path)
                else:
                    state = torch.load(local_path, map_location="cpu")
                    if "model_state_dict" in state:
                        state = state["model_state_dict"]
                    elif "ema_model_state_dict" in state:
                        state = state["ema_model_state_dict"]

                base_model.load_state_dict(state, strict=False)
                print(f"[Model] Base model loaded")

        except ImportError:
            print("[WARNING] f5_tts not installed, using placeholder")
            base_model = nn.Linear(100, 100)  # placeholder

        # Wrap with enhanced model
        lora_cfg = adapter_cfg.get("lora", {})
        emo_ext_cfg = emo_cfg.get("extractor", {})

        enhanced = EnhancedF5TTS(
            base_model=base_model,
            emotion_dim=emo_ext_cfg.get("emotion_dim", 768),
            emotion_mode=emo_cfg.get("conditioning", {}).get("mode", "adaln"),
            lora_rank=lora_cfg.get("rank", 16),
            lora_alpha=lora_cfg.get("alpha", 16.0),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            cond_adapter_compression=adapter_cfg.get("conditioning_adapter", {}).get("compression_factor", 0.25),
            prompt_drop_path=adapter_cfg.get("prompt_adapter", {}).get("drop_path_rate", 0.3),
            add_language_emb=adapter_cfg.get("language_embedding", {}).get("enabled", True),
            freeze_base=adapter_cfg.get("freeze_base", True),
            num_dit_blocks=model_cfg.get("depth", 22),
            hidden_dim=model_cfg.get("dim", 1024),
        )

        # Register LoRA hooks
        enhanced.register_lora_hooks()

        enhanced = enhanced.to(self.device)
        return enhanced

    def _build_emotion_extractor(self) -> Emotion2vecExtractor:
        """Загрузить emotion2vec с кешем."""
        emo_cfg = self.config.get("emotion", {}).get("extractor", {})

        extractor = Emotion2vecExtractor(
            model_size=emo_cfg.get("model_size", "large"),
            emotion_dim=emo_cfg.get("emotion_dim", 768),
            hub=emo_cfg.get("hub", "hf"),
            cache_dir=str(self.save_dir / "emotion_cache"),
            enable_disk_cache=True,
            enable_memory_cache=True,
        )
        extractor.load_model()

        # Freeze (no grad ever)
        for p in extractor.parameters():
            p.requires_grad = False
        extractor.eval()

        return extractor

    def _build_dataloaders(self):
        """Создать train + val DataLoader."""
        ds_cfg = self.config.get("datasets", {})
        mel_cfg = self.config.get("model", {}).get("mel_spec", {})
        base_cfg = self.config.get("base_model", {})

        # Paths from config
        data_root = ds_cfg.get("data_root", "/data/prepared")
        metadata_train = ds_cfg.get("metadata_train", os.path.join(data_root, "metadata.csv"))
        metadata_val = ds_cfg.get("metadata_val", os.path.join(data_root, "metadata_val.csv"))

        vocab_path = base_cfg.get("vocab", os.path.join(data_root, "vocab.txt"))

        # Resolve hf:// vocab path
        if vocab_path.startswith("hf://"):
            from huggingface_hub import hf_hub_download
            parts = vocab_path.replace("hf://", "").split("/")
            repo = "/".join(parts[:2])
            filename = "/".join(parts[2:])
            vocab_path = hf_hub_download(repo_id=repo, filename=filename)

        common_kwargs = dict(
            data_root=data_root,
            vocab_path=vocab_path,
            target_sr=mel_cfg.get("target_sample_rate", 24000),
            n_mels=mel_cfg.get("n_mel_channels", 100),
            hop_length=mel_cfg.get("hop_length", 256),
            max_duration=ds_cfg.get("max_duration", 30.0),
        )

        max_frames = ds_cfg.get("batch_size_per_gpu", 3200)
        max_samples = ds_cfg.get("max_samples", 64)
        num_workers = ds_cfg.get("num_workers", 4)

        print(f"[Data] train: {metadata_train}")
        print(f"[Data] val:   {metadata_val}")
        print(f"[Data] vocab: {vocab_path}")
        print(f"[Data] root:  {data_root}")

        train_ds, train_dl = create_dataloader(
            metadata_path=metadata_train,
            max_frames_per_batch=max_frames,
            max_samples_per_batch=max_samples,
            num_workers=num_workers,
            shuffle=True,
            **common_kwargs,
        )

        val_ds, val_dl = None, None
        if os.path.exists(metadata_val):
            val_ds, val_dl = create_dataloader(
                metadata_path=metadata_val,
                max_frames_per_batch=max_frames,
                max_samples_per_batch=max_samples,
                num_workers=num_workers,
                shuffle=False,
                **common_kwargs,
            )

        # Warmup emotion cache
        print("[Trainer] Warming up emotion cache...")
        all_audio_paths = [s["audio_path"] for s in train_ds.samples]
        self.emotion_extractor.warmup_cache(all_audio_paths)
        self.emotion_extractor.print_cache_stats()

        return train_dl, val_dl

    def _build_optimizer(self):
        """AdamW only on trainable params + cosine scheduler."""
        optim_cfg = self.config.get("optim", {})
        lr = optim_cfg.get("learning_rate", 1e-5)
        wd = optim_cfg.get("weight_decay", 0.01)
        max_steps = optim_cfg.get("max_steps", 150000)
        warmup = optim_cfg.get("num_warmup_updates", 2000)

        trainable = get_trainable_params(self.model)
        print(f"[Optimizer] {len(trainable)} parameter groups, lr={lr}, wd={wd}")

        if optim_cfg.get("bnb_optimizer", False):
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(trainable, lr=lr, weight_decay=wd)
                print("[Optimizer] Using 8-bit AdamW (bitsandbytes)")
            except ImportError:
                optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
        else:
            optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)

        # Cosine with warm-up
        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, max_steps - warmup)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return optimizer, scheduler

    def _use_amp(self) -> bool:
        mp = self.config.get("optim", {}).get("mixed_precision", "bf16")
        return mp in ("fp16", "bf16") and torch.cuda.is_available()

    def _amp_dtype(self):
        mp = self.config.get("optim", {}).get("mixed_precision", "bf16")
        if mp == "bf16" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _build_logger(self):
        """TensorBoard or WandB."""
        logger_type = self.config.get("logging", {}).get("logger", "tensorboard")
        if logger_type == "wandb":
            try:
                import wandb
                wandb.init(project="f5tts-enhanced", config=self.config, dir=str(self.log_dir))
                return wandb
            except ImportError:
                print("[Logger] wandb not installed, falling back to tensorboard")

        try:
            from torch.utils.tensorboard import SummaryWriter
            return SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:
            print("[Logger] tensorboard not installed, logging disabled")
            return None

    # ----- Train step -----

    @torch.no_grad()
    def _extract_emotion_batch(self, audio_paths: list) -> torch.Tensor:
        """Извлечь emotion embeddings для батча (из кеша после warmup)."""
        return self.emotion_extractor.get_batch_embeddings(audio_paths).to(self.device)

    def _make_cond_mel(self, mel: torch.Tensor, mel_lengths: torch.Tensor) -> torch.Tensor:
        """
        Создать condition mel: маскируем часть mel, оставляя prompt.
        F5-TTS style: первые ~3 секунды = prompt (condition), остальное = target.

        Args:
            mel: (B, C, T)
            mel_lengths: (B,) — реальные длины

        Returns:
            cond: (B, C, T) — condition mel (нули где нет prompt'а)
        """
        B, C, T = mel.shape
        cond = torch.zeros_like(mel)

        for i in range(B):
            L = mel_lengths[i].item()
            # Prompt = первые 30% (min 10, max 300 frames ≈ 3sec)
            prompt_len = min(max(int(L * 0.3), 10), min(300, L))
            cond[i, :, :prompt_len] = mel[i, :, :prompt_len]

        return cond

    def train_step(self, batch: dict) -> dict:
        """
        Один шаг обучения.

        Returns:
            dict с loss и метриками
        """
        self.model.train()

        mel = batch["mel"].to(self.device)              # (B, C, T)
        text_ids = batch["text_ids"].to(self.device)     # (B, L)
        mel_lengths = batch["mel_lengths"].to(self.device)
        lang_ids = batch["lang_ids"].to(self.device)     # (B,)
        mask = batch["mask"].to(self.device)              # (B, T)
        audio_paths = batch["audio_paths"]                # list[str]

        B, C, T = mel.shape

        # 1. Emotion embeddings (из кеша — <1ms)
        emotion_emb = self._extract_emotion_batch(audio_paths)  # (B, emo_dim)

        # 2. Condition mel (prompt)
        cond = self._make_cond_mel(mel, mel_lengths)  # (B, C, T)

        # 3. Flow matching: sample noise and timestep
        x_0 = torch.randn_like(mel)  # (B, C, T)
        t = self.flow.sample_timesteps(B, self.device)  # (B,)
        x_t, v_target = self.flow.interpolate(x_0, mel, t)  # (B, C, T) each

        # 4. Forward pass
        with torch.amp.autocast("cuda", dtype=self._amp_dtype(), enabled=self._use_amp()):
            v_pred = self.model(
                x=x_t,           # noisy mel
                cond=cond,        # condition mel (prompt)
                text=text_ids,    # text token ids
                time=t,           # diffusion timestep
                emotion_emb=emotion_emb,
                lang_id=lang_ids,
                mask=mask,
            )

            # 5. Loss
            loss = self.flow.compute_loss(v_pred, v_target, mask)

        return {"loss": loss, "batch_size": B}

    # ----- Validation -----

    @torch.no_grad()
    def validate(self) -> float:
        """Прогнать validation set, вернуть средний loss."""
        if self.val_loader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        total_frames = 0

        for batch in self.val_loader:
            mel = batch["mel"].to(self.device)
            text_ids = batch["text_ids"].to(self.device)
            mel_lengths = batch["mel_lengths"].to(self.device)
            lang_ids = batch["lang_ids"].to(self.device)
            mask = batch["mask"].to(self.device)
            audio_paths = batch["audio_paths"]

            B, C, T = mel.shape
            emotion_emb = self._extract_emotion_batch(audio_paths)
            cond = self._make_cond_mel(mel, mel_lengths)
            x_0 = torch.randn_like(mel)
            t = self.flow.sample_timesteps(B, self.device)
            x_t, v_target = self.flow.interpolate(x_0, mel, t)

            with torch.amp.autocast("cuda", dtype=self._amp_dtype(), enabled=self._use_amp()):
                v_pred = self.model(
                    x=x_t, cond=cond, text=text_ids, time=t,
                    emotion_emb=emotion_emb, lang_id=lang_ids, mask=mask,
                )
                loss = self.flow.compute_loss(v_pred, v_target, mask)

            n_frames = mask.sum().item()
            total_loss += loss.item() * n_frames
            total_frames += n_frames

        avg_loss = total_loss / max(total_frames, 1)
        self.model.train()
        return avg_loss

    # ----- Save / Resume -----

    def save_checkpoint(self, tag: str = None):
        """Сохранить только адаптерные параметры + optimizer state."""
        if tag is None:
            tag = f"step_{self.global_step}"
        save_path = self.save_dir / tag
        save_path.mkdir(parents=True, exist_ok=True)

        # Adapter weights
        save_adapters(self.model, str(save_path / "adapters.pt"))

        # Optimizer + scheduler + step
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "config": self.config,
        }, str(save_path / "trainer_state.pt"))

        print(f"[Checkpoint] Saved to {save_path}")

        # Cleanup old checkpoints
        keep_n = self.config.get("checkpoint", {}).get("keep_last_n", 3)
        self._cleanup_old_checkpoints(keep_n)

    def _cleanup_old_checkpoints(self, keep_n: int):
        """Удалить старые чекпоинты, оставить keep_n последних."""
        ckpts = sorted(
            [d for d in self.save_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: int(d.name.split("_")[1]),
        )
        if len(ckpts) > keep_n:
            for old in ckpts[:-keep_n]:
                import shutil
                shutil.rmtree(old)
                print(f"[Checkpoint] Removed old: {old.name}")

    def _resume(self, path: str):
        """Resume training from checkpoint."""
        path = Path(path)
        if not path.exists():
            print(f"[Resume] Path not found: {path}, starting fresh")
            return

        # Load adapter weights
        adapter_path = path / "adapters.pt"
        if adapter_path.exists():
            load_adapters(self.model, str(adapter_path))

        # Load trainer state
        state_path = path / "trainer_state.pt"
        if state_path.exists():
            state = torch.load(str(state_path), map_location="cpu")
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.global_step = state.get("global_step", 0)
            self.epoch = state.get("epoch", 0)
            self.best_loss = state.get("best_loss", float("inf"))
            print(f"[Resume] Restored from step {self.global_step}, epoch {self.epoch}")

    # ----- Logging -----

    def _log_scalar(self, tag: str, value: float, step: int):
        if self.writer is None:
            return
        if hasattr(self.writer, "add_scalar"):
            self.writer.add_scalar(tag, value, step)
        elif hasattr(self.writer, "log"):  # wandb
            self.writer.log({tag: value}, step=step)

    def _log_audio(self, tag: str, audio: torch.Tensor, sr: int, step: int):
        if self.writer is None:
            return
        if hasattr(self.writer, "add_audio"):
            self.writer.add_audio(tag, audio.cpu(), step, sample_rate=sr)

    # ----- Generate eval samples -----

    @torch.no_grad()
    def generate_eval_samples(self):
        """Сгенерировать evaluation аудио для мониторинга."""
        eval_cfg = self.config.get("evaluation", {}).get("eval_refs", [])
        if not eval_cfg:
            return

        self.model.eval()
        sr = self.config.get("model", {}).get("mel_spec", {}).get("target_sample_rate", 24000)

        for i, ref in enumerate(eval_cfg):
            ref_audio_path = ref.get("audio", "")
            if not os.path.exists(ref_audio_path):
                continue

            gen_text = ref.get("gen_text", "")
            gen_lang = ref.get("gen_lang", "en")

            try:
                # Простая генерация через инференс (если доступен)
                from f5_tts_enhanced.model.enhanced_dit import EnhancedF5TTS

                # Extract emotion from reference
                emotion_emb = self.emotion_extractor.get_emotion_embedding(ref_audio_path)
                emotion_desc = self.emotion_extractor.describe_emotion(ref_audio_path)

                self._log_scalar(f"eval/sample_{i}_emotion", 1.0, self.global_step)
                print(f"  [Eval {i}] {gen_lang}: '{gen_text[:50]}...' | emotion: {emotion_desc}")

            except Exception as e:
                print(f"  [Eval {i}] Generation failed: {e}")

        self.model.train()

    # ----- Main training loop -----

    def _print_summary(self):
        """Вывести сводку перед обучением."""
        optim_cfg = self.config.get("optim", {})
        ckpt_cfg = self.config.get("checkpoint", {})

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\n{'='*60}")
        print(f"  Training Summary")
        print(f"{'='*60}")
        print(f"  Total params:     {total:>12,}")
        print(f"  Trainable params: {trainable:>12,} ({100*trainable/total:.2f}%)")
        print(f"  Max steps:        {optim_cfg.get('max_steps', 150000):>12,}")
        print(f"  Learning rate:    {optim_cfg.get('learning_rate', 1e-5):>12}")
        print(f"  Grad accum:       {optim_cfg.get('grad_accumulation_steps', 4):>12}")
        print(f"  Mixed precision:  {optim_cfg.get('mixed_precision', 'bf16'):>12}")
        print(f"  Save every:       {ckpt_cfg.get('save_per_updates', 5000):>12} steps")
        print(f"  Resume step:      {self.global_step:>12}")
        print(f"  Train batches:    {len(self.train_loader):>12}")
        print(f"  Device:           {self.device}")
        print(f"{'='*60}\n")

    def train(self):
        """Основной цикл обучения."""
        optim_cfg = self.config.get("optim", {})
        ckpt_cfg = self.config.get("checkpoint", {})
        log_cfg = self.config.get("logging", {})

        max_steps = optim_cfg.get("max_steps", 150000)
        max_epochs = optim_cfg.get("epochs", 100)
        grad_accum = optim_cfg.get("grad_accumulation_steps", 4)
        max_grad_norm = optim_cfg.get("max_grad_norm", 1.0)

        save_every = ckpt_cfg.get("save_per_updates", 5000)
        last_every = ckpt_cfg.get("last_per_steps", 1000)
        log_every = log_cfg.get("log_every_steps", 100)
        eval_every = self.config.get("evaluation", {}).get("eval_every_steps", 5000)

        # Accumulators
        accum_loss = 0.0
        accum_steps = 0
        step_start = time.time()

        self.model.train()

        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch

            if hasattr(self.train_loader, "batch_sampler"):
                sampler = self.train_loader.batch_sampler
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(self.train_loader):
                if self.global_step >= max_steps:
                    break

                # Forward + loss
                result = self.train_step(batch)
                loss = result["loss"]

                # Scale for accumulation
                scaled_loss = loss / grad_accum

                # Backward
                self.scaler.scale(scaled_loss).backward()
                accum_loss += loss.item()
                accum_steps += 1

                # Optimizer step (after accumulation)
                if accum_steps >= grad_accum:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        get_trainable_params(self.model),
                        max_grad_norm,
                    )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                    self.global_step += 1
                    avg_loss = accum_loss / accum_steps

                    # Logging
                    if self.global_step % log_every == 0:
                        elapsed = time.time() - step_start
                        steps_per_sec = log_every / elapsed
                        lr = self.scheduler.get_last_lr()[0]

                        print(
                            f"[Step {self.global_step:>7d}] "
                            f"loss={avg_loss:.4f}  "
                            f"grad={grad_norm:.3f}  "
                            f"lr={lr:.2e}  "
                            f"speed={steps_per_sec:.1f} steps/s  "
                            f"epoch={epoch}"
                        )

                        self._log_scalar("train/loss", avg_loss, self.global_step)
                        self._log_scalar("train/grad_norm", grad_norm.item(), self.global_step)
                        self._log_scalar("train/lr", lr, self.global_step)
                        self._log_scalar("train/steps_per_sec", steps_per_sec, self.global_step)

                        step_start = time.time()

                    # Save "last" checkpoint
                    if self.global_step % last_every == 0:
                        self.save_checkpoint("last")

                    # Save numbered checkpoint
                    if self.global_step % save_every == 0:
                        self.save_checkpoint()

                        # Validation
                        val_loss = self.validate()
                        print(f"  [Val] loss={val_loss:.4f} (best={self.best_loss:.4f})")
                        self._log_scalar("val/loss", val_loss, self.global_step)

                        if val_loss < self.best_loss:
                            self.best_loss = val_loss
                            self.save_checkpoint("best")
                            print(f"  [Val] New best! Saved as 'best'")

                    # Eval generation
                    if self.global_step % eval_every == 0:
                        self.generate_eval_samples()

                    # Reset accumulators
                    accum_loss = 0.0
                    accum_steps = 0

            if self.global_step >= max_steps:
                print(f"\n[Done] Reached max_steps={max_steps}")
                break

        # Final save
        self.save_checkpoint("final")
        val_loss = self.validate()
        print(f"\n[Final] val_loss={val_loss:.4f}, best={self.best_loss:.4f}")
        print(f"[Final] Checkpoints saved in {self.save_dir}")

        # Close logger
        if hasattr(self.writer, "close"):
            self.writer.close()


# =========================================================================
# Export adapter for inference
# =========================================================================

def export_for_inference(checkpoint_dir: str, output_path: str):
    """
    Экспорт только адаптерных весов в один компактный файл для инференса.

    Usage:
        python finetune_enhanced.py --export ckpts/enhanced/best --output model_adapters.pt
    """
    ckpt_dir = Path(checkpoint_dir)
    adapter_path = ckpt_dir / "adapters.pt"

    if not adapter_path.exists():
        print(f"Error: {adapter_path} not found")
        return

    adapters = torch.load(str(adapter_path), map_location="cpu")
    n_params = sum(v.numel() for v in adapters.values())
    size_mb = sum(v.element_size() * v.numel() for v in adapters.values()) / 1024 / 1024

    # Сохранить с метаданными
    state_path = ckpt_dir / "trainer_state.pt"
    meta = {}
    if state_path.exists():
        state = torch.load(str(state_path), map_location="cpu")
        meta = {
            "global_step": state.get("global_step", 0),
            "best_loss": state.get("best_loss", 0),
            "config": state.get("config", {}),
        }

    torch.save({
        "adapters": adapters,
        "metadata": meta,
    }, output_path)

    print(f"Exported {len(adapters)} adapter params ({n_params:,} values, {size_mb:.1f} MB)")
    print(f"Saved to: {output_path}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="F5-TTS Enhanced Training")
    parser.add_argument("--config", type=str, default="configs/finetune_crosslang_emotion.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint dir")
    parser.add_argument("--export", type=str, default=None, help="Export checkpoint for inference")
    parser.add_argument("--output", type=str, default="model_adapters.pt", help="Export output path")
    args = parser.parse_args()

    if args.export:
        export_for_inference(args.export, args.output)
        return

    trainer = Trainer(
        config_path=args.config,
        resume_from=args.resume,
    )
    trainer.train()


if __name__ == "__main__":
    main()
