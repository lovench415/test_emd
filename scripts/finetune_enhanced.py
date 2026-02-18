#!/usr/bin/env python3
"""
F5-TTS Enhanced — Training Pipeline (v3 — fully corrected)
============================================================

Fixed vs earlier versions:
  - EnhancedF5TTS wraps CFM (not bare DiT): cfm_model=cfm
  - Uses enhanced.install_hooks() for all hook registration
  - Uses enhanced.predict_velocity() for direct DiT forward
  - All mel tensors: (B, T, mel_dim) — channels LAST
  - Random span masking consistent with F5-TTS training
  - CFG dropout during training (drop_audio_cond / drop_text)
  - EMA with proper device handling on resume
  - Loss only on generation region (not reference/padding)

Usage:
    python scripts/finetune_enhanced.py --config configs/finetune_crosslang_emotion.yaml
    python scripts/finetune_enhanced.py --config ... --resume ckpts/enhanced/step_50000
    python scripts/finetune_enhanced.py --export ckpts/enhanced/best --output adapters.pt
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from f5_tts_enhanced.model.enhanced_dit import EnhancedF5TTS
from f5_tts_enhanced.model.emotion_extractor import Emotion2vecExtractor
from f5_tts_enhanced.data.dataset import create_dataloader


# =========================================================================
# EMA for adapter params (with device-safe resume)
# =========================================================================

class AdapterEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        self.backup.clear()
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self):
        return {"shadow": {k: v.cpu() for k, v in self.shadow.items()},
                "decay": self.decay}

    def load_state_dict(self, state, device=None):
        """Load EMA state with proper device mapping."""
        self.decay = state.get("decay", self.decay)
        loaded_shadow = state.get("shadow", {})
        for name in self.shadow:
            if name in loaded_shadow:
                src = loaded_shadow[name]
                if device is not None:
                    src = src.to(device)
                self.shadow[name].copy_(src)


# =========================================================================
# Flow Matching (OT-CFM)
# =========================================================================

class FlowMatchingScheduler:
    """σ=0: x_t = (1-t)·x₀ + t·x₁,  v = x₁ - x₀"""

    def __init__(self, sigma: float = 0.0):
        self.sigma = sigma

    def sample_timesteps(self, B: int, device: torch.device) -> torch.Tensor:
        return torch.rand(B, device=device)

    def interpolate(self, x_0, x_1, t):
        t = t[:, None, None]
        mu_t = (1 - (1 - self.sigma) * t) * x_0 + t * x_1
        v_target = x_1 - (1 - self.sigma) * x_0
        return mu_t, v_target

    def compute_loss(self, v_pred, v_target, loss_mask):
        """Masked MSE. loss_mask: (B, T) True=compute loss."""
        mask = loss_mask.unsqueeze(-1).float()
        loss = F.mse_loss(v_pred, v_target, reduction="none")
        return (loss * mask).sum() / (mask.sum() * v_pred.shape[-1] + 1e-8)


# =========================================================================
# Random Span Masking
# =========================================================================

def random_span_mask(mel_lengths, max_len, mask_ratio_range=(0.7, 1.0)):
    """True = generate (target), False = condition (reference prompt)."""
    B = mel_lengths.shape[0]
    gen_mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i in range(B):
        L = mel_lengths[i].item()
        if L < 4:
            gen_mask[i, :L] = True
            continue
        ratio = random.uniform(*mask_ratio_range)
        mask_len = max(int(L * ratio), 1)
        start = random.randint(0, max(L - mask_len, 1) - 1)
        gen_mask[i, start:min(start + mask_len, L)] = True
    return gen_mask


# =========================================================================
# Trainer
# =========================================================================

class Trainer:

    def __init__(self, config_path: str, resume_from: str = None):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        ckpt_cfg = self.config.get("checkpoint", {})
        self.save_dir = Path(ckpt_cfg.get("save_dir", "ckpts/enhanced"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        log_cfg = self.config.get("logging", {})
        self.log_dir = Path(log_cfg.get("log_dir", "logs/crosslang_emotion"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("  F5-TTS Enhanced Training (v3)")
        print("=" * 60)

        self.enhanced = self._build_model()
        self.emotion_extractor = self._build_emotion_extractor()
        self.flow = FlowMatchingScheduler(sigma=0.0)
        self.train_loader, self.val_loader = self._build_dataloaders()
        self.optimizer, self.scheduler = self._build_optimizer()
        self.ema = AdapterEMA(self.enhanced, decay=0.9999)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp())
        self.writer = self._build_logger()

        # CFG dropout probabilities (matching F5-TTS defaults)
        self.audio_drop_prob = self.config.get("model", {}).get("audio_drop_prob", 0.3)
        self.cond_drop_prob = self.config.get("model", {}).get("cond_drop_prob", 0.2)

        if resume_from:
            self._resume(resume_from)
        elif ckpt_cfg.get("resume_from"):
            self._resume(ckpt_cfg["resume_from"])

        self._print_summary()

    # ─── Build model ──────────────────────────────────────────────────

    def _build_model(self) -> EnhancedF5TTS:
        """Build CFM → wrap with EnhancedF5TTS → install hooks → to device."""
        base_cfg = self.config.get("base_model", {})
        model_cfg = self.config.get("model", {}).get("arch", {})
        adapter_cfg = self.config.get("adapters", {})
        emo_cfg = self.config.get("emotion", {})
        mel_cfg = self.config.get("model", {}).get("mel_spec", {})

        try:
            from f5_tts.model import CFM, DiT
            from f5_tts.model.utils import get_tokenizer

            vocab_path = base_cfg.get("vocab", "")
            if vocab_path.startswith("hf://"):
                from huggingface_hub import hf_hub_download
                parts = vocab_path.replace("hf://", "").split("/")
                vocab_path = hf_hub_download("/".join(parts[:2]), "/".join(parts[2:]))

            vocab_char_map, vocab_size = get_tokenizer(vocab_path, "custom")

            dit = DiT(
                dim=model_cfg.get("dim", 1024),
                depth=model_cfg.get("depth", 22),
                heads=model_cfg.get("heads", 16),
                ff_mult=model_cfg.get("ff_mult", 2),
                text_dim=model_cfg.get("text_dim", 512),
                conv_layers=model_cfg.get("conv_layers", 4),
                text_num_embeds=vocab_size,
                mel_dim=mel_cfg.get("n_mel_channels", 100),
            )

            cfm = CFM(
                transformer=dit,
                mel_spec_kwargs=dict(
                    n_fft=mel_cfg.get("n_fft", 1024),
                    hop_length=mel_cfg.get("hop_length", 256),
                    win_length=mel_cfg.get("win_length", 1024),
                    n_mel_channels=mel_cfg.get("n_mel_channels", 100),
                    target_sample_rate=mel_cfg.get("target_sample_rate", 24000),
                    mel_spec_type=mel_cfg.get("mel_spec_type", "vocos"),
                ),
                vocab_char_map=vocab_char_map,
            )

            # Load pretrained weights
            ckpt_path = base_cfg.get("checkpoint", "")
            if ckpt_path:
                if ckpt_path.startswith("hf://"):
                    from huggingface_hub import hf_hub_download
                    parts = ckpt_path.replace("hf://", "").split("/")
                    local_path = hf_hub_download("/".join(parts[:2]), "/".join(parts[2:]))
                else:
                    local_path = ckpt_path

                print(f"[Model] Loading weights: {local_path}")
                if local_path.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state = load_file(local_path, device="cpu")
                else:
                    state = torch.load(local_path, map_location="cpu", weights_only=True)
                    if "ema_model_state_dict" in state:
                        state = state["ema_model_state_dict"]
                    elif "model_state_dict" in state:
                        state = state["model_state_dict"]

                cfm.load_state_dict(state, strict=False)
                print("[Model] Pretrained weights loaded")

        except ImportError as e:
            print(f"[WARNING] f5_tts not installed ({e}), using placeholder")
            dit = nn.Linear(100, 100)  # placeholder

            class FakeCFM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.transformer = dit
            cfm = FakeCFM()

        # Wrap CFM with EnhancedF5TTS (pass entire CFM, not just DiT!)
        lora_cfg = adapter_cfg.get("lora", {})
        emo_ext = emo_cfg.get("extractor", {})

        enhanced = EnhancedF5TTS(
            cfm_model=cfm,                                     # ← full CFM
            emotion_dim=emo_ext.get("emotion_dim", 768),
            emotion_mode=emo_cfg.get("conditioning", {}).get("mode", "adaln"),
            lora_rank=lora_cfg.get("rank", 16),
            lora_alpha=lora_cfg.get("alpha", 16.0),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            freeze_base=adapter_cfg.get("freeze_base", True),
            num_dit_blocks=model_cfg.get("depth", 22),
            hidden_dim=model_cfg.get("dim", 1024),
            text_dim=model_cfg.get("text_dim", 512),
        )
        enhanced.install_hooks()                                # ← install all hooks
        enhanced = enhanced.to(self.device)

        return enhanced

    def _build_emotion_extractor(self) -> Emotion2vecExtractor:
        emo_cfg = self.config.get("emotion", {}).get("extractor", {})
        extractor = Emotion2vecExtractor(
            model_size=emo_cfg.get("model_size", "large"),
            emotion_dim=emo_cfg.get("emotion_dim", 768),
            hub=emo_cfg.get("hub", "hf"),
            cache_dir=str(self.save_dir / "emotion_cache"),
        )
        extractor.load_model()
        for p in extractor.parameters():
            p.requires_grad = False
        extractor.eval()
        return extractor

    def _build_dataloaders(self):
        ds_cfg = self.config.get("datasets", {})
        mel_cfg = self.config.get("model", {}).get("mel_spec", {})
        base_cfg = self.config.get("base_model", {})

        data_root = ds_cfg.get("data_root", "/data/prepared")
        meta_train = ds_cfg.get("metadata_train", os.path.join(data_root, "metadata.csv"))
        meta_val = ds_cfg.get("metadata_val", os.path.join(data_root, "metadata_val.csv"))
        vocab_path = base_cfg.get("vocab", os.path.join(data_root, "vocab.txt"))
        if vocab_path.startswith("hf://"):
            from huggingface_hub import hf_hub_download
            parts = vocab_path.replace("hf://", "").split("/")
            vocab_path = hf_hub_download("/".join(parts[:2]), "/".join(parts[2:]))

        kw = dict(
            data_root=data_root, vocab_path=vocab_path,
            target_sr=mel_cfg.get("target_sample_rate", 24000),
            n_mels=mel_cfg.get("n_mel_channels", 100),
            hop_length=mel_cfg.get("hop_length", 256),
            max_duration=ds_cfg.get("max_duration", 30.0),
        )
        mf = ds_cfg.get("batch_size_per_gpu", 3200)
        ms = ds_cfg.get("max_samples", 64)
        nw = ds_cfg.get("num_workers", 4)

        print(f"[Data] train={meta_train}  val={meta_val}")
        train_ds, train_dl = create_dataloader(
            meta_train, max_frames_per_batch=mf, max_samples_per_batch=ms,
            num_workers=nw, shuffle=True, **kw)

        val_dl = None
        if os.path.exists(meta_val):
            _, val_dl = create_dataloader(
                meta_val, max_frames_per_batch=mf, max_samples_per_batch=ms,
                num_workers=nw, shuffle=False, **kw)

        print("[Trainer] Warming up emotion cache...")
        paths = [s["audio_path"] for s in train_ds.samples]
        self.emotion_extractor.warmup_cache(paths)
        self.emotion_extractor.print_cache_stats()

        return train_dl, val_dl

    def _build_optimizer(self):
        cfg = self.config.get("optim", {})
        lr = cfg.get("learning_rate", 1e-5)
        wd = cfg.get("weight_decay", 0.01)
        max_steps = cfg.get("max_steps", 150000)
        warmup = cfg.get("num_warmup_updates", 2000)

        params = [p for p in self.enhanced.parameters() if p.requires_grad]
        print(f"[Optimizer] {len(params)} param groups, lr={lr}")

        if cfg.get("bnb_optimizer", False):
            try:
                import bitsandbytes as bnb
                opt = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=wd)
            except ImportError:
                opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        else:
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, max_steps - warmup)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def _use_amp(self):
        return (self.config.get("optim", {}).get("mixed_precision", "bf16")
                in ("fp16", "bf16") and torch.cuda.is_available())

    def _amp_dtype(self):
        mp = self.config.get("optim", {}).get("mixed_precision", "bf16")
        if mp == "bf16" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _build_logger(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            return SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:
            return None

    # ─── Train step ───────────────────────────────────────────────────

    @torch.no_grad()
    def _extract_emotion_batch(self, audio_paths):
        return self.emotion_extractor.get_batch_embeddings(audio_paths).to(self.device)

    def train_step(self, batch: dict) -> dict:
        self.enhanced.train()

        mel = batch["mel"].to(self.device)                # (B, T, mel_dim) channels LAST
        text_ids = batch["text_ids"].to(self.device)       # (B, L)
        mel_lengths = batch["mel_lengths"].to(self.device)  # (B,)
        lang_ids = batch["lang_ids"].to(self.device)        # (B,)
        mask = batch["mask"].to(self.device)                # (B, T) padding
        audio_paths = batch["audio_paths"]

        B, T, D = mel.shape

        # 1. Emotion embeddings (cached → <1ms)
        emotion_emb = self._extract_emotion_batch(audio_paths)

        # 2. Random span masking: gen_mask True = generate, False = condition
        gen_mask = random_span_mask(mel_lengths, T).to(self.device)

        # 3. Condition mel: keep reference, zero out generation region
        cond = mel.clone()
        cond[gen_mask] = 0.0

        # 4. Flow matching
        x_0 = torch.randn_like(mel)
        t = self.flow.sample_timesteps(B, self.device)
        x_t, v_target = self.flow.interpolate(x_0, mel, t)

        # 5. CFG dropout (match F5-TTS training)
        drop_audio = random.random() < self.audio_drop_prob
        drop_text = random.random() < self.cond_drop_prob

        # 6. Forward — hooks fire inside DiT
        with torch.amp.autocast("cuda", dtype=self._amp_dtype(), enabled=self._use_amp()):
            v_pred = self.enhanced.predict_velocity(
                x=x_t,
                cond=cond,
                text=text_ids,
                time=t,
                mask=mask,
                drop_audio_cond=drop_audio,
                drop_text=drop_text,
                emotion_emb=emotion_emb,
                lang_id=lang_ids,
            )

            # 7. Loss on GENERATED region only (not reference, not padding)
            loss_mask = gen_mask & mask
            loss = self.flow.compute_loss(v_pred, v_target, loss_mask)

        return {"loss": loss, "batch_size": B}

    # ─── Validation ───────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self) -> float:
        if self.val_loader is None:
            return float("inf")

        self.ema.apply(self.enhanced)
        self.enhanced.eval()
        total_loss, total_frames = 0.0, 0

        for batch in self.val_loader:
            mel = batch["mel"].to(self.device)
            text_ids = batch["text_ids"].to(self.device)
            mel_lengths = batch["mel_lengths"].to(self.device)
            lang_ids = batch["lang_ids"].to(self.device)
            mask = batch["mask"].to(self.device)
            paths = batch["audio_paths"]

            B, T, D = mel.shape
            emo = self._extract_emotion_batch(paths)
            gen_mask = random_span_mask(mel_lengths, T).to(self.device)
            cond = mel.clone()
            cond[gen_mask] = 0.0

            x_0 = torch.randn_like(mel)
            t = self.flow.sample_timesteps(B, self.device)
            x_t, v_target = self.flow.interpolate(x_0, mel, t)

            with torch.amp.autocast("cuda", dtype=self._amp_dtype(), enabled=self._use_amp()):
                v_pred = self.enhanced.predict_velocity(
                    x_t, cond, text_ids, t, mask=mask,
                    emotion_emb=emo, lang_id=lang_ids)
                loss_mask = gen_mask & mask
                loss = self.flow.compute_loss(v_pred, v_target, loss_mask)

            n = loss_mask.sum().item()
            total_loss += loss.item() * n
            total_frames += n

        self.ema.restore(self.enhanced)
        self.enhanced.train()
        return total_loss / max(total_frames, 1)

    # ─── Save / Resume ────────────────────────────────────────────────

    def save_checkpoint(self, tag=None):
        tag = tag or f"step_{self.global_step}"
        path = self.save_dir / tag
        path.mkdir(parents=True, exist_ok=True)

        # Save adapter params only (~15-20 MB)
        torch.save(self.enhanced.get_adapter_state_dict(), str(path / "adapters.pt"))
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "ema": self.ema.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
        }, str(path / "trainer_state.pt"))
        print(f"[Checkpoint] → {path}")

        # Cleanup old checkpoints
        keep = self.config.get("checkpoint", {}).get("keep_last_n", 3)
        ckpts = sorted(
            [d for d in self.save_dir.iterdir()
             if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: int(d.name.split("_")[1]))
        for old in ckpts[:-keep]:
            import shutil
            shutil.rmtree(old)

    def _resume(self, path):
        path = Path(path)
        if not path.exists():
            print(f"[Resume] Not found: {path}")
            return

        # Load adapter weights
        ap = path / "adapters.pt"
        if ap.exists():
            state = torch.load(str(ap), map_location=self.device)
            self.enhanced.load_adapter_state_dict(state)

        # Load trainer state
        sp = path / "trainer_state.pt"
        if sp.exists():
            s = torch.load(str(sp), map_location="cpu")
            self.optimizer.load_state_dict(s["optimizer"])
            self.scheduler.load_state_dict(s["scheduler"])
            if "ema" in s:
                self.ema.load_state_dict(s["ema"], device=self.device)  # ← device-safe
            self.global_step = s.get("global_step", 0)
            self.epoch = s.get("epoch", 0)
            self.best_loss = s.get("best_loss", float("inf"))
            print(f"[Resume] step={self.global_step}, epoch={self.epoch}")

    # ─── Logging / Summary ────────────────────────────────────────────

    def _log(self, tag, val, step):
        if self.writer:
            self.writer.add_scalar(tag, val, step)

    def _print_summary(self):
        cfg = self.config.get("optim", {})
        total = sum(p.numel() for p in self.enhanced.parameters())
        trainable = sum(p.numel() for p in self.enhanced.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"  Total:      {total:>12,}")
        print(f"  Trainable:  {trainable:>12,}  ({100*trainable/max(total,1):.2f}%)")
        print(f"  Max steps:  {cfg.get('max_steps', 150000):>12,}")
        print(f"  LR:         {cfg.get('learning_rate', 1e-5):>12}")
        print(f"  Grad accum: {cfg.get('grad_accumulation_steps', 4):>12}")
        print(f"  AMP:        {cfg.get('mixed_precision', 'bf16'):>12}")
        print(f"  CFG drop:   audio={self.audio_drop_prob}  text={self.cond_drop_prob}")
        print(f"  Step:       {self.global_step:>12}")
        print(f"  Batches:    {len(self.train_loader):>12}")
        print(f"  Device:     {self.device}")
        print(f"{'='*60}\n")

    # ─── Main loop ────────────────────────────────────────────────────

    def train(self):
        cfg = self.config.get("optim", {})
        ckpt_cfg = self.config.get("checkpoint", {})
        log_cfg = self.config.get("logging", {})

        max_steps = cfg.get("max_steps", 150000)
        max_epochs = cfg.get("epochs", 100)
        grad_accum = cfg.get("grad_accumulation_steps", 4)
        max_grad_norm = cfg.get("max_grad_norm", 1.0)

        save_every = ckpt_cfg.get("save_per_updates", 5000)
        last_every = ckpt_cfg.get("last_per_steps", 1000)
        log_every = log_cfg.get("log_every_steps", 100)

        acc_loss, acc_n = 0.0, 0
        t0 = time.time()
        self.enhanced.train()

        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch
            sampler = getattr(self.train_loader, "batch_sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            for batch in self.train_loader:
                if self.global_step >= max_steps:
                    break

                result = self.train_step(batch)
                loss = result["loss"]

                self.scaler.scale(loss / grad_accum).backward()
                acc_loss += loss.item()
                acc_n += 1

                if acc_n >= grad_accum:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.enhanced.parameters() if p.requires_grad],
                        max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.ema.update(self.enhanced)

                    self.global_step += 1
                    avg = acc_loss / acc_n

                    if self.global_step % log_every == 0:
                        elapsed = time.time() - t0
                        sps = log_every / max(elapsed, 0.01)
                        lr = self.scheduler.get_last_lr()[0]
                        gn = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                        print(f"[{self.global_step:>7}] loss={avg:.4f}  "
                              f"grad={gn:.3f}  lr={lr:.2e}  "
                              f"speed={sps:.1f}it/s  ep={epoch}")
                        self._log("train/loss", avg, self.global_step)
                        self._log("train/grad_norm", gn, self.global_step)
                        self._log("train/lr", lr, self.global_step)
                        t0 = time.time()

                    if self.global_step % last_every == 0:
                        self.save_checkpoint("last")

                    if self.global_step % save_every == 0:
                        self.save_checkpoint()
                        val = self.validate()
                        print(f"  [Val] loss={val:.4f}  best={self.best_loss:.4f}")
                        self._log("val/loss", val, self.global_step)
                        if val < self.best_loss:
                            self.best_loss = val
                            self.save_checkpoint("best")
                            print("  [Val] ★ New best!")

                    acc_loss, acc_n = 0.0, 0

            if self.global_step >= max_steps:
                break

        self.save_checkpoint("final")
        val = self.validate()
        print(f"\n[Done] val={val:.4f}  best={self.best_loss:.4f}")
        if self.writer:
            self.writer.close()


# =========================================================================
# Export
# =========================================================================

def export_for_inference(ckpt_dir, output):
    ckpt_dir = Path(ckpt_dir)
    adapters = torch.load(str(ckpt_dir / "adapters.pt"), map_location="cpu",
                          weights_only=True)
    n = sum(v.numel() for v in adapters.values())
    mb = sum(v.element_size() * v.numel() for v in adapters.values()) / 1024 / 1024

    meta = {}
    sp = ckpt_dir / "trainer_state.pt"
    if sp.exists():
        s = torch.load(str(sp), map_location="cpu", weights_only=True)
        meta = {k: s.get(k) for k in ("global_step", "best_loss") if k in s}

    torch.save({"adapters": adapters, "metadata": meta}, output)
    print(f"Exported {len(adapters)} params ({n:,} values, {mb:.1f}MB) → {output}")


# =========================================================================
# Main
# =========================================================================

def main():
    p = argparse.ArgumentParser(description="F5-TTS Enhanced Training")
    p.add_argument("--config", default="configs/finetune_crosslang_emotion.yaml")
    p.add_argument("--resume", default=None)
    p.add_argument("--export", default=None)
    p.add_argument("--output", default="model_adapters.pt")
    args = p.parse_args()

    if args.export:
        export_for_inference(args.export, args.output)
        return

    Trainer(args.config, args.resume).train()


if __name__ == "__main__":
    main()
