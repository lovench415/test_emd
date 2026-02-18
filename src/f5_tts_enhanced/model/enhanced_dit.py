"""
Enhanced DiT Wrapper (v2 — corrected for real F5-TTS architecture)
===================================================================

F5-TTS architecture (from source):
  CFM(transformer=DiT(...))
    ├─ CFM.forward(inp, text, lens)       → loss, cond, pred   [training]
    ├─ CFM.sample(cond, text, duration)   → mel, trajectory    [inference]
    └─ DiT = CFM.transformer

  DiT.forward(x, cond, text, time, mask, drop_audio_cond, drop_text, ...):
    t = self.time_embed(time)                              # (B,) → (B, dim)
    text_emb = self.text_embed(text, seq_len, drop_text)   # (B, nt) → (B, N, 512)
    x = self.input_embed(x, cond, text_emb, drop_audio_cond)
        # proj(cat(x, cond, text_emb, dim=-1)) + conv_pos_embed → (B, N, 1024)
    rope = self.rotary_embed(seq_len)
    for block in self.transformer_blocks:
        x = block(x, t, mask=mask, rope=rope)              # block NEEDS t & rope!
    x = self.norm_out(x, t)                                 # AdaLN NEEDS t!
    return self.proj_out(x)                                 # (B, N, 100)

  ALL tensors: (B, N, D) — channels LAST

Integration strategy:
  - Do NOT reimplement DiT forward — hook into it
  - Monkey-patch DiT to inject emotion + language
  - LoRA via forward hooks on Q/V Linear layers
  - ConvNeXt adapter hooks on text_embed.text_blocks
  - Use CFM.forward() directly for training (handles flow matching)
"""

import torch
import torch.nn as nn
from typing import Optional
from functools import wraps

from .adapters import (
    ConditioningAdapter,
    PromptAdapter,
    DiTLoRAAdapter,
    LanguageEmbedding,
    EmotionConditioningLayer,
    get_trainable_params,
    save_adapters,
    load_adapters,
)


class EnhancedF5TTS(nn.Module):
    """
    Обёртка над F5-TTS CFM моделью с PEFT адаптерами.

    Ключевые отличия от v1:
      - Работает с CFM(transformer=DiT), а не с голым DiT
      - Все тензоры (B, N, D) — channels-last (как в оригинале)
      - DiTBlock вызывается как block(x, t, mask, rope) — не block(x)
      - norm_out(x, t) — нужен timestep
      - LoRA/emotion/language через forward hooks (не переписывая forward)
      - CFM.forward() используется напрямую для training loss
    """

    def __init__(
        self,
        cfm_model: nn.Module,          # CFM(transformer=DiT)
        emotion_dim: int = 768,
        emotion_mode: str = "adaln",
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        cond_adapter_compression: float = 0.25,
        prompt_drop_path: float = 0.3,
        add_language_emb: bool = True,
        freeze_base: bool = True,
        num_dit_blocks: int = 22,
        hidden_dim: int = 1024,
        text_dim: int = 512,
    ):
        super().__init__()

        self.cfm_model = cfm_model
        self.hidden_dim = hidden_dim
        self.emotion_dim = emotion_dim
        self.text_dim = text_dim

        # Получаем DiT внутри CFM
        self.dit = cfm_model.transformer

        # Заморозить всю базовую модель
        if freeze_base:
            for param in cfm_model.parameters():
                param.requires_grad = False

        # ---- Emotion Conditioning Layers (per DiT block) ----
        self.emotion_cond_layers = nn.ModuleList([
            EmotionConditioningLayer(
                hidden_dim=hidden_dim,
                emotion_dim=emotion_dim,
                mode=emotion_mode,
            )
            for _ in range(num_dit_blocks)
        ])

        # ---- LoRA Adapters для Q,V (per block) ----
        self.lora_q = nn.ModuleList([
            DiTLoRAAdapter(hidden_dim, hidden_dim, rank=lora_rank,
                           alpha=lora_alpha, dropout=lora_dropout)
            for _ in range(num_dit_blocks)
        ])
        self.lora_v = nn.ModuleList([
            DiTLoRAAdapter(hidden_dim, hidden_dim, rank=lora_rank,
                           alpha=lora_alpha, dropout=lora_dropout)
            for _ in range(num_dit_blocks)
        ])

        # ---- ConvNeXt Conditioning Adapters (text_embed) ----
        num_conv_layers = 4
        self.cond_adapters = nn.ModuleList([
            ConditioningAdapter(text_dim, compression_factor=cond_adapter_compression)
            for _ in range(num_conv_layers)
        ])

        # ---- Prompt Adapter на выход input_embed (dim=1024) ----
        self.prompt_adapter = PromptAdapter(
            in_features=hidden_dim,
            out_features=hidden_dim,
            rank=lora_rank,
            drop_path_rate=prompt_drop_path,
        )

        # ---- Language Embedding ----
        self.language_embedding = None
        if add_language_emb:
            self.language_embedding = LanguageEmbedding(embed_dim=hidden_dim)

        # ---- Emotion → timestep-space projection ----
        self.emotion_to_timestep = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Runtime state (set per forward, read by hooks)
        self._current_emotion_emb = None   # (B, emotion_dim)
        self._current_emotion_bias = None  # (B, hidden_dim)
        self._current_lang_id = None       # (B,)
        self._hooks = []

        self._count_params()

    def _count_params(self):
        adapter_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        base_params = sum(p.numel() for p in self.cfm_model.parameters())
        total = base_params + adapter_params
        pct = 100 * adapter_params / total if total > 0 else 0
        print(f"[EnhancedF5TTS] Base CFM: {base_params:,} (frozen)")
        print(f"[EnhancedF5TTS] Adapters: {adapter_params:,} (trainable, {pct:.2f}%)")

    # ================================================================
    # Hook installation
    # ================================================================

    def install_hooks(self):
        """Install all forward hooks into DiT. Call once after model creation."""
        self._remove_hooks()
        self._install_lora_hooks()
        self._install_input_embed_hook()
        self._install_convnext_hooks()
        self._install_block_hooks()
        print(f"[EnhancedF5TTS] Installed {len(self._hooks)} hooks")

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _install_lora_hooks(self):
        """LoRA hooks на Q/V проекции в DiT блоках."""
        dit_blocks = list(self.dit.transformer_blocks)
        q_names = {"to_q", "q_proj", "wq"}
        v_names = {"to_v", "v_proj", "wv"}

        for block_idx, block in enumerate(dit_blocks):
            if block_idx >= len(self.lora_q):
                break

            lora_q = self.lora_q[block_idx]
            lora_v = self.lora_v[block_idx]

            for name, module in block.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                last = name.split(".")[-1]
                if last in q_names:
                    h = module.register_forward_hook(_make_lora_hook(lora_q))
                    self._hooks.append(h)
                elif last in v_names:
                    h = module.register_forward_hook(_make_lora_hook(lora_v))
                    self._hooks.append(h)

    def _install_input_embed_hook(self):
        """
        Hook на InputEmbedding — инъекция:
          - PromptAdapter residual
          - Language embedding
          - Emotion bias
        """
        if not hasattr(self.dit, "input_embed"):
            return

        parent = self
        prompt_adapter = self.prompt_adapter
        lang_emb = self.language_embedding
        emo_proj = self.emotion_to_timestep

        def hook(module, input, output):
            # output: (B, N, dim=1024)
            x = output

            # 1. Prompt adapter residual
            x = x + prompt_adapter(x)

            # 2. Language embedding
            if lang_emb is not None and parent._current_lang_id is not None:
                x = x + lang_emb(parent._current_lang_id, x.shape[1])

            # 3. Emotion bias (global, broadcast over time)
            if parent._current_emotion_bias is not None:
                # Handle CFG doubling: if batch is doubled, repeat emotion
                B_x = x.shape[0]
                B_emo = parent._current_emotion_bias.shape[0]
                emo = parent._current_emotion_bias
                if B_x == B_emo * 2:  # CFG: cond + uncond batched together
                    emo = emo.repeat(2, 1)
                x = x + emo.unsqueeze(1)

            return x

        h = self.dit.input_embed.register_forward_hook(hook)
        self._hooks.append(h)

    def _install_convnext_hooks(self):
        """Hooks на ConvNeXt V2 блоки внутри TextEmbedding."""
        if not hasattr(self.dit, "text_embed"):
            return
        text_embed = self.dit.text_embed
        if not hasattr(text_embed, "text_blocks"):
            return

        blocks = list(text_embed.text_blocks)
        for i, block in enumerate(blocks):
            if i >= len(self.cond_adapters):
                break
            adapter = self.cond_adapters[i]

            def make_hook(adpt):
                def hook(module, input, output):
                    # output: (B, N, text_dim=512) — channels last
                    h = output.transpose(1, 2)  # → (B, 512, N) for Conv1d
                    h = adpt(h)
                    return h.transpose(1, 2)    # → (B, N, 512)
                return hook

            h = block.register_forward_hook(make_hook(adapter))
            self._hooks.append(h)

    def _install_block_hooks(self):
        """Post-block hooks для emotion conditioning."""
        dit_blocks = list(self.dit.transformer_blocks)
        parent = self

        for block_idx, block in enumerate(dit_blocks):
            if block_idx >= len(self.emotion_cond_layers):
                break
            emo_layer = self.emotion_cond_layers[block_idx]

            def make_hook(el):
                def hook(module, input, output):
                    if parent._current_emotion_emb is None:
                        return output
                    # Handle CFG doubling
                    B_x = output.shape[0]
                    emo = parent._current_emotion_emb
                    B_emo = emo.shape[0]
                    if B_x == B_emo * 2:
                        emo = emo.repeat(2, 1)
                    return el(output, emo)
                return hook

            h = block.register_forward_hook(make_hook(emo_layer))
            self._hooks.append(h)

    # ================================================================
    # Forward (training) — delegates to CFM.forward()
    # ================================================================

    def forward(
        self,
        inp: torch.Tensor,                         # raw audio or mel
        text: torch.Tensor,                         # text tokens (B, nt)
        emotion_emb: Optional[torch.Tensor] = None, # (B, emotion_dim)
        lang_id: Optional[torch.Tensor] = None,     # (B,)
        **cfm_kwargs,                                # lens, etc.
    ):
        """
        Training forward: flow matching loss via CFM.forward().

        Sets emotion/language context → CFM.forward() calls DiT.forward()
        with our hooks active → hooks inject adapters.

        Returns:
            loss: scalar
            cond: condition mel (B, N, mel_dim)
            pred: model velocity prediction (B, N, mel_dim)
        """
        self._set_context(emotion_emb, lang_id)
        try:
            loss, cond, pred = self.cfm_model(inp, text, **cfm_kwargs)
        finally:
            self._clear_context()
        return loss, cond, pred

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        text: torch.Tensor,
        duration,
        emotion_emb: Optional[torch.Tensor] = None,
        lang_id: Optional[torch.Tensor] = None,
        **sample_kwargs,
    ):
        """Inference: mel generation via CFM.sample()."""
        self._set_context(emotion_emb, lang_id)
        try:
            out, trajectory = self.cfm_model.sample(
                cond=cond, text=text, duration=duration, **sample_kwargs
            )
        finally:
            self._clear_context()
        return out, trajectory

    def _set_context(self, emotion_emb, lang_id):
        self._current_emotion_emb = emotion_emb
        if emotion_emb is not None:
            self._current_emotion_bias = self.emotion_to_timestep(emotion_emb)
        else:
            self._current_emotion_bias = None
        self._current_lang_id = lang_id

    def _clear_context(self):
        self._current_emotion_emb = None
        self._current_emotion_bias = None
        self._current_lang_id = None


# =========================================================================
# Helpers
# =========================================================================

def _make_lora_hook(lora: DiTLoRAAdapter):
    """Create a forward hook that adds LoRA residual."""
    def hook(module, input, output):
        return output + lora(input[0])
    return hook


# =========================================================================
# Factory
# =========================================================================

def create_enhanced_model(
    ckpt_path: str = "",
    vocab_path: str = "",
    device: str = "cuda",
    emotion_dim: int = 768,
    emotion_mode: str = "adaln",
    lora_rank: int = 16,
    freeze_base: bool = True,
    model_cfg: dict = None,
) -> EnhancedF5TTS:
    """
    Create Enhanced F5-TTS from pretrained checkpoint.

    Returns EnhancedF5TTS with hooks installed, on device.
    """
    if model_cfg is None:
        model_cfg = {
            "dim": 1024, "depth": 22, "heads": 16,
            "ff_mult": 2, "text_dim": 512, "conv_layers": 4,
        }

    from f5_tts.model import CFM, DiT
    from f5_tts.model.utils import get_tokenizer

    # Tokenizer
    vocab_char_map, vocab_size = get_tokenizer(vocab_path, "custom")

    # Create architecture
    dit = DiT(
        **model_cfg,
        text_num_embeds=vocab_size,
        mel_dim=100,
    )

    cfm = CFM(
        transformer=dit,
        mel_spec_kwargs=dict(
            n_fft=1024, hop_length=256, win_length=1024,
            n_mel_channels=100, target_sample_rate=24000,
            mel_spec_type="vocos",
        ),
        vocab_char_map=vocab_char_map,
    )

    # Load pretrained weights
    if ckpt_path:
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(ckpt_path)
        else:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if "ema_model_state_dict" in state:
                state = state["ema_model_state_dict"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]

        cfm.load_state_dict(state, strict=False)
        print(f"[Factory] Loaded base F5-TTS from {ckpt_path}")

    # Wrap
    enhanced = EnhancedF5TTS(
        cfm_model=cfm,
        emotion_dim=emotion_dim,
        emotion_mode=emotion_mode,
        lora_rank=lora_rank,
        freeze_base=freeze_base,
        num_dit_blocks=model_cfg.get("depth", 22),
        hidden_dim=model_cfg.get("dim", 1024),
        text_dim=model_cfg.get("text_dim", 512),
    )

    enhanced.install_hooks()
    enhanced = enhanced.to(device)
    return enhanced
