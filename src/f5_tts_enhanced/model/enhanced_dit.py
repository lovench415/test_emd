"""
Enhanced DiT Wrapper (v4 — dynamic dimension detection)
=========================================================

Fixes vs v2/v3:
  - LoRA dimensions detected from ACTUAL Q/V Linear layers (not hardcoded)
  - Supports BOTH separate to_q/to_v AND fused to_qkv projections
  - DiT.forward() called with robust signature detection
  - ConditioningAdapter channels detected from actual ConvNeXt blocks
  - PromptAdapter dimensions detected from actual input_embed output
  - Full diagnostic printing on hook installation

Architecture:
  CFM(transformer=DiT(...))
    DiT.forward(x, cond, text, time, ...) → (B, N, mel_dim)
      Hooks intercept sub-modules to inject adapters:
        - text_embed ConvNeXt → conditioning adapters
        - input_embed → + prompt adapter + language embedding + emotion bias
        - DiTBlock Q/V → + LoRA
        - DiTBlock output → + emotion conditioning
"""

import torch
import torch.nn as nn
import inspect
from typing import Optional


from .adapters import (
    ConditioningAdapter,
    PromptAdapter,
    DiTLoRAAdapter,
    LanguageEmbedding,
    EmotionConditioningLayer,
)


class EnhancedF5TTS(nn.Module):
    """
    Hook-based PEFT wrapper for F5-TTS CFM model.
    All adapter dimensions are auto-detected from the loaded model.
    """

    def __init__(
        self,
        cfm_model: nn.Module,
        emotion_dim: int = 768,
        emotion_mode: str = "adaln",
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        cond_adapter_compression: float = 0.25,
        prompt_drop_path: float = 0.3,
        add_language_emb: bool = True,
        freeze_base: bool = True,
        # These are now FALLBACKS only — actual dims auto-detected
        num_dit_blocks: int = 22,
        hidden_dim: int = 1024,
        text_dim: int = 512,
    ):
        super().__init__()

        self.cfm_model = cfm_model
        self.emotion_dim = emotion_dim
        self.dit = cfm_model.transformer

        # ── Auto-detect dimensions from loaded model ──
        self._detected = self._detect_architecture(hidden_dim, text_dim, num_dit_blocks)
        hidden_dim = self._detected["hidden_dim"]
        text_dim = self._detected["text_dim"]
        num_dit_blocks = self._detected["num_blocks"]

        self.hidden_dim = hidden_dim
        self.text_dim = text_dim

        print(f"[EnhancedF5TTS] Detected: dim={hidden_dim}, text_dim={text_dim}, "
              f"blocks={num_dit_blocks}")

        # ── Freeze base ──
        if freeze_base:
            for param in cfm_model.parameters():
                param.requires_grad = False

        # ── Emotion Conditioning Layers (per block) ──
        self.emotion_cond_layers = nn.ModuleList([
            EmotionConditioningLayer(hidden_dim, emotion_dim, mode=emotion_mode)
            for _ in range(num_dit_blocks)
        ])

        # ── LoRA: created DYNAMICALLY in install_hooks ──
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_adapters = nn.ModuleDict()  # filled by _install_lora_hooks

        # ── ConvNeXt Conditioning Adapters: channels auto-detected ──
        convnext_channels = self._detected.get("convnext_channels", text_dim)
        num_convnext = self._detected.get("num_convnext_blocks", 4)
        self.cond_adapters = nn.ModuleList([
            ConditioningAdapter(convnext_channels,
                                compression_factor=cond_adapter_compression)
            for _ in range(num_convnext)
        ])

        # ── Prompt Adapter ──
        input_embed_dim = self._detected.get("input_embed_dim", hidden_dim)
        self.prompt_adapter = PromptAdapter(
            in_features=input_embed_dim,
            out_features=input_embed_dim,
            rank=lora_rank,
            drop_path_rate=prompt_drop_path,
        )

        # ── Language Embedding ──
        self.language_embedding = None
        if add_language_emb:
            self.language_embedding = LanguageEmbedding(embed_dim=hidden_dim)

        # ── Emotion → timestep bias ──
        self.emotion_to_timestep = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.emotion_to_timestep[-1].weight)
        nn.init.zeros_(self.emotion_to_timestep[-1].bias)

        # ── Detect DiT.forward signature ──
        self._dit_forward_params = set(inspect.signature(self.dit.forward).parameters.keys())
        print(f"[EnhancedF5TTS] DiT.forward params: {sorted(self._dit_forward_params)}")

        # Runtime state
        self._current_emotion_emb = None
        self._current_emotion_bias = None
        self._current_lang_id = None
        self._hooks = []

    # ================================================================
    # Architecture auto-detection
    # ================================================================

    def _detect_architecture(self, fallback_dim, fallback_text_dim, fallback_blocks):
        """Scan loaded DiT to detect actual dimensions."""
        dit = self.dit
        result = {
            "hidden_dim": fallback_dim,
            "text_dim": fallback_text_dim,
            "num_blocks": fallback_blocks,
            "convnext_channels": fallback_text_dim,
            "num_convnext_blocks": 4,
            "input_embed_dim": fallback_dim,
            "qv_layers": [],
        }

        # 1. Count DiT blocks
        if hasattr(dit, "transformer_blocks"):
            result["num_blocks"] = len(dit.transformer_blocks)

        # 2. Detect hidden_dim
        if hasattr(dit, "dim"):
            result["hidden_dim"] = dit.dim
        elif hasattr(dit, "transformer_blocks") and len(dit.transformer_blocks) > 0:
            block0 = dit.transformer_blocks[0]
            for m in block0.modules():
                if isinstance(m, nn.LayerNorm) and hasattr(m, "normalized_shape"):
                    result["hidden_dim"] = m.normalized_shape[0]
                    break

        # 3. Detect text_dim
        if hasattr(dit, "text_embed"):
            te = dit.text_embed
            if hasattr(te, "text_embed") and isinstance(te.text_embed, nn.Embedding):
                result["text_dim"] = te.text_embed.embedding_dim

        # 4. Detect ConvNeXt block count and channels
        if hasattr(dit, "text_embed"):
            te = dit.text_embed
            convnext_blocks = self._find_convnext_blocks(te)
            if convnext_blocks:
                result["num_convnext_blocks"] = len(convnext_blocks)
                for m in convnext_blocks[0].modules():
                    if isinstance(m, nn.Conv1d):
                        result["convnext_channels"] = m.in_channels
                        break
                    if isinstance(m, nn.Linear):
                        result["convnext_channels"] = m.in_features
                        break

        # 5. Detect input_embed output dim
        if hasattr(dit, "input_embed"):
            ie = dit.input_embed
            last_linear = None
            for m in ie.modules():
                if isinstance(m, nn.Linear):
                    last_linear = m
            if last_linear:
                result["input_embed_dim"] = last_linear.out_features

        # 6. Scan Q/V projections in blocks
        if hasattr(dit, "transformer_blocks"):
            for idx, block in enumerate(dit.transformer_blocks):
                qv = self._find_qv_layers(block, idx)
                result["qv_layers"].extend(qv)

        return result

    def _find_convnext_blocks(self, text_embed):
        """Find ConvNeXt blocks in TextEmbedding."""
        for attr in ("text_blocks", "extra", "conv_blocks"):
            if hasattr(text_embed, attr):
                candidate = getattr(text_embed, attr)
                if isinstance(candidate, (nn.ModuleList, nn.Sequential)):
                    return list(candidate)

        for child in text_embed.children():
            if isinstance(child, (nn.ModuleList, nn.Sequential)):
                items = list(child)
                if len(items) >= 2:
                    return items
        return []

    def _find_qv_layers(self, block, block_idx):
        """
        Find Q/V Linear layers in a DiT block.
        Handles separate (to_q, to_v) and fused (to_qkv).
        Returns: list of (block_idx, full_name, in_features, out_features, kind)
        """
        found = []
        sep_q = {"to_q", "q_proj", "wq"}
        sep_v = {"to_v", "v_proj", "wv"}
        fused = {"to_qkv", "qkv_proj", "Wqkv"}

        for name, module in block.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            leaf = name.split(".")[-1]

            if leaf in sep_q:
                found.append((block_idx, name, module.in_features,
                              module.out_features, "q"))
            elif leaf in sep_v:
                found.append((block_idx, name, module.in_features,
                              module.out_features, "v"))
            elif leaf in fused:
                found.append((block_idx, name, module.in_features,
                              module.out_features, "qkv"))
        return found

    # ================================================================
    # Hook installation
    # ================================================================

    def install_hooks(self):
        """Install all forward hooks. Call once after creation."""
        self._remove_hooks()
        n_lora = self._install_lora_hooks()
        n_input = self._install_input_embed_hook()
        n_conv = self._install_convnext_hooks()
        n_block = self._install_block_hooks()

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        base = sum(p.numel() for p in self.cfm_model.parameters())
        print(f"[EnhancedF5TTS] Hooks: {len(self._hooks)} "
              f"(LoRA={n_lora}, input={n_input}, conv={n_conv}, block={n_block})")
        print(f"[EnhancedF5TTS] Trainable: {total:,} ({100*total/max(base,1):.2f}%)")

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ─── LoRA hooks ───────────────────────────────────────────────────

    def _install_lora_hooks(self) -> int:
        """Create LoRA adapters with dimensions from ACTUAL Q/V layers."""
        qv_layers = self._detected["qv_layers"]
        count = 0

        if not qv_layers:
            print("[EnhancedF5TTS] WARNING: No Q/V layers found! LoRA disabled.")
            self._fallback_lora_scan()
            qv_layers = self._detected["qv_layers"]
            if not qv_layers:
                return 0

        blocks = list(self.dit.transformer_blocks)
        by_block = {}
        for (bidx, name, in_f, out_f, kind) in qv_layers:
            by_block.setdefault(bidx, []).append((name, in_f, out_f, kind))

        for bidx, layers in by_block.items():
            block = blocks[bidx]
            for (name, in_f, out_f, kind) in layers:
                module = block
                for part in name.split("."):
                    module = getattr(module, part)

                if kind in ("q", "v"):
                    key = f"lora_{kind}_{bidx}"
                    lora = DiTLoRAAdapter(
                        in_features=in_f, out_features=out_f,
                        rank=self.lora_rank, alpha=self.lora_alpha,
                        dropout=self.lora_dropout)
                    self.lora_adapters[key] = lora
                    h = module.register_forward_hook(_make_lora_hook(lora))
                    self._hooks.append(h)
                    count += 1
                    if count <= 2:
                        print(f"  LoRA {key}: Linear({in_f}, {out_f})")

                elif kind == "qkv":
                    split = out_f // 3
                    lora_q = DiTLoRAAdapter(
                        in_f, split, self.lora_rank, self.lora_alpha, self.lora_dropout)
                    lora_v = DiTLoRAAdapter(
                        in_f, split, self.lora_rank, self.lora_alpha, self.lora_dropout)
                    self.lora_adapters[f"lora_q_{bidx}"] = lora_q
                    self.lora_adapters[f"lora_v_{bidx}"] = lora_v
                    h = module.register_forward_hook(
                        _make_fused_qkv_lora_hook(lora_q, lora_v, split))
                    self._hooks.append(h)
                    count += 1
                    if count <= 2:
                        print(f"  LoRA fused block {bidx}: ({in_f},{out_f}) → Q,V @{split}")

        if count > 2:
            print(f"  ... +{count-2} more LoRA hooks ({count} total)")
        return count

    def _fallback_lora_scan(self):
        """Fallback: scan ALL Linear layers in blocks for likely Q/V."""
        if not hasattr(self.dit, "transformer_blocks"):
            return
        print("[EnhancedF5TTS] Fallback LoRA scan — looking for attention projections...")
        for idx, block in enumerate(self.dit.transformer_blocks):
            attn_linears = []
            for name, m in block.named_modules():
                if isinstance(m, nn.Linear) and "attn" in name.lower():
                    attn_linears.append((name, m.in_features, m.out_features))
            # Heuristic: if we find ≥3 same-sized Linear in attn, assume Q,K,V
            if len(attn_linears) >= 3:
                sizes = [(in_f, out_f) for _, in_f, out_f in attn_linears]
                # Most common size = Q/K/V size
                from collections import Counter
                common = Counter(sizes).most_common(1)[0][0]
                assigned = 0
                for name, in_f, out_f in attn_linears:
                    if (in_f, out_f) == common and assigned < 2:
                        kind = "q" if assigned == 0 else "v"
                        self._detected["qv_layers"].append(
                            (idx, name, in_f, out_f, kind))
                        assigned += 1
                if assigned and idx == 0:
                    print(f"  Found Q/V via fallback: Linear({common[0]}, {common[1]})")

    # ─── Input embed hook ─────────────────────────────────────────────

    def _install_input_embed_hook(self) -> int:
        if not hasattr(self.dit, "input_embed"):
            return 0

        parent = self
        prompt_adapter = self.prompt_adapter
        lang_emb = self.language_embedding

        def hook(module, input, output):
            x = output
            x = x + prompt_adapter(x)

            if lang_emb is not None and parent._current_lang_id is not None:
                lid = parent._current_lang_id
                if x.shape[0] != lid.shape[0]:
                    lid = lid.repeat(x.shape[0] // max(lid.shape[0], 1))
                x = x + lang_emb(lid, x.shape[1])

            if parent._current_emotion_bias is not None:
                emo = parent._current_emotion_bias
                if x.shape[0] != emo.shape[0]:
                    emo = emo.repeat(x.shape[0] // max(emo.shape[0], 1), 1)
                x = x + emo.unsqueeze(1)

            return x

        h = self.dit.input_embed.register_forward_hook(hook)
        self._hooks.append(h)
        return 1

    # ─── ConvNeXt hooks ───────────────────────────────────────────────

    def _install_convnext_hooks(self) -> int:
        if not hasattr(self.dit, "text_embed"):
            return 0
        blocks = self._find_convnext_blocks(self.dit.text_embed)

        if not blocks:
            if len(self.cond_adapters) > 0:
                adapters = self.cond_adapters

                def fallback(module, input, output):
                    if output.dim() != 3:
                        return output
                    h = output.transpose(1, 2)
                    for a in adapters:
                        h = a(h)
                    return h.transpose(1, 2)

                h = self.dit.text_embed.register_forward_hook(fallback)
                self._hooks.append(h)
                return 1
            return 0

        count = 0
        for i, block in enumerate(blocks):
            if i >= len(self.cond_adapters):
                break
            adapter = self.cond_adapters[i]

            def make_hook(adpt):
                def hook(module, input, output):
                    if output.dim() != 3:
                        return output
                    h = output.transpose(1, 2)
                    h = adpt(h)
                    return h.transpose(1, 2)
                return hook

            h = block.register_forward_hook(make_hook(adapter))
            self._hooks.append(h)
            count += 1
        return count

    # ─── Block emotion hooks ──────────────────────────────────────────

    def _install_block_hooks(self) -> int:
        if not hasattr(self.dit, "transformer_blocks"):
            return 0
        parent = self
        count = 0
        for idx, block in enumerate(self.dit.transformer_blocks):
            if idx >= len(self.emotion_cond_layers):
                break
            el = self.emotion_cond_layers[idx]

            def make_hook(layer):
                def hook(module, input, output):
                    if parent._current_emotion_emb is None:
                        return output
                    emo = parent._current_emotion_emb
                    if output.shape[0] != emo.shape[0]:
                        emo = emo.repeat(output.shape[0] // max(emo.shape[0], 1), 1)
                    return layer(output, emo)
                return hook

            h = block.register_forward_hook(make_hook(el))
            self._hooks.append(h)
            count += 1
        return count

    # ================================================================
    # Forward methods
    # ================================================================

    def forward(self, inp, text, emotion_emb=None, lang_id=None, **cfm_kwargs):
        """Training via CFM.forward() → returns (loss, cond, pred)."""
        self._set_context(emotion_emb, lang_id)
        try:
            return self.cfm_model(inp, text, **cfm_kwargs)
        finally:
            self._clear_context()

    @torch.no_grad()
    def sample(self, cond, text, duration,
               emotion_emb=None, lang_id=None, **kw):
        """Inference via CFM.sample()."""
        self._set_context(emotion_emb, lang_id)
        try:
            return self.cfm_model.sample(cond=cond, text=text,
                                         duration=duration, **kw)
        finally:
            self._clear_context()

    def predict_velocity(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        time: torch.Tensor,
        mask: torch.Tensor = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        emotion_emb: torch.Tensor = None,
        lang_id: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Direct DiT.forward() — adapters applied via hooks.
        Handles DiT.forward() signature variations automatically.
        """
        self._set_context(emotion_emb, lang_id)
        try:
            kwargs = {}
            params = self._dit_forward_params

            if "drop_audio_cond" in params:
                kwargs["drop_audio_cond"] = drop_audio_cond
            if "drop_text" in params:
                kwargs["drop_text"] = drop_text

            # Handle mask parameter name variations across F5-TTS versions
            if mask is not None:
                if "mask" in params:
                    kwargs["mask"] = mask
                elif "audio_mask" in params:
                    kwargs["audio_mask"] = mask
                elif "padding_mask" in params:
                    kwargs["padding_mask"] = mask

            return self.dit(x, cond, text, time, **kwargs)
        finally:
            self._clear_context()

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

    # ================================================================
    # Save / Load
    # ================================================================

    def get_adapter_state_dict(self) -> dict:
        return {
            name: param.data.clone()
            for name, param in self.named_parameters()
            if param.requires_grad
        }

    def load_adapter_state_dict(self, state_dict: dict, strict: bool = False):
        own = dict(self.named_parameters())
        loaded = skipped = 0
        for name, param in state_dict.items():
            if name in own and own[name].requires_grad:
                own[name].data.copy_(param)
                loaded += 1
            else:
                skipped += 1
        print(f"[EnhancedF5TTS] Loaded {loaded}/{len(state_dict)} adapter params"
              f"{f' (skipped {skipped})' if skipped else ''}")

    # ================================================================
    # Diagnostics
    # ================================================================

    def print_model_structure(self, max_depth=3):
        """Print DiT structure for debugging dimension mismatches."""
        print("\n=== DiT Module Structure ===")
        for name, module in self.dit.named_modules():
            depth = name.count(".")
            if depth > max_depth:
                continue
            if isinstance(module, nn.Linear):
                print(f"  {name}: Linear({module.in_features}, {module.out_features})")
            elif isinstance(module, nn.Conv1d):
                print(f"  {name}: Conv1d({module.in_channels}, {module.out_channels}, "
                      f"k={module.kernel_size})")
            elif isinstance(module, nn.Embedding):
                print(f"  {name}: Embedding({module.num_embeddings}, "
                      f"{module.embedding_dim})")
            elif isinstance(module, nn.LayerNorm):
                print(f"  {name}: LayerNorm({module.normalized_shape})")
        print("===========================\n")


# =========================================================================
# Hook helpers
# =========================================================================

def _make_lora_hook(lora: DiTLoRAAdapter):
    """Hook for separate Q or V: output += LoRA(input)."""
    def hook(module, input, output):
        return output + lora(input[0])
    return hook


def _make_fused_qkv_lora_hook(lora_q, lora_v, split_size):
    """Hook for fused to_qkv: apply LoRA to Q and V portions."""
    def hook(module, input, output):
        q_out, k_out, v_out = output.split(split_size, dim=-1)
        x = input[0]
        q_out = q_out + lora_q(x)
        v_out = v_out + lora_v(x)
        return torch.cat([q_out, k_out, v_out], dim=-1)
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
    if model_cfg is None:
        model_cfg = {
            "dim": 1024, "depth": 22, "heads": 16,
            "ff_mult": 2, "text_dim": 512, "conv_layers": 4,
        }

    from f5_tts.model import CFM, DiT
    from f5_tts.model.utils import get_tokenizer

    vocab_char_map, vocab_size = get_tokenizer(vocab_path, "custom")
    dit = DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100)

    cfm = CFM(
        transformer=dit,
        mel_spec_kwargs=dict(
            n_fft=1024, hop_length=256, win_length=1024,
            n_mel_channels=100, target_sample_rate=24000,
            mel_spec_type="vocos",
        ),
        vocab_char_map=vocab_char_map,
    )

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

    enhanced.print_model_structure(max_depth=2)
    enhanced.install_hooks()
    enhanced = enhanced.to(device)
    return enhanced
