"""
Cross-Language PEFT Adapters
============================
Три типа адаптеров для эффективного файнтюнинга F5-TTS на новые языки
при сохранении способностей базовой модели.

На основе PEFT-TTS (Interspeech 2025): Conditioning + Prompt + DiT LoRA адаптеры.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# 1. Conditioning Adapter — встраивается в ConvNeXt V2 блоки
# =============================================================================

class ConditioningAdapter(nn.Module):
    """
    Conv-Adapter для ConvNeXt V2 блоков в Text Embedding.
    Depth-wise conv → Point-wise conv → SE modulation.
    Помогает адаптировать текстовые представления под фонетику нового языка.

    Параметры: ~50K per adapter (compression=0.25)
    """

    def __init__(
        self,
        channels: int,
        compression_factor: float = 0.25,
        kernel_size: int = 3,
    ):
        super().__init__()
        hidden = max(int(channels * compression_factor), 16)

        # Depth-wise conv
        self.dw_conv = nn.Conv1d(
            channels, channels, kernel_size,
            padding=kernel_size // 2, groups=channels, bias=False
        )

        # Point-wise down-projection
        self.pw_down = nn.Conv1d(channels, hidden, 1, bias=False)
        self.act = nn.GELU()

        # Point-wise up-projection
        self.pw_up = nn.Conv1d(hidden, channels, 1, bias=False)

        # Squeeze-and-Excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

        # Residual scaling (initialized small)
        self.scale = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.dw_conv.weight)
        nn.init.kaiming_normal_(self.pw_down.weight)
        nn.init.zeros_(self.pw_up.weight)  # near-zero init for residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len) — output of ConvNeXt depth-wise conv
        Returns:
            adapted: (batch, channels, seq_len)
        """
        residual = x

        h = self.dw_conv(x)
        h = self.pw_down(h)
        h = self.act(h)
        h = self.pw_up(h)

        # SE modulation
        se_weight = self.se(h).unsqueeze(-1)  # (B, C, 1)
        h = h * se_weight

        return residual + self.scale * h


# =============================================================================
# 2. Prompt Adapter — на Input Embedding (конкатенация text + audio)
# =============================================================================

class PromptAdapter(nn.Module):
    """
    LoRA-адаптер для линейной проекции после конкатенации text и audio фич.
    Включает DropPath для регуляризации при обучении на single-speaker данных.

    Балансирует trade-off: произношение ↔ сохранение голоса.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.05,
        drop_path_rate: float = 0.3,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.drop_path_rate = drop_path_rate

        # DropPath for regularization
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, in_features) — вход линейного слоя
        Returns:
            delta: (batch, seq_len, out_features) — добавка к основному выходу
        """
        h = self.dropout(x)
        h = self.lora_A(h)
        h = self.lora_B(h)
        h = h * self.scaling
        h = self.drop_path(h)
        return h


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    Randomly drops entire residual branches during training.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output


# =============================================================================
# 3. DiT LoRA Adapter — для Q, V проекций в Multi-Head Self-Attention
# =============================================================================

class DiTLoRAAdapter(nn.Module):
    """
    LoRA адаптер для DiT блоков.
    Применяется к Q и V проекциям в MHSA.

    rank=16 оптимален: сохраняет гибкость модели, не overfitting на speaker.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(x)
        h = self.lora_A(h)
        h = self.lora_B(h)
        return h * self.scaling


# =============================================================================
# 4. Language Embedding — опциональный language ID для cross-lingual
# =============================================================================

class LanguageEmbedding(nn.Module):
    """
    Learnable embedding для языка, добавляется к входу DiT.
    Помогает модели различать фонетические системы разных языков.
    """

    LANG_MAP = {
        "en": 0, "zh": 1, "ru": 2, "de": 3, "fr": 4,
        "es": 5, "ja": 6, "ko": 7, "it": 8, "pt": 9,
        "ar": 10, "hi": 11, "other": 12,
    }

    def __init__(self, embed_dim: int, num_languages: int = 13):
        super().__init__()
        self.embedding = nn.Embedding(num_languages, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.proj.weight)

    def forward(self, lang_id: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Args:
            lang_id: (batch,) — language index
            seq_len: int — sequence length для expand
        Returns:
            (batch, seq_len, embed_dim)
        """
        emb = self.embedding(lang_id)  # (B, D)
        emb = self.proj(emb)
        return emb.unsqueeze(1).expand(-1, seq_len, -1)

    @classmethod
    def get_lang_id(cls, lang_code: str) -> int:
        return cls.LANG_MAP.get(lang_code, cls.LANG_MAP["other"])


# =============================================================================
# 5. Emotion Conditioning Layer — внедряет эмоцию в DiT блоки
# =============================================================================

class EmotionConditioningLayer(nn.Module):
    """
    Cross-attention слой для внедрения эмоционального вектора в DiT.
    Эмоция из reference аудио модулирует генерацию через AdaLN-подобный механизм.

    Два режима:
    1. AdaLN-style: scale + shift модуляция (быстрый, стабильный)
    2. Cross-attention: query=DiT hidden, key/value=emotion (более выразительный)
    """

    def __init__(
        self,
        hidden_dim: int,
        emotion_dim: int = 768,  # emotion2vec native dimension
        mode: str = "adaln",  # "adaln" or "cross_attention"
        num_heads: int = 4,
    ):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim

        if mode == "adaln":
            # Adaptive Layer Norm — эмоция модулирует scale и shift
            self.emotion_mlp = nn.Sequential(
                nn.Linear(emotion_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
            )
            # Инициализация near-zero чтобы не ломать базовую модель
            nn.init.zeros_(self.emotion_mlp[-1].weight)
            nn.init.zeros_(self.emotion_mlp[-1].bias)

        elif mode == "cross_attention":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True,
            )
            self.emotion_proj = nn.Linear(emotion_dim, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)
            self.scale = nn.Parameter(torch.tensor(0.0))  # gated

    def forward(
        self,
        hidden: torch.Tensor,
        emotion_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden: (batch, seq_len, hidden_dim) — DiT hidden states
            emotion_emb: (batch, emotion_dim) — emotion embedding
        Returns:
            modulated: (batch, seq_len, hidden_dim)
        """
        if self.mode == "adaln":
            params = self.emotion_mlp(emotion_emb)  # (B, hidden_dim * 2)
            scale, shift = params.chunk(2, dim=-1)   # каждый (B, hidden_dim)
            scale = scale.unsqueeze(1)  # (B, 1, D)
            shift = shift.unsqueeze(1)  # (B, 1, D)
            return hidden * (1 + scale) + shift

        elif self.mode == "cross_attention":
            emotion_kv = self.emotion_proj(emotion_emb).unsqueeze(1)  # (B, 1, D)
            attn_out, _ = self.cross_attn(
                query=self.norm(hidden),
                key=emotion_kv,
                value=emotion_kv,
            )
            return hidden + self.scale * attn_out

        return hidden


# =============================================================================
# Helper: Inject adapters into existing F5-TTS model
# =============================================================================

def inject_adapters_into_f5tts(
    model: nn.Module,
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
    conditioning_compression: float = 0.25,
    prompt_drop_path: float = 0.3,
    emotion_dim: int = 768,
    emotion_mode: str = "adaln",
    add_language_emb: bool = True,
    freeze_base: bool = True,
) -> dict:
    """
    Внедряет PEFT-адаптеры в существующую модель F5-TTS.

    Args:
        model: предобученная F5-TTS модель
        lora_rank: ранг LoRA для DiT блоков
        ...

    Returns:
        dict с информацией о добавленных модулях

    Пример использования:
        >>> from f5_tts.model import DiT  # оригинальный F5-TTS
        >>> model = DiT(...)  # загрузить предобученную
        >>> adapters_info = inject_adapters_into_f5tts(model, lora_rank=16)
    """
    added_modules = {}

    # Заморозить базовую модель
    if freeze_base:
        for name, param in model.named_parameters():
            param.requires_grad = False

    # 1. Conditioning Adapters для ConvNeXt блоков
    if hasattr(model, "text_embed") or hasattr(model, "input_embed"):
        # Ищем ConvNeXt блоки в text embedding
        for name, module in model.named_modules():
            if "conv" in name.lower() and "next" in name.lower():
                if hasattr(module, "dwconv") or hasattr(module, "dw_conv"):
                    channels = _get_channels(module)
                    if channels > 0:
                        adapter = ConditioningAdapter(
                            channels=channels,
                            compression_factor=conditioning_compression,
                        )
                        # Регистрируем как дочерний модуль
                        adapter_name = f"cond_adapter_{name.replace('.', '_')}"
                        model.add_module(adapter_name, adapter)
                        added_modules[adapter_name] = adapter

    # 2. DiT LoRA Adapters для Q, V проекций
    dit_blocks = _find_dit_blocks(model)
    for block_idx, block in enumerate(dit_blocks):
        # Ищем attention Q и V проекции
        for proj_name in ["to_q", "to_v", "wq", "wv", "q_proj", "v_proj"]:
            proj = _get_submodule(block, proj_name)
            if proj is not None and isinstance(proj, nn.Linear):
                lora = DiTLoRAAdapter(
                    in_features=proj.in_features,
                    out_features=proj.out_features,
                    rank=lora_rank,
                    alpha=lora_alpha,
                )
                adapter_name = f"dit_lora_block{block_idx}_{proj_name}"
                model.add_module(adapter_name, lora)
                added_modules[adapter_name] = lora

        # 3. Emotion Conditioning Layer (по одному на блок)
        hidden_dim = _get_hidden_dim(block)
        if hidden_dim > 0:
            emo_layer = EmotionConditioningLayer(
                hidden_dim=hidden_dim,
                emotion_dim=emotion_dim,
                mode=emotion_mode,
            )
            emo_name = f"emotion_cond_block{block_idx}"
            model.add_module(emo_name, emo_layer)
            added_modules[emo_name] = emo_layer

    # 4. Language Embedding
    if add_language_emb:
        hidden_dim = _get_model_hidden_dim(model)
        if hidden_dim > 0:
            lang_emb = LanguageEmbedding(embed_dim=hidden_dim)
            model.add_module("language_embedding", lang_emb)
            added_modules["language_embedding"] = lang_emb

    # Подсчёт параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        "added_modules": added_modules,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": 100.0 * trainable_params / total_params,
    }

    print(f"[PEFT] Total params: {total_params:,}")
    print(f"[PEFT] Trainable params: {trainable_params:,} ({info['trainable_pct']:.2f}%)")

    return info


# =============================================================================
# Utility functions
# =============================================================================

def _get_channels(module: nn.Module) -> int:
    for attr in ["out_channels", "in_channels", "channels"]:
        if hasattr(module, attr):
            return getattr(module, attr)
    return 0


def _find_dit_blocks(model: nn.Module) -> list:
    """Находит DiT блоки (трансформерные слои) в модели."""
    blocks = []
    for name, module in model.named_modules():
        module_type = type(module).__name__.lower()
        if any(k in module_type for k in ["ditblock", "transformerblock", "basicblock"]):
            blocks.append(module)
    # Fallback: ищем в model.transformer_blocks или model.blocks
    if not blocks:
        for attr in ["transformer_blocks", "blocks", "layers", "dit_blocks"]:
            if hasattr(model, attr):
                candidate = getattr(model, attr)
                if isinstance(candidate, (nn.ModuleList, list)):
                    blocks = list(candidate)
                    break
    return blocks


def _get_submodule(module: nn.Module, name: str) -> Optional[nn.Module]:
    """Безопасно получает подмодуль по имени."""
    for n, m in module.named_modules():
        if n.split(".")[-1] == name:
            return m
    return None


def _get_hidden_dim(block: nn.Module) -> int:
    """Определяет hidden dimension DiT блока."""
    for name, module in block.named_modules():
        if isinstance(module, nn.LayerNorm):
            return module.normalized_shape[0]
        if isinstance(module, nn.Linear):
            return module.in_features
    return 0


def _get_model_hidden_dim(model: nn.Module) -> int:
    """Определяет hidden dimension модели."""
    if hasattr(model, "dim"):
        return model.dim
    if hasattr(model, "hidden_size"):
        return model.hidden_size
    if hasattr(model, "config") and hasattr(model.config, "dim"):
        return model.config.dim
    return 1024  # default for F5-TTS Base


def get_trainable_params(model: nn.Module) -> list:
    """Возвращает только trainable параметры (для optimizer)."""
    return [p for p in model.parameters() if p.requires_grad]


def save_adapters(model: nn.Module, save_path: str):
    """Сохраняет только adapter-параметры (trainable)."""
    adapter_state = {
        name: param.data
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    torch.save(adapter_state, save_path)
    print(f"[PEFT] Saved {len(adapter_state)} adapter params to {save_path}")


def load_adapters(model: nn.Module, load_path: str):
    """Загружает adapter-параметры."""
    adapter_state = torch.load(load_path, map_location="cpu")
    model_state = model.state_dict()

    loaded = 0
    for name, param in adapter_state.items():
        if name in model_state:
            model_state[name].copy_(param)
            loaded += 1

    print(f"[PEFT] Loaded {loaded}/{len(adapter_state)} adapter params from {load_path}")
