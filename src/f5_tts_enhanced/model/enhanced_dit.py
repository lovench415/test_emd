"""
Enhanced DiT Wrapper
====================
Обёртка над оригинальным DiT из F5-TTS, интегрирующая:
- Emotion conditioning (AdaLN / Cross-Attention)
- LoRA adapters для Q,V проекций
- Language embedding
- Conditioning adapters для ConvNeXt

Не меняет исходный код F5-TTS — работает как monkey-patch wrapper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union
from functools import wraps

from .emotion_extractor import Emotion2vecExtractor, create_emotion_extractor
from .adapters import (
    ConditioningAdapter,
    PromptAdapter,
    DiTLoRAAdapter,
    LanguageEmbedding,
    EmotionConditioningLayer,
)


class EnhancedF5TTS(nn.Module):
    """
    Обёртка над F5-TTS DiT моделью с дополнительными модулями.

    Использование:
        >>> from f5_tts.model import DiT
        >>> base_model = DiT(...)  # загрузить базовую
        >>> enhanced = EnhancedF5TTS(base_model, config)
        >>> # Теперь enhanced.forward() принимает emotion и language
    """

    def __init__(
        self,
        base_model: nn.Module,
        emotion_dim: int = 768,
        emotion_mode: str = "adaln",  # "adaln" or "cross_attention"
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        cond_adapter_compression: float = 0.25,
        prompt_drop_path: float = 0.3,
        add_language_emb: bool = True,
        freeze_base: bool = True,
        num_dit_blocks: int = 22,
        hidden_dim: int = 1024,
        emotion_model_size: str = "large",  # emotion2vec model variant
    ):
        super().__init__()

        self.base_model = base_model
        self.hidden_dim = hidden_dim
        self.emotion_dim = emotion_dim
        self.emotion_mode = emotion_mode

        # Заморозить базовую модель
        if freeze_base:
            for param in base_model.parameters():
                param.requires_grad = False

        # ---- Emotion Extractor (pretrained emotion2vec — no training needed) ----
        self.emotion_extractor = Emotion2vecExtractor(
            model_size=emotion_model_size,
            emotion_dim=emotion_dim,
        )

        # ---- Emotion Conditioning Layers (per DiT block) ----
        self.emotion_cond_layers = nn.ModuleList([
            EmotionConditioningLayer(
                hidden_dim=hidden_dim,
                emotion_dim=emotion_dim,
                mode=emotion_mode,
            )
            for _ in range(num_dit_blocks)
        ])

        # ---- LoRA Adapters для DiT Q,V (per block) ----
        self.lora_q = nn.ModuleList([
            DiTLoRAAdapter(hidden_dim, hidden_dim, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
            for _ in range(num_dit_blocks)
        ])
        self.lora_v = nn.ModuleList([
            DiTLoRAAdapter(hidden_dim, hidden_dim, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
            for _ in range(num_dit_blocks)
        ])

        # ---- Conditioning Adapters для ConvNeXt ----
        # Предполагается 4 ConvNeXt блока с text_dim=512
        text_dim = 512
        self.cond_adapters = nn.ModuleList([
            ConditioningAdapter(text_dim, compression_factor=cond_adapter_compression)
            for _ in range(4)
        ])

        # ---- Prompt Adapter ----
        # Применяется к проекции после concat text + audio
        # F5-TTS: input_embed проецирует mel_dim*2 + text_dim → hidden_dim
        input_proj_dim = hidden_dim
        self.prompt_adapter = PromptAdapter(
            in_features=input_proj_dim,
            out_features=input_proj_dim,
            rank=lora_rank,
            drop_path_rate=prompt_drop_path,
        )

        # ---- Language Embedding ----
        self.language_embedding = None
        if add_language_emb:
            self.language_embedding = LanguageEmbedding(embed_dim=hidden_dim)

        # Emotion projection to match timestep embedding format
        self.emotion_to_timestep = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._count_params()

    def _count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        base = sum(p.numel() for p in self.base_model.parameters())
        print(f"[EnhancedF5TTS] Base model: {base:,} params")
        print(f"[EnhancedF5TTS] Total: {total:,} params")
        print(f"[EnhancedF5TTS] Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

    def load_emotion_extractor(self):
        """Загрузить emotion2vec модель (вызвать перед инференсом)."""
        self.emotion_extractor.load_model()

    def extract_emotion_from_audio(
        self,
        ref_audio: Union[str, torch.Tensor],
        sr: int = 16000,
    ) -> torch.Tensor:
        """
        Извлечь эмоциональный эмбеддинг из reference аудио.

        Args:
            ref_audio: путь к файлу или waveform tensor
            sr: sample rate (для tensor)
        Returns:
            emotion_emb: (emotion_dim,) — по умолчанию (768,)
        """
        return self.emotion_extractor.get_emotion_embedding(ref_audio, sr=sr)

    def forward(
        self,
        x: torch.Tensor,           # noisy input mel
        cond: torch.Tensor,         # condition (masked reference mel)
        text: torch.Tensor,         # text input (padded)
        time: torch.Tensor,         # diffusion timestep
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        mask: Optional[torch.Tensor] = None,
        # --- Enhanced inputs ---
        emotion_emb: Optional[torch.Tensor] = None,  # (B, emotion_dim)
        lang_id: Optional[torch.Tensor] = None,       # (B,) language index
    ) -> torch.Tensor:
        """
        Enhanced forward pass.

        Если emotion_emb не предоставлен, модель работает в обычном режиме F5-TTS.
        Это обеспечивает обратную совместимость.
        """
        # Делегируем основную логику базовой модели,
        # но перехватываем промежуточные вычисления
        # через hook-based подход

        # Подготовка emotion conditioning
        emotion_bias = None
        if emotion_emb is not None:
            emotion_bias = self.emotion_to_timestep(emotion_emb)  # (B, D)

        # Language embedding bias
        lang_bias = None

        # Собираем extra conditioning
        extra_cond = {
            "emotion_emb": emotion_emb,
            "emotion_bias": emotion_bias,
            "lang_id": lang_id,
        }

        # Вызываем forward базовой модели через patched версию
        return self._enhanced_forward(x, cond, text, time, drop_audio_cond, drop_text, mask, extra_cond)

    def _enhanced_forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        time: torch.Tensor,
        drop_audio_cond: bool,
        drop_text: bool,
        mask: Optional[torch.Tensor],
        extra_cond: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Реплицирует forward pass F5-TTS DiT с инъекцией адаптеров.

        Это основная функция, которую нужно адаптировать под конкретную
        версию F5-TTS (v1 vs Base).
        """
        model = self.base_model
        batch_size = x.shape[0]

        # ------ Text Embedding с Conditioning Adapters ------
        # Оригинал: text_embed = model.text_embed(text)
        # Мы вставляем cond_adapters после каждого ConvNeXt блока

        # Если прямой доступ к промежуточным слоям недоступен,
        # используем hook-based подход (см. register_hooks)
        text_embed = self._text_embed_with_adapters(model, text, drop_text)

        # ------ Input Embedding + Prompt Adapter ------
        # Оригинал: inp = model.input_embed(x_noisy, cond, text_embed, time)
        if hasattr(model, "input_embed"):
            inp = model.input_embed(x, cond, text_embed, time)
        else:
            # Fallback для разных версий
            inp = self._default_input_embed(model, x, cond, text_embed, time, drop_audio_cond)

        # Добавляем Prompt Adapter
        inp = inp + self.prompt_adapter(inp)

        # Добавляем Language Embedding
        if self.language_embedding is not None and extra_cond.get("lang_id") is not None:
            lang_bias = self.language_embedding(extra_cond["lang_id"], inp.shape[1])
            inp = inp + lang_bias

        # Добавляем emotion bias к timestep embedding
        if extra_cond.get("emotion_bias") is not None:
            # emotion_bias добавляется ко всем позициям
            inp = inp + extra_cond["emotion_bias"].unsqueeze(1)

        # ------ DiT Blocks с LoRA + Emotion Conditioning ------
        hidden = inp
        dit_blocks = self._get_dit_blocks(model)

        for block_idx, block in enumerate(dit_blocks):
            # Стандартный forward блока
            hidden = block(hidden) if mask is None else block(hidden, mask=mask)

            # LoRA residual для Q, V
            if block_idx < len(self.lora_q):
                # LoRA добавляется как residual — в идеале нужен hook внутрь block
                # Упрощённый вариант: post-block modulation
                pass  # LoRA интегрируется через register_hooks() ниже

            # Emotion conditioning
            if extra_cond.get("emotion_emb") is not None and block_idx < len(self.emotion_cond_layers):
                hidden = self.emotion_cond_layers[block_idx](hidden, extra_cond["emotion_emb"])

        # ------ Final layer ------
        if hasattr(model, "final_layer"):
            output = model.final_layer(hidden)
        elif hasattr(model, "to_out"):
            output = model.to_out(hidden)
        else:
            output = hidden

        return output

    def _text_embed_with_adapters(
        self, model: nn.Module, text: torch.Tensor, drop_text: bool
    ) -> torch.Tensor:
        """Прогон text embedding с Conditioning Adapters."""
        if hasattr(model, "text_embed"):
            # Простой вариант: вызываем text_embed и post-process
            text_emb = model.text_embed(text, drop_text=drop_text) if drop_text else model.text_embed(text)
            # Транспонируем для conv1d: (B, T, D) → (B, D, T)
            if text_emb.dim() == 3 and len(self.cond_adapters) > 0:
                h = text_emb.transpose(1, 2)
                for adapter in self.cond_adapters:
                    h = adapter(h)
                text_emb = h.transpose(1, 2)
            return text_emb
        return text

    def _default_input_embed(self, model, x, cond, text_embed, time, drop_audio_cond):
        """Fallback input embedding."""
        return x  # Override в конкретной имплементации

    def _get_dit_blocks(self, model: nn.Module) -> list:
        """Извлечь DiT блоки из модели."""
        for attr in ["transformer_blocks", "blocks", "layers", "dit_blocks"]:
            if hasattr(model, attr):
                return list(getattr(model, attr))
        return []

    def register_lora_hooks(self):
        """
        Регистрирует forward hooks для инъекции LoRA в Q,V проекции
        внутри DiT блоков (не требует изменения исходного кода F5-TTS).
        """
        dit_blocks = self._get_dit_blocks(self.base_model)

        for block_idx, block in enumerate(dit_blocks):
            if block_idx >= len(self.lora_q):
                break

            lora_q = self.lora_q[block_idx]
            lora_v = self.lora_v[block_idx]

            # Ищем Q и V проекции
            for name, module in block.named_modules():
                if isinstance(module, nn.Linear):
                    if any(q_name in name for q_name in ["to_q", "q_proj", "wq"]):
                        self._register_lora_hook(module, lora_q)
                    elif any(v_name in name for v_name in ["to_v", "v_proj", "wv"]):
                        self._register_lora_hook(module, lora_v)

    @staticmethod
    def _register_lora_hook(linear: nn.Linear, lora: DiTLoRAAdapter):
        """Регистрирует hook, добавляющий LoRA output к Linear."""
        def hook(module, input, output):
            return output + lora(input[0])
        linear.register_forward_hook(hook)


def create_enhanced_model(
    base_model_path: str = "SWivid/F5-TTS",
    model_name: str = "F5TTS_v1_Base",
    device: str = "cuda",
    emotion_dim: int = 768,
    lora_rank: int = 16,
    freeze_base: bool = True,
) -> EnhancedF5TTS:
    """
    Фабричная функция для создания enhanced модели.

    Пример:
        >>> model = create_enhanced_model()
        >>> model.load_emotion_extractor()
        >>> model.register_lora_hooks()
        >>> model = model.to("cuda")
    """
    # Загружаем базовую F5-TTS модель
    try:
        from f5_tts.infer.utils_infer import load_model as f5_load_model
        base_model = f5_load_model(model_name, device=device)
    except ImportError:
        print("[WARNING] f5_tts not installed, creating placeholder model")
        base_model = nn.Identity()

    # Определяем параметры архитектуры
    config_map = {
        "F5TTS_v1_Base": {"hidden_dim": 1024, "num_dit_blocks": 22},
        "F5TTS_Base": {"hidden_dim": 1024, "num_dit_blocks": 22},
        "F5TTS_Small": {"hidden_dim": 768, "num_dit_blocks": 18},
    }
    arch_config = config_map.get(model_name, config_map["F5TTS_v1_Base"])

    enhanced = EnhancedF5TTS(
        base_model=base_model,
        emotion_dim=emotion_dim,
        lora_rank=lora_rank,
        freeze_base=freeze_base,
        **arch_config,
    )

    return enhanced
