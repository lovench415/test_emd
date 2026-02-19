# F5-TTS Enhanced: Architecture Documentation

## Обзор

**F5-TTS Enhanced** — усовершенствованная архитектура на базе F5-TTS для кросс-лингвального клонирования голоса с передачей эмоций. Референсное аудио может быть на **любом языке**, а генерация производится на **русском** (или любом целевом языке).

### Ключевые улучшения

| Компонент | Оригинал F5-TTS | Enhanced |
|-----------|----------------|----------|
| Голос (тембр) | Только через mel-conditioning | + Speaker Embedding (WavLM-SV) |
| Эмоции | Не поддерживается | Emotion Embedding (emotion2vec) |
| Кросс-лингвальность | Ограничена mel-matching | Language-agnostic embeddings |
| Контроль эмоций | — | Emotion-Guided CFG (runtime) |
| Временной контроль | — | Frame-level emotion features |
| Обучение | С нуля | Finetune только conditioning модулей |

---

## 1. Общая архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE PIPELINE                          │
│                                                                 │
│  Reference Audio ──┬── SpeakerEncoder ──→ speaker_emb (b, 512) │
│  (любой язык)      │                                            │
│                    ├── EmotionEncoder ──→ emotion_global (b,512)│
│                    │                  └→ emotion_frame (b,T,512)│
│                    │                                            │
│                    └── Mel Spectrogram ──→ cond (b, n, 100)     │
│                                                                 │
│  Target Text ─────── TextEmbedding ──→ text_embed (b, n, 512)  │
│  (русский)                                                      │
│                                                                 │
│              ┌──────────────────────────────┐                   │
│              │      EnhancedDiT             │                   │
│              │                              │                   │
│              │  ┌─ InputEmbedding ──────┐   │                   │
│              │  │  + EmbeddingAdd(fused) │   │                   │
│              │  └───────────────────────┘   │                   │
│              │           ↓                  │                   │
│              │  ┌─ DiTBlock 0 ──────────┐   │                   │
│              │  │  + AdaLN residual     │   │                   │
│              │  │  + CrossAttn(emo_frm) │   │                   │
│              │  └───────────────────────┘   │                   │
│              │           ↓                  │                   │
│              │       ... × 22 blocks        │                   │
│              │           ↓                  │                   │
│              │  ┌─ Output Projection ───┐   │                   │
│              │  │  norm → proj → mel    │   │                   │
│              │  └───────────────────────┘   │                   │
│              └──────────────────────────────┘                   │
│                          ↓                                      │
│              ODE Solver (Euler) + EG-CFG                        │
│                          ↓                                      │
│              Vocoder (Vocos) → Waveform                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Компоненты

### 2.1 Speaker Encoder (`speaker_encoder.py`)

**Назначение:** Извлечение language-agnostic вектора идентичности голоса.

**Архитектура:**
```
Raw Waveform (любой SR)
    ↓ resample to 16kHz
Pretrained WavLM-SV (frozen)
    ↓ 512-d x-vector
Trainable Projection MLP
    ↓ Linear(512→512) → SiLU → Linear(512→512)
L2-Normalize
    ↓
speaker_emb: (batch, 512)
```

**Почему WavLM-SV:**
- State-of-the-art speaker verification, обученная на VoxCeleb
- Работает на уровне utterance (один вектор на высказывание)
- Language-agnostic: тембр голоса не зависит от языка
- 512-d компактное представление

**Альтернативные бэкенды:**
- `ecapa_tdnn` — SpeechBrain ECAPA-TDNN (192-d, быстрый)
- `resemblyzer` — GE2E encoder (256-d, лёгкий для CPU)

### 2.2 Emotion Encoder (`emotion_encoder.py`)

**Назначение:** Извлечение эмоциональных характеристик — глобальных и покадровых.

**Архитектура:**
```
Raw Waveform (любой SR)
    ↓ resample to 16kHz
Pretrained emotion2vec (frozen)
    ↓
    ├── Frame-level features: (batch, T_frames, 768)
    │       ↓ Trainable FrameProj MLP
    │       ↓ Linear(768→512) → SiLU → Linear(512→512)
    │       ↓ DepthwiseConv1d (temporal smoothing, kernel=5)
    │       ↓ Interpolate to target mel length
    │       ↓ L2-Normalize
    │       → emotion_frame: (batch, mel_frames, 512)
    │
    └── Mean Pool over time → (batch, 768)
            ↓ Trainable GlobalProj MLP
            ↓ Linear(768→512) → SiLU → Linear(512→512)
            ↓ L2-Normalize
            → emotion_global: (batch, 512)
```

**Почему emotion2vec:**
- Self-supervised модель для эмоциональных представлений
- Мультиязычная (обучена на данных из разных языков)
- Даёт frame-level фичи для временного контроля эмоций
- 768-d богатое представление эмоционального спектра

**Почему два уровня (global + frame):**
- **Global**: общая эмоция высказывания (радость, грусть, гнев...)
- **Frame**: временная динамика эмоции (нарастание, кульминация, спад)
- Frame-level фичи обрабатываются temporal smoothing (Conv1d) для плавности

### 2.3 Conditioning Module (`conditioning.py`)

**Назначение:** Fusion и инъекция эмбеддингов в DiT backbone.

#### Стратегия 1: AdaLN Modulation (EmoSteer-TTS)

```
speaker_emb ─┐
             ├─ concat → Fusion MLP → fused_emb
emotion_global┘
                    ↓
            PreProj: Linear(1024→1024) → SiLU
                    ↓
            Per-block Projs: Linear(1024→6144) × 22 блоков
                    ↓
            [δshift_msa, δscale_msa, δgate_msa,
             δshift_mlp, δscale_mlp, δgate_mlp]  ← residuals
                    
            В каждом DiT блоке:
            final_gate_msa = timestep_gate_msa + δgate_msa
            (аналогично для остальных 5 параметров)
```

**Инициализация нулями:** Все per-block проекции инициализированы нулями,
поэтому на старте conditioning не влияет на генерацию (identity transform).
Это критически важно для стабильного warm-start от оригинальных весов F5-TTS.

#### Стратегия 2: Cross-Attention (TTS-CtrlNet)

```
DiT hidden states (Query) ←──── Cross-Attention ────→ emotion_frame (Key/Value)
         (b, n, 1024)                                    (b, T, 512)
              ↓
        LayerNorm → Q proj          LayerNorm → K, V proj
              ↓                              ↓
        Multi-Head Attention (8 heads, dim_head=64)
              ↓
        Linear proj → Dropout
              ↓
        Gated Residual: x = x + tanh(gate) × attn_output
                              ↑
                        gate init = 0 (zero-init для стабильности)
```

**Расположение:** каждый 4-й блок DiT (блоки 0, 4, 8, 12, 16, 20).
Это балансирует качество и compute cost.

#### Стратегия 3: Embedding Addition (ece-tts)

```
fused_emb: (batch, 1024)
    ↓ Projection MLP (init near-zero)
proj_emb: (batch, 1024)
    ↓ unsqueeze + broadcast over time
    ↓
x = x + proj_emb  (на уровне input embedding)
```

Простейший метод, но эффективен для глобальной идентичности говорящего.

### 2.4 Enhanced DiT (`backbones/enhanced_dit.py`)

**Изменения относительно оригинального DiT:**

| Слой | Оригинал | Enhanced |
|------|----------|----------|
| InputEmbedding | proj(x‖cond‖text) | + embedding_add(fused) |
| DiTBlock.forward | block(x, t, mask, rope) | + cond_adaln_residual param |
| После каждого 4-го блока | — | + cross_attn(x, emotion_frame) |
| forward() signature | x, cond, text, time | + speaker_emb, emotion_global, emotion_frame |

**Обратная совместимость:**
- Все оригинальные параметры сохраняют те же имена и размеры
- Новые параметры под `cond_aggregator.*`
- `load_state_dict(strict=False)` загружает оригинальные веса F5-TTS

### 2.5 Enhanced CFM (`enhanced_cfm.py`)

#### Training: Multi-Condition Dropout

```python
# Для каждого batch sample:
if random() < 0.2:   # p_uncond
    drop ALL conditions  →  unconditional baseline
else:
    drop_audio  with prob 0.3   (оригинал F5-TTS)
    drop_speaker with prob 0.1  (НОВОЕ)
    drop_emotion with prob 0.1  (НОВОЕ)
```

Это позволяет при инференсе использовать гибкий CFG по любой комбинации условий.

#### Inference: Emotion-Guided CFG (EG-CFG)

Стандартный CFG в F5-TTS:
```
pred = pred_uncond + cfg_strength × (pred_cond − pred_uncond)
```

EG-CFG добавляет отдельное управление эмоциями:
```
pred = pred_uncond
     + cfg_strength × (pred_full − pred_uncond)
     + emotion_cfg_strength × (pred_full − pred_no_emotion)
```

Где:
- `pred_full` — предсказание со ВСЕМИ условиями
- `pred_uncond` — без условий
- `pred_no_emotion` — с голосом но БЕЗ эмоции

**emotion_cfg_strength** контролирует интенсивность:
- `0.0` — нейтральная речь (голос клонирован, но без эмоций)
- `1.0` — естественная передача эмоции из референса
- `2.0+` — усиленная эмоция

---

## 3. Поток данных

### 3.1 Training Flow

```
Audio File → mel_spec (b, t, 100)
           → SpeakerEncoder(frozen) → speaker_raw (b, 512)
           → EmotionEncoder(frozen) → emotion_global_raw (b, 768)
                                    → emotion_frame_raw (b, T, 768)

speaker_raw → SpeakerEncoder.proj (trainable) → speaker_emb (b, 512)
emotion_*_raw → EmotionEncoder.proj (trainable) → emotion_global (b, 512)
                                                → emotion_frame (b, mel_t, 512)

[speaker_emb, emotion_global] → ConditioningAggregator.fusion → fused (b, 1024)
fused → AdaLN projections → adaln_params [22 × (b, 6144)]
fused → InputAdd projection → added to input

EnhancedDiT(φ, cond, text, time, speaker_emb, emotion_global, emotion_frame)
    → predicted flow (b, t, 100)

Loss = MSE(predicted_flow, true_flow) [only on masked span]
```

### 3.2 Inference Flow

```
Reference Audio (EN/ZH/DE/...) → preprocess (trim, normalize)
    → SpeakerEncoder → speaker_emb
    → EmotionEncoder → emotion_global + emotion_frame
    → mel_spec → cond

Target Text (Russian) → TextEmbedding → text_embed

ODE Solver: t=0→1, 32 steps
    At each step t:
        noise x_t → EnhancedDiT(x_t, cond, text, t,
                                  speaker_emb, emotion_global, emotion_frame)
        Apply EG-CFG (3 forward passes: full, no_emotion, uncond)
        → flow prediction → x_{t+1}

Final mel → Vocoder (Vocos) → Waveform (24kHz)
```

---

## 4. Стратегия обучения

### Этап 0: Предвычисление эмбеддингов (offline)

```bash
python precompute_embeddings.py \
    --dataset_path data/Emilia_ZH_EN_pinyin \
    --output_dir data/embeddings_cache \
    --speaker_backend wavlm_sv \
    --emotion_backend emotion2vec_base
```

Сохраняет `.pt` файл на каждый сэмпл (speaker_raw + emotion_raw).
Это **одноразовая** операция, ускоряет обучение в ~5x.

### Этап 1: Finetune conditioning модулей (основной)

```
Заморожено: весь оригинальный F5-TTS (~335M параметров)
Обучается:  ConditioningAggregator (~15-25M параметров)
            - AdaLN projections: 22 × Linear(1024→6144)
            - CrossAttention (6 слоёв): Q/K/V/Out + gate
            - InputAdd projection
            - Fusion MLP
            + Speaker/Emotion projection heads

LR:         3e-4 (высокий, т.к. мало параметров)
Epochs:     3-5
Batch:      ~19K frames/GPU
Warmup:     2000 steps
```

**Почему freeze base:** Оригинальная F5-TTS уже отлично синтезирует речь.
Мы добавляем только модули управления голосом и эмоциями — это PEFT подход.

### Этап 2 (опциональный): Разморозка top-K блоков

```
Разморожено: top-4 DiT блока (блоки 18-21) + output norm/proj
LR:          1e-5 (низкий, чтобы не разрушить base)
Epochs:      1-2
```

---

## 5. Инференс: использование

### Python API

```python
from f5_tts.infer.enhanced_infer import (
    load_enhanced_model,
    load_embedding_extractors,
    load_vocoder,
    infer_enhanced,
)

# Загрузка моделей (один раз)
model = load_enhanced_model("path/to/enhanced_checkpoint.pt")
vocoder = load_vocoder()
speaker_enc, emotion_enc = load_embedding_extractors()

# Генерация (многократно)
wave, sr = infer_enhanced(
    ref_audio="reference_english.wav",        # английский референс
    ref_text="Hello, how are you doing?",     # текст референса
    gen_text="Привет, как у тебя дела?",      # русский текст для генерации
    model=model,
    vocoder=vocoder,
    speaker_encoder=speaker_enc,
    emotion_encoder=emotion_enc,
    emotion_cfg_strength=1.5,                 # усиленная эмоция
)
```

### Контроль эмоций через emotion_cfg_strength

| Значение | Эффект |
|----------|--------|
| 0.0 | Нейтральная речь, только голос клонирован |
| 0.5 | Лёгкий намёк на эмоцию из референса |
| 1.0 | Естественная передача эмоции (рекомендуется) |
| 1.5 | Усиленная эмоция |
| 2.0+ | Преувеличенная эмоция (может снизить качество) |

---

## 6. Структура файлов

```
F5-TTS-Enhanced/
├── src/f5_tts/
│   ├── model/
│   │   ├── speaker_encoder.py      # Speaker identity extraction
│   │   ├── emotion_encoder.py      # Emotion extraction (global + frame)
│   │   ├── conditioning.py         # Fusion & injection (AdaLN, CrossAttn, Add)
│   │   ├── enhanced_cfm.py         # Enhanced CFM with EG-CFG
│   │   ├── enhanced_dataset.py     # Dataset with embedding caching
│   │   ├── enhanced_trainer.py     # PEFT trainer (freeze base)
│   │   └── backbones/
│   │       └── enhanced_dit.py     # Modified DiT backbone
│   ├── infer/
│   │   └── enhanced_infer.py       # Inference pipeline
│   └── configs/
│       └── F5TTS_Enhanced.yaml     # Training config
└── ARCHITECTURE.md                 # This document
```

---

## 7. Вдохновение из архитектур-референсов

| Архитектура | Что взято |
|-------------|-----------|
| **PEFT-TTS** | Стратегия finetune: freeze base, train adapters |
| **F5-TTS-Emotional-CFG** | Emotion-Guided CFG для раздельного управления |
| **ece-tts** | Emotion Condition Embedding (embedding addition) |
| **Time-Varying Emotion Control** | Frame-level emotion features + temporal smoothing |
| **EmoSteer-TTS** | AdaLN modulation с emotion vectors |
| **TTS-CtrlNet** | Cross-attention injection, frozen external encoders |

---

## 8. Зависимости

### Обязательные
- PyTorch >= 2.0
- transformers >= 4.35 (для WavLM, emotion models)
- torchaudio
- torchdiffeq
- vocos
- x-transformers

### Для speaker encoder
- `wavlm_sv`: transformers (microsoft/wavlm-base-plus-sv)
- `ecapa_tdnn`: speechbrain
- `resemblyzer`: resemblyzer

### Для emotion encoder
- `emotion2vec_base`: funasr (iic/emotion2vec_base)
- `wav2vec2_ser`: transformers
- `hubert_ser`: transformers

---

## 9. Количество параметров

| Компонент | Параметры | Trainable |
|-----------|-----------|-----------|
| Base F5-TTS DiT (depth=22) | ~335M | ❄️ Frozen |
| ConditioningAggregator | ~18M | ✅ |
| ├── Fusion MLP | ~2M | ✅ |
| ├── AdaLN projections (22 блоков) | ~6.5M | ✅ |
| ├── CrossAttention (6 слоёв) | ~6M | ✅ |
| └── InputAdd projection | ~2M | ✅ |
| Speaker projection head | ~0.5M | ✅ |
| Emotion projection heads | ~1.5M | ✅ |
| **Итого trainable** | **~20M** | **~5.6% от всей модели** |
