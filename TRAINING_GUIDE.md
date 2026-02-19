# Запуск обучения F5-TTS Enhanced

## Быстрый старт (5 минут до запуска)

```bash
# 0. Установка зависимостей
pip install f5-tts torch torchaudio transformers accelerate ema-pytorch \
    cached-path datasets vocos torchdiffeq x-transformers \
    funasr speechbrain wandb --break-system-packages

# 1. Подготовка данных (если ещё не сделано)
python -m f5_tts.scripts.prepare_data \
    --stage prepare \
    --audio_dir /data/audio \
    --metadata /data/metadata.csv \
    --output_dir /data/f5tts_dataset

# 2. Извлечение эмбеддингов (однократно, ~2-5 часов на 100K сэмплов)
python -m f5_tts.scripts.prepare_data \
    --stage embeddings \
    --dataset_dir /data/f5tts_dataset \
    --embedding_dir /data/f5tts_embeddings \
    --device cuda

# 3. Запуск обучения (1 GPU)
python train_enhanced.py \
    --dataset_dir /data/f5tts_dataset \
    --embedding_dir /data/f5tts_embeddings
```

Всё. Скрипт сам скачает претрейн F5-TTS, заморозит базовые веса, и начнёт обучать conditioning модули.

---

## Пошаговая инструкция

### Шаг 0: Установка зависимостей

```bash
# Создать виртуальное окружение
conda create -n f5tts_enhanced python=3.10
conda activate f5tts_enhanced

# PyTorch (подбирайте под вашу CUDA)
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# F5-TTS base
pip install f5-tts

# Зависимости Enhanced
pip install transformers>=4.35       # для WavLM, HuBERT
pip install accelerate>=0.25         # для multi-GPU
pip install ema-pytorch              # EMA
pip install cached-path              # скачивание чекпоинтов
pip install datasets vocos torchdiffeq x-transformers wandb
pip install funasr                   # для emotion2vec
pip install speechbrain              # для ecapa_tdnn (опционально)
```

**Требования к железу:**

| Конфигурация | GPU VRAM | Batch size | Скорость |
|-------------|----------|------------|----------|
| Минимальная | 16 GB (RTX 4080) | 6400 frames | ~0.5 updates/sec |
| Рекомендуемая | 24 GB (RTX 3090/4090) | 19200 frames | ~1.5 updates/sec |
| Оптимальная | 40+ GB (A100) | 38400 frames | ~3 updates/sec |
| Multi-GPU | 4× 24 GB | 76800 frames | ~5 updates/sec |

### Шаг 1: Скачать претрейн F5-TTS

Скрипт скачает автоматически при первом запуске. Вручную:

```bash
huggingface-cli download SWivid/F5-TTS F5TTS_Base/model_1200000.pt \
    --local-dir ./ckpts
```

### Шаг 2: Подготовить данные

Если не сделано — см. [DATA_PREPARATION.md](DATA_PREPARATION.md).

```bash
# Проверка готовности:
ls /data/f5tts_dataset/raw/          # HuggingFace dataset
ls /data/f5tts_dataset/duration.json  # длительности
ls /data/f5tts_embeddings/0.pt        # хотя бы один эмбеддинг
```

### Шаг 3: Запуск обучения

#### Один GPU (простейший)

```bash
python train_enhanced.py \
    --dataset_dir /data/f5tts_dataset \
    --embedding_dir /data/f5tts_embeddings \
    --checkpoint_dir ./ckpts/enhanced_v1 \
    --epochs 5 \
    --learning_rate 3e-4 \
    --batch_size_per_gpu 19200 \
    --logger wandb
```

#### Multi-GPU (через Accelerate)

```bash
accelerate config  # настроить multi-GPU
accelerate launch --multi_gpu --num_processes 4 \
    train_enhanced.py \
    --dataset_dir /data/f5tts_dataset \
    --embedding_dir /data/f5tts_embeddings \
    --checkpoint_dir ./ckpts/enhanced_v1 \
    --epochs 5
```

#### Минимальный ресурс (16 GB VRAM)

```bash
python train_enhanced.py \
    --dataset_dir /data/f5tts_dataset \
    --embedding_dir /data/f5tts_embeddings \
    --batch_size_per_gpu 6400 \
    --max_samples 16 \
    --grad_accumulation_steps 4 \
    --no_cross_attn \
    --epochs 10
```

#### Двухстадийное обучение (лучшее качество)

```bash
# Стадия 1: только conditioning модули (быстро)
python train_enhanced.py \
    --dataset_dir /data/f5tts_dataset \
    --embedding_dir /data/f5tts_embeddings \
    --checkpoint_dir ./ckpts/stage1 \
    --epochs 3 --learning_rate 3e-4 \
    --freeze_base --unfreeze_top_k 0

# Стадия 2: + top-4 DiT блока (fine-grained)
python train_enhanced.py \
    --dataset_dir /data/f5tts_dataset \
    --embedding_dir /data/f5tts_embeddings \
    --pretrain_ckpt ./ckpts/stage1/model_last.pt \
    --checkpoint_dir ./ckpts/stage2 \
    --epochs 2 --learning_rate 1e-5 \
    --freeze_base --unfreeze_top_k 4
```

---

## Параметры — что и зачем

### Критичные

| Параметр | Значение | Зачем |
|----------|---------|-------|
| `--learning_rate 3e-4` | Высокий | Conditioning модули zero-init, нужен высокий LR. Базовые веса заморожены → безопасно |
| `--epochs 5` | Мало | ~20M trainable params сходятся быстро |
| `--freeze_base` | Заморозка | Базовая F5-TTS уже хорошо синтезирует. Мы только добавляем управление |
| `--batch_size_per_gpu 19200` | В фреймах | ~6-10 аудио по 5-10 сек. Подбирать под VRAM |

### Экономия памяти

| Параметр | Экономия VRAM | Влияние на качество |
|----------|--------------|---------------------|
| `--no_cross_attn` | ~2-3 GB | Теряется временной контроль эмоций |
| `--max_samples 16` | ~1-2 GB | Медленнее обучение |
| `--grad_accumulation_steps 4` | Позволяет уменьшить batch | Эффективный batch не меняется |

---

## Что происходит внутри

### Training loop (каждый batch)

```
1. Загрузить batch из кэша:
   mel_spec + text + speaker_raw + emotion_raw

2. Проецировать через trainable heads:
   speaker_raw → proj → speaker_emb
   emotion_raw → proj + smooth + interpolate → emotion_global + emotion_frame

3. Multi-condition dropout (для CFG):
   20%: drop ALL (unconditional)
   30%: drop audio condition
   10%: drop speaker
   10%: drop emotion

4. Flow matching:
   noise + mel → interpolate → EnhancedDiT(с conditioning) → predicted flow
   Loss = MSE(predicted_flow, true_flow) на masked span

5. Backward → clip grad → AdamW step → EMA update
```

### Как conditioning инъецируется в каждом DiT block

```
Block i:
  AdaLN(x, timestep) → gates
    + ConditioningAdaLN[i](fused) → δgates    [TRAINABLE]
    gates = gates + δgates

  Self-Attention(x) → attn_out
    x = x + gate × attn_out

  [если i ∈ {0,4,8,12,16,20}]:
    CrossAttention(x → emotion_frame)           [TRAINABLE]
    x = x + tanh(gate) × cross_attn_out

  FFN(x) → x
```

---

## Мониторинг

### Что отслеживать

```
Нормальное поведение:
  Epoch 1: loss 0.15 → 0.08   (conditioning "оживает")
  Epoch 2: loss 0.08 → 0.05   (стабилизация)
  Epoch 3: loss 0.05 → 0.04   (плато)

Проблемы:
  loss > 0.5 после 1000 шагов → увеличить LR
  loss прыгает ±0.1           → уменьшить LR
  loss NaN                     → проверить данные
  loss не падает               → проверить что не всё заморожено
```

### Тест качества по ходу обучения

```python
from f5_tts.infer.enhanced_infer import *

model = load_enhanced_model("ckpts/enhanced_v1/model_last.pt")
vocoder = load_vocoder()
spk, emo = load_embedding_extractors()

wave, sr = infer_enhanced(
    ref_audio="reference.wav",
    ref_text="", gen_text="Тестовая генерация.",
    model=model, vocoder=vocoder,
    speaker_encoder=spk, emotion_encoder=emo,
    emotion_cfg_strength=1.0,
)
import soundfile as sf
sf.write("test.wav", wave, sr)
```

---

## Resume после прерывания

Запустите ту же команду — скрипт найдёт `model_last.pt` автоматически:

```bash
python train_enhanced.py \
    --dataset_dir /data/f5tts_dataset \
    --embedding_dir /data/f5tts_embeddings \
    --checkpoint_dir ./ckpts/enhanced_v1 \
    --epochs 5
# → "Resuming from ckpts/enhanced_v1/model_last.pt at update 12500"
```

---

## FAQ

**Q: Сколько данных нужно?**
Минимум 1 час. Рекомендуется 10-50 часов. Разнообразие спикеров важнее количества.

**Q: Можно без эмоций, только голос?**
Да. Не создавайте emotion embeddings — модель будет клонировать только голос.

**Q: Время обучения?**
1× RTX 3090 + 50K сэмплов: ~2-4 часа/эпоха × 5 = 10-20 часов.
4× A100 + 100K сэмплов: ~1-2 часа всего.

**Q: emotion_cfg_strength — когда настраивать?**
Только при инференсе. Обучение автоматически поддерживает любые значения через multi-condition dropout.
