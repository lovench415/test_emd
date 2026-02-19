# Подготовка данных для F5-TTS Enhanced

## Общая схема пайплайна

```
┌──────────────────────────────────────────────────────────────────────┐
│                      ПОДГОТОВКА ДАННЫХ                               │
│                                                                      │
│  Этап 0: Сбор сырых данных                                          │
│  ─────────────────────                                               │
│  Аудиофайлы (wav/flac/mp3) + транскрипции (text)                    │
│  Любые языки: RU, EN, ZH, DE, FR, ...                               │
│                    ↓                                                 │
│  Этап 1: Валидация и подготовка датасета                             │
│  ──────────────────────────────────────                              │
│  Фильтрация → нормализация → metadata.json + raw.arrow              │
│                    ↓                                                 │
│  Этап 2: Извлечение эмбеддингов (OFFLINE, однократно)               │
│  ──────────────────────────────────────────────────                  │
│  Audio → WavLM-SV     → speaker_raw   (512-d)   ┐                  │
│  Audio → emotion2vec  → emotion_global (768-d)   ├→ {i}.pt файлы   │
│                       → emotion_frame  (T×768-d) ┘                  │
│                    ↓                                                 │
│  Этап 3: Верификация                                                │
│  ───────────────────                                                │
│  Проверка целостности: аудио, тексты, эмбеддинги                    │
│                    ↓                                                 │
│  ГОТОВО К ОБУЧЕНИЮ                                                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Этап 0: Требования к сырым данным

### Формат аудио

| Параметр | Требование | Примечание |
|----------|-----------|------------|
| Формат | WAV, FLAC, MP3, OGG | WAV предпочтителен |
| Sample rate | Любой ≥ 16kHz | Будет ресемплирован в 24kHz |
| Каналы | Моно или стерео | Стерео → усреднение в моно |
| Длительность | 0.3 — 30 секунд | Оптимально 3-15 секунд |
| Качество | Чистая запись, минимум шума | Не критично для speaker/emotion encoders |
| Битрейт | ≥ 128 kbps | Для MP3/OGG |

### Формат метаданных

**Вариант A: CSV/LST файл (pipe-separated)**
```
# audio_path|text|language
audio/ru/001.wav|Привет, как у тебя дела?|ru
audio/en/002.wav|Hello, how are you doing?|en
audio/zh/003.wav|你好，你怎么样？|zh
audio/de/004.wav|Hallo, wie geht es dir?|de
```

**Вариант B: JSON файл**
```json
[
  {
    "audio_path": "audio/ru/001.wav",
    "text": "Привет, как у тебя дела?",
    "language": "ru"
  },
  {
    "audio_path": "audio/en/002.wav",
    "text": "Hello, how are you doing?",
    "language": "en"
  }
]
```

### Рекомендации по данным

**Для кросс-лингвального клонирования (главный сценарий):**

Модель учится переносить голос и эмоции из референса на ЛЮБОМ языке
в генерацию на русском. Для этого нужны данные с разнообразными:

1. **Языками** — чем больше языков в обучающей выборке, тем лучше
   кросс-лингвальный перенос. Минимум: русский + английский.
   Рекомендация: 4-6 языков.

2. **Говорящими** — разнообразие тембров, возрастов, полов.
   Минимум: 50-100 спикеров. Рекомендация: 500+.

3. **Эмоциями** — спонтанная речь содержит естественные эмоции.
   Если доступно, добавить данные актёрской озвучки с яркими эмоциями.

**Источники данных (примеры):**

| Датасет | Языки | Часы | Эмоции |
|---------|-------|------|--------|
| Emilia (основа F5-TTS) | ZH, EN | 101K | Нейтральные |
| Common Voice | 100+ языков | Варьируется | Нейтральные |
| RAVDESS | EN | 24 мин | 8 эмоций |
| EmoV-DB | EN, FR | 7 часов | 5 эмоций |
| IEMOCAP | EN | 12 часов | 9 эмоций |
| Ruslan | RU | 31 час | Нейтральный |
| NATASHA | RU | ~5 часов | 6 эмоций |
| M-AILABS | RU, EN, DE, FR | 1000+ часов | Книги |

**Приоритеты для сбора данных:**
1. Качество > количество (лучше 10 часов чистых данных, чем 100 часов шумных)
2. Разнообразие спикеров > количество данных на спикера
3. Эмоциональные данные — даже 1-2 часа сильно помогают

---

## Этап 1: Валидация и подготовка

### Запуск

```bash
python -m f5_tts.scripts.prepare_data \
    --stage prepare \
    --audio_dir /path/to/audio/root \
    --metadata /path/to/metadata.csv \
    --output_dir /data/f5tts_prepared \
    --min_duration 0.3 \
    --max_duration 30.0
```

### Что происходит

```
1. Чтение метаданных (CSV/JSON)
          ↓
2. Для каждого аудиофайла:
   ├── Проверка существования файла
   ├── Загрузка и проверка на NaN/Inf
   ├── Проверка длительности [0.3s, 30s]
   ├── Проверка на тишину (RMS > 1e-5)
   └── Запись валидных записей
          ↓
3. Создание HuggingFace Dataset (raw.arrow)
          ↓
4. Создание duration.json для DynamicBatchSampler
          ↓
5. Построение словаря символов (vocab.txt)
   ├── Все символы из текстов
   ├── Русский алфавит (А-Яа-яЁё)
   └── Стандартная пунктуация
          ↓
6. Сохранение статистики (metadata.json)
```

### Выходная структура

```
/data/f5tts_prepared/
├── raw/                    # HuggingFace Dataset на диске
│   ├── data-00000-of-00001.arrow
│   ├── dataset_info.json
│   └── state.json
├── duration.json           # {"duration": [2.31, 4.56, ...]}
├── vocab.txt               # По символу на строку
└── metadata.json           # Статистика датасета
```

### Формат записи в raw.arrow

Каждая запись содержит:
```python
{
    "audio_path": "/abs/path/to/audio.wav",   # абсолютный путь
    "text": "Привет, как дела?",               # транскрипция
    "duration": 3.45,                           # длительность в секундах
    "language": "ru",                           # язык (опционально)
}
```

---

## Этап 2: Извлечение эмбеддингов

### Зачем offline-извлечение

**Без кэша (online extraction):**
```
Каждая эпоха: Audio → WavLM-SV (300ms/sample) → speaker_emb
              Audio → emotion2vec (500ms/sample) → emotion_emb
              
Для 100K сэмплов × 10 эпох = 1M вызовов encoder'ов
≈ 200 часов чистого compute на GPU
```

**С кэшем (offline extraction):**
```
Однократно:  Audio → WavLM-SV → speaker_raw  → сохранить .pt
             Audio → emotion2vec → emotion_raw → сохранить .pt

Каждая эпоха: Загрузка .pt (< 1ms/sample) → projection MLP (trainable)

Для 100K сэмплов: ~13 часов однократно, потом бесплатно
```

**Экономия: 15-20x ускорение обучения.**

### Запуск

```bash
python -m f5_tts.scripts.prepare_data \
    --stage embeddings \
    --dataset_dir /data/f5tts_prepared \
    --embedding_dir /data/f5tts_embeddings \
    --speaker_backend wavlm_sv \
    --emotion_backend emotion2vec_base \
    --device cuda \
    --resume  # пропускает уже вычисленные
```

### Что именно извлекается

#### Speaker Embedding (WavLM-SV)

```
Audio (24kHz)
    ↓ Resample to 16kHz
    ↓
WavLM-base-plus-sv (frozen, 94.7M params)
    ↓ x-vector extraction
    ↓
speaker_raw: tensor shape (512,)
```

**Что кодирует speaker_raw:**
- Тембр голоса (формантная структура)
- Высота голоса (средний F0)
- Качество голоса (breathiness, creakiness)
- Характерные особенности (lisp, accent traits)
- **НЕ кодирует:** содержание речи, язык, эмоцию

**Почему language-agnostic:**
WavLM-SV обучена на speaker verification задаче (VoxCeleb),
где один и тот же говорящий может говорить на разных языках.
Модель учится игнорировать лингвистическое содержание и
фокусироваться на акустических характеристиках голоса.

#### Emotion Embedding (emotion2vec)

```
Audio (24kHz)
    ↓ Resample to 16kHz
    ↓
emotion2vec_base (frozen, ~90M params)
    ↓
    ├── Frame-level features: (T_frames, 768)
    │   T_frames ≈ duration_sec × 50 (50 fps)
    │   Кодирует: моментальное эмоциональное состояние
    │   Пример: 5-секундное аудио → (250, 768)
    │
    └── Global feature: mean(frame_features) → (768,)
        Кодирует: усреднённая эмоция всего высказывания
```

**Что кодирует emotion_raw:**
- Valence (позитивность/негативность)
- Arousal (активность/спокойствие)
- Dominance (уверенность/подчинённость)
- Конкретная эмоция (радость, грусть, гнев, страх, удивление, отвращение)
- Интонационные паттерны (rising/falling pitch, emphasis)
- **НЕ кодирует:** идентичность говорящего, содержание речи

**Почему два уровня:**
- **Global** — для общего эмоционального окраса генерации
  (инъецируется через AdaLN + InputAdd)
- **Frame-level** — для временной динамики эмоции
  (инъецируется через Cross-Attention)
  
  Пример: фраза начинается спокойно, переходит в волнение,
  заканчивается решительно — frame-level фичи передают эту
  динамику через cross-attention в DiT блоках.

### Формат кэш-файлов

```
/data/f5tts_embeddings/
├── 0.pt                    # Embedding для sample 0
├── 1.pt                    # Embedding для sample 1
├── ...
├── 99999.pt
└── embedding_meta.json     # Метаданные извлечения
```

Каждый `{i}.pt` файл:
```python
{
    "speaker_raw": torch.Tensor(512),        # float32, ~2 KB
    "emotion_global_raw": torch.Tensor(768), # float32, ~3 KB  
    "emotion_frame_raw": torch.Tensor(T, 768), # float32, ~T×3 KB
}
# Для 5-секундного аудио: T ≈ 250, размер файла ~0.8 MB
# Для 100K сэмплов: ~80 GB (в основном из-за frame features)
```

### Время извлечения (ориентиры)

| Компонент | GPU (A100) | GPU (RTX 3090) | CPU |
|-----------|-----------|----------------|-----|
| WavLM-SV, 1 сэмпл | ~30ms | ~60ms | ~500ms |
| emotion2vec, 1 сэмпл | ~50ms | ~100ms | ~800ms |
| 100K сэмплов | ~2.2 часа | ~4.5 часа | ~36 часов |
| 1M сэмплов | ~22 часа | ~45 часов | ~360 часов |

**Совет:** Используйте `--resume` для продолжения после прерывания.

---

## Этап 3: Верификация

```bash
python -m f5_tts.scripts.prepare_data \
    --stage verify \
    --dataset_dir /data/f5tts_prepared \
    --embedding_dir /data/f5tts_embeddings
```

Проверяет:
- Все аудиофайлы существуют и загружаются
- Для каждого сэмпла есть файл эмбеддинга
- Размерности эмбеддингов консистентны
- Нет NaN/Inf в эмбеддингах
- Длительности совпадают

---

## Важные нюансы

### Как embedding projection работает при обучении

```
┌─────────────────────────────────────────────────┐
│  Offline (Этап 2):                              │
│  Audio → WavLM-SV(frozen) → speaker_raw (512)   │
│          saved to .pt file                       │
│                                                  │
│  Training (каждый batch):                        │
│  Load speaker_raw from .pt                       │
│       ↓                                          │
│  SpeakerEncoder.proj (TRAINABLE)                 │
│       Linear(512→512) → SiLU → Linear(512→512)  │
│       ↓                                          │
│  L2-Normalize                                    │
│       ↓                                          │
│  speaker_emb (512) → инъекция в DiT              │
└─────────────────────────────────────────────────┘
```

**Ключевой момент:** Мы кэшируем RAW эмбеддинги (до projection),
потому что projection MLP — это обучаемый компонент, который
меняется каждую итерацию. Сырые фичи фиксированы (frozen encoder),
поэтому их можно безопасно вычислить один раз.

### Обработка отсутствующих эмбеддингов

Если для какого-то сэмпла эмбеддинг не удалось извлечь (ошибка audio):
- Сохраняется нулевой вектор `torch.zeros(dim)`
- Модель обрабатывает нулевые эмбеддинги как "отсутствие условия"
  (аналогично dropout при обучении)
- Это безопасно и не ломает обучение

### Языковая разметка

Поле `language` в метаданных **не используется** моделью напрямую.
Оно нужно:
1. Для статистики и анализа датасета
2. Для обеспечения баланса языков при семплировании батчей (опционально)
3. Для выбора правильного tokenizer при подготовке текста

**Для текстов на русском** используется tokenizerpin = "pinyin" / "char" / "custom".
При обучении на мультиязычных данных рекомендуется "custom" tokenizer
с объединённым словарём.

### Аугментация (опционально)

При обучении можно применять аугментацию к аудио,
которое отправляется в speaker/emotion encoder:

```python
# Применяется ТОЛЬКО к копии для encoder'ов
# НЕ к mel-спектрограмме для генерации
augmented_audio = augment_reference_audio(
    audio,
    noise_level=0.005,        # лёгкий шум
    speed_range=(0.95, 1.05), # ±5% скорость
)
speaker_emb = speaker_encoder(augmented_audio)
```

Это делает модель более робастной к:
- Разным условиям записи (микрофоны, помещения)
- Разной громкости
- Небольшим вариациям скорости речи

---

## Полный пример: от нуля до обучения

```bash
# 1. Подготовить метаданные (вручную или скриптом)
cat > metadata.csv << 'EOF'
audio/ru_happy_001.wav|Сегодня чудесный день!|ru
audio/ru_sad_002.wav|Мне очень грустно.|ru
audio/en_angry_003.wav|This is absolutely unacceptable!|en
audio/zh_neutral_004.wav|今天天气很好。|zh
EOF

# 2. Валидация и подготовка
python -m f5_tts.scripts.prepare_data \
    --stage prepare \
    --audio_dir /data/raw_audio \
    --metadata metadata.csv \
    --output_dir /data/f5tts_dataset

# 3. Извлечение эмбеддингов (~2-5 часов для 100K сэмплов)
python -m f5_tts.scripts.prepare_data \
    --stage embeddings \
    --dataset_dir /data/f5tts_dataset \
    --embedding_dir /data/f5tts_embeddings \
    --device cuda

# 4. Верификация
python -m f5_tts.scripts.prepare_data \
    --stage verify \
    --dataset_dir /data/f5tts_dataset \
    --embedding_dir /data/f5tts_embeddings

# 5. Запуск обучения
python train_enhanced.py \
    --config configs/F5TTS_Enhanced.yaml \
    --dataset_dir /data/f5tts_dataset \
    --embedding_dir /data/f5tts_embeddings \
    --pretrain_ckpt /path/to/original/f5tts_base.pt
```

---

## Требования к дисковому пространству

| Компонент | Размер (100K сэмплов) | Размер (1M сэмплов) |
|-----------|----------------------|---------------------|
| Аудио (WAV, 24kHz) | ~150 GB | ~1.5 TB |
| raw.arrow (метаданные) | ~50 MB | ~500 MB |
| Speaker embeddings | ~200 MB | ~2 GB |
| Emotion global embeddings | ~300 MB | ~3 GB |
| Emotion frame embeddings | ~80 GB | ~800 GB |
| **Итого (без аудио)** | **~80 GB** | **~805 GB** |

**Совет:** Если дисковое пространство ограничено, можно отключить
frame-level emotion embeddings (`emotion_encoder.frame_level=false`).
Это сократит объём в ~100x, но потеряется временной контроль эмоций.
Global emotion embedding занимает мало места и должен быть всегда.
