"""
F5-TTS Enhanced — Custom Model Training Scripts
=================================================

Ready-to-run scripts for common fine-tuning scenarios:

1. SingleSpeakerFinetune  — клон одного голоса (5-30 минут аудио)
2. EmotionalVoiceFinetune — голос + эмоции (1-5 часов аудио)  
3. MultiSpeakerFinetune   — универсальная модель (10+ часов, 50+ спикеров)
4. QuickTest              — проверка результата

Usage:
    python custom_train.py single_speaker \
        --audio_dir /data/my_voice \
        --output_dir ./my_model

    python custom_train.py emotional \
        --audio_dir /data/emotional_speech \
        --metadata /data/metadata.csv \
        --output_dir ./emo_model

    python custom_train.py multi_speaker \
        --dataset_dir /data/prepared \
        --embedding_dir /data/embeddings \
        --output_dir ./universal_model

    python custom_train.py test \
        --checkpoint ./my_model/model_last.pt \
        --ref_audio ./reference.wav \
        --text "Привет! Это тест моей модели."
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torchaudio
import numpy as np


# =====================================================================
# Scenario 1: Single Speaker Voice Clone
# =====================================================================

def single_speaker_finetune(args):
    """
    Обучение модели на голос ОДНОГО человека.
    
    Входные данные:
        - Папка с аудиофайлами (WAV/FLAC/MP3), 5-30 минут
        - Опционально: текстовые транскрипции
    
    Результат:
        - Модель, клонирующая этот конкретный голос
        - При инференсе: ref_audio всё ещё нужен, но модель
          "настроена" на этот тембр → лучшее качество
    
    Типичный use case:
        - Виртуальный ассистент с голосом заказчика
        - Озвучка книг голосом автора
        - Персонализированный TTS
    """
    print("=" * 60)
    print("  Сценарий: Клонирование одного голоса")
    print("=" * 60)
    
    audio_dir = args.audio_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # ── Step 1: Найти все аудиофайлы ──
    audio_files = []
    for ext in ["*.wav", "*.flac", "*.mp3", "*.ogg"]:
        audio_files.extend(Path(audio_dir).glob(ext))
        audio_files.extend(Path(audio_dir).glob(f"**/{ext}"))
    audio_files = sorted(set(audio_files))
    
    if not audio_files:
        print(f"ERROR: Нет аудиофайлов в {audio_dir}")
        return
    
    print(f"\nНайдено {len(audio_files)} аудиофайлов")
    
    # ── Step 2: Подсчитать общую длительность ──
    total_duration = 0
    valid_files = []
    for af in audio_files:
        try:
            info = torchaudio.info(str(af))
            dur = info.num_frames / info.sample_rate
            if 0.5 <= dur <= 30.0:
                valid_files.append({"path": str(af), "duration": dur})
                total_duration += dur
        except:
            pass
    
    print(f"Валидных файлов: {len(valid_files)}")
    print(f"Общая длительность: {total_duration/60:.1f} минут")
    
    if total_duration < 30:
        print("WARNING: Менее 30 секунд аудио. Качество будет низким.")
        print("Рекомендуется минимум 5 минут.")
    
    # ── Step 3: Создать метаданные ──
    # Если есть транскрипции — использовать
    # Если нет — оставить пустыми (F5-TTS может работать без текста ref)
    metadata_path = os.path.join(output_dir, "metadata.csv")
    transcript_dir = args.transcript_dir or audio_dir
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        for item in valid_files:
            audio_path = item["path"]
            # Поискать транскрипцию
            txt_path = Path(audio_path).with_suffix(".txt")
            lab_path = Path(audio_path).with_suffix(".lab")
            
            text = ""
            if txt_path.exists():
                text = txt_path.read_text(encoding="utf-8").strip()
            elif lab_path.exists():
                text = lab_path.read_text(encoding="utf-8").strip()
            
            f.write(f"{audio_path}|{text}|ru\n")
    
    print(f"Метаданные: {metadata_path}")
    
    # ── Step 4: Подготовить dataset ──
    dataset_dir = os.path.join(output_dir, "dataset")
    print("\n[1/4] Подготовка dataset...")
    _run_cmd(f"""
python -m f5_tts.scripts.prepare_data \
    --stage prepare \
    --audio_dir "{audio_dir}" \
    --metadata "{metadata_path}" \
    --output_dir "{dataset_dir}"
""")
    
    # ── Step 5: Извлечь эмбеддинги ──
    embedding_dir = os.path.join(output_dir, "embeddings")
    print("\n[2/4] Извлечение эмбеддингов...")
    _run_cmd(f"""
python -m f5_tts.scripts.prepare_data \
    --stage embeddings \
    --dataset_dir "{dataset_dir}" \
    --embedding_dir "{embedding_dir}" \
    --device {"cuda" if torch.cuda.is_available() else "cpu"}
""")
    
    # ── Step 6: Запустить обучение ──
    # Для одного спикера: мало данных → больше эпох, маленький batch
    epochs = _auto_epochs(total_duration)
    lr = 3e-4 if total_duration > 300 else 5e-4  # выше LR для мало данных
    
    print(f"\n[3/4] Обучение ({epochs} эпох, LR={lr})...")
    
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    _run_cmd(f"""
python train_enhanced.py \
    --dataset_dir "{dataset_dir}" \
    --embedding_dir "{embedding_dir}" \
    --checkpoint_dir "{ckpt_dir}" \
    --epochs {epochs} \
    --learning_rate {lr} \
    --batch_size_per_gpu {_auto_batch_size()} \
    --max_samples 16 \
    --num_warmup_updates {min(500, len(valid_files) * 2)} \
    --save_per_updates 2000 \
    --last_per_updates 500 \
    --freeze_base \
    --unfreeze_top_k 0
""")
    
    # ── Step 7: Выбрать лучший референс ──
    print("\n[4/4] Выбор лучшего референса...")
    best_ref = _select_best_reference(valid_files)
    shutil.copy2(best_ref, os.path.join(output_dir, "best_reference.wav"))
    
    # ── Готово ──
    print(f"""
{'=' * 60}
  ГОТОВО!
{'=' * 60}

Checkpoint:   {ckpt_dir}/model_last.pt
Референс:     {output_dir}/best_reference.wav

Инференс:
    python custom_train.py test \\
        --checkpoint {ckpt_dir}/model_last.pt \\
        --ref_audio {output_dir}/best_reference.wav \\
        --text "Ваш текст для озвучки."
""")


# =====================================================================
# Scenario 2: Emotional Voice Finetune
# =====================================================================

def emotional_voice_finetune(args):
    """
    Обучение модели с переносом эмоций.
    
    Входные данные:
        - Аудио с разными эмоциями (1-5 часов)
        - Метаданные: audio_path|text|language
        - Желательно: несколько спикеров с разными эмоциями
    
    Результат:
        - Модель, переносящая эмоцию из референса в генерацию
        - emotion_cfg_strength: 0→нейтрально, 1→как в референсе, 2+→усиленно
    
    Источники эмоциональных данных:
        - RAVDESS (EN, 8 эмоций)
        - EmoV-DB (EN/FR, 5 эмоций)
        - NATASHA (RU, 6 эмоций)
        - Собственные записи актёров
    """
    print("=" * 60)
    print("  Сценарий: Голос + эмоции")
    print("=" * 60)
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # ── Проверить метаданные ──
    if not args.metadata:
        print("ERROR: --metadata обязателен для эмоционального обучения")
        print("Формат: audio_path|text|language")
        return
    
    # ── Подготовка данных ──
    dataset_dir = os.path.join(output_dir, "dataset")
    embedding_dir = os.path.join(output_dir, "embeddings")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    
    print("\n[1/3] Подготовка dataset + эмбеддинги...")
    _run_cmd(f"""
python -m f5_tts.scripts.prepare_data \
    --stage all \
    --audio_dir "{args.audio_dir}" \
    --metadata "{args.metadata}" \
    --output_dir "{dataset_dir}" \
    --embedding_dir "{embedding_dir}" \
    --emotion_backend emotion2vec_base \
    --device {"cuda" if torch.cuda.is_available() else "cpu"}
""")
    
    # Подсчитать статистику
    meta_path = os.path.join(dataset_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            stats = json.load(f)
        total_hours = stats.get("total_duration_hours", 0)
        total_samples = stats.get("total_samples", 0)
    else:
        total_hours = 1
        total_samples = 100
    
    # ── Обучение: Stage 1 ──
    epochs_s1 = max(3, min(10, int(20 / max(total_hours, 0.1))))
    
    print(f"\n[2/3] Stage 1: conditioning modules ({epochs_s1} эпох)...")
    _run_cmd(f"""
python train_enhanced.py \
    --dataset_dir "{dataset_dir}" \
    --embedding_dir "{embedding_dir}" \
    --checkpoint_dir "{ckpt_dir}" \
    --epochs {epochs_s1} \
    --learning_rate 3e-4 \
    --batch_size_per_gpu {_auto_batch_size()} \
    --num_warmup_updates 2000 \
    --save_per_updates 5000 \
    --freeze_base --unfreeze_top_k 0 \
    --logger {args.logger or 'None'}
""")
    
    # ── Обучение: Stage 2 (optional) ──
    if total_hours >= 2:
        print(f"\n[3/3] Stage 2: + top-4 DiT blocks (2 эпохи)...")
        _run_cmd(f"""
python train_enhanced.py \
    --dataset_dir "{dataset_dir}" \
    --embedding_dir "{embedding_dir}" \
    --pretrain_ckpt "{ckpt_dir}/model_last.pt" \
    --checkpoint_dir "{ckpt_dir}_stage2" \
    --epochs 2 \
    --learning_rate 1e-5 \
    --batch_size_per_gpu {_auto_batch_size()} \
    --freeze_base --unfreeze_top_k 4
""")
        final_ckpt = f"{ckpt_dir}_stage2/model_last.pt"
    else:
        print("\n[3/3] Пропуск Stage 2 (< 2 часов данных)")
        final_ckpt = f"{ckpt_dir}/model_last.pt"
    
    print(f"""
{'=' * 60}
  ГОТОВО!
{'=' * 60}

Checkpoint: {final_ckpt}

Пример инференса с эмоциями:
    python custom_train.py test \\
        --checkpoint {final_ckpt} \\
        --ref_audio happy_reference.wav \\
        --text "Сегодня замечательный день!" \\
        --emotion_strength 1.5

Emotion strength:
    0.0  → нейтральная речь (эмоция из референса игнорируется)
    1.0  → эмоция как в референсе
    1.5  → усиленная эмоция
    2.0+ → преувеличенная эмоция (может быть нестабильно)
""")


# =====================================================================
# Scenario 3: Multi-Speaker Universal Model
# =====================================================================

def multi_speaker_finetune(args):
    """
    Обучение универсальной модели на множество спикеров.
    
    Входные данные:
        - Подготовленный dataset (10+ часов, 50+ спикеров)
        - Предвычисленные эмбеддинги
    
    Результат:
        - Модель, клонирующая ЛЮБОЙ голос из короткого референса
        - Перенос эмоций между языками
    
    Это основной сценарий для production-ready модели.
    """
    print("=" * 60)
    print("  Сценарий: Универсальная multi-speaker модель")
    print("=" * 60)
    
    dataset_dir = args.dataset_dir
    embedding_dir = args.embedding_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    
    # Проверить данные
    if not os.path.exists(os.path.join(dataset_dir, "duration.json")):
        print("ERROR: dataset_dir не содержит duration.json")
        print("Запустите prepare_data.py --stage prepare сначала")
        return
    
    with open(os.path.join(dataset_dir, "duration.json")) as f:
        durations = json.load(f)["duration"]
    total_hours = sum(durations) / 3600
    total_samples = len(durations)
    
    print(f"Dataset: {total_samples} сэмплов, {total_hours:.1f} часов")
    
    # ── Stage 1 ──
    epochs = max(3, min(10, int(50 / max(total_hours, 1))))
    
    print(f"\n[1/2] Stage 1: conditioning ({epochs} эпох)...")
    _run_cmd(f"""
python train_enhanced.py \
    --dataset_dir "{dataset_dir}" \
    --embedding_dir "{embedding_dir}" \
    --checkpoint_dir "{ckpt_dir}" \
    --epochs {epochs} \
    --learning_rate 3e-4 \
    --batch_size_per_gpu {_auto_batch_size()} \
    --max_samples 32 \
    --num_warmup_updates 2000 \
    --save_per_updates 10000 \
    --freeze_base --unfreeze_top_k 0 \
    --logger {args.logger or 'None'}
""")
    
    # ── Stage 2 ──
    print(f"\n[2/2] Stage 2: + top-4 blocks (2 эпохи)...")
    _run_cmd(f"""
python train_enhanced.py \
    --dataset_dir "{dataset_dir}" \
    --embedding_dir "{embedding_dir}" \
    --pretrain_ckpt "{ckpt_dir}/model_last.pt" \
    --checkpoint_dir "{ckpt_dir}_stage2" \
    --epochs 2 \
    --learning_rate 1e-5 \
    --batch_size_per_gpu {_auto_batch_size()} \
    --freeze_base --unfreeze_top_k 4 \
    --logger {args.logger or 'None'}
""")
    
    print(f"""
{'=' * 60}
  ГОТОВО!
{'=' * 60}

Checkpoint: {ckpt_dir}_stage2/model_last.pt

Эта модель может клонировать любой голос из 3-10 секунд референса.
""")


# =====================================================================
# Quick Test
# =====================================================================

def quick_test(args):
    """Быстрый тест обученной модели."""
    print("=" * 60)
    print("  Тестирование модели")
    print("=" * 60)
    
    checkpoint = args.checkpoint
    ref_audio = args.ref_audio
    text = args.text
    emotion_strength = args.emotion_strength
    output_path = args.output or "test_output.wav"
    
    if not os.path.exists(checkpoint):
        print(f"ERROR: Checkpoint не найден: {checkpoint}")
        return
    
    print(f"Checkpoint:  {checkpoint}")
    print(f"Reference:   {ref_audio}")
    print(f"Text:        {text}")
    print(f"Emotion:     {emotion_strength}")
    print(f"Output:      {output_path}")
    
    from f5_tts.infer.enhanced_infer import (
        load_enhanced_model,
        load_vocoder,
        load_embedding_extractors,
        infer_enhanced,
    )
    
    print("\nЗагрузка модели...")
    model = load_enhanced_model(checkpoint)
    vocoder = load_vocoder()
    spk_enc, emo_enc = load_embedding_extractors()
    
    print("Генерация...")
    wave, sr = infer_enhanced(
        ref_audio=ref_audio,
        ref_text="",
        gen_text=text,
        model=model,
        vocoder=vocoder,
        speaker_encoder=spk_enc,
        emotion_encoder=emo_enc,
        emotion_cfg_strength=emotion_strength,
    )
    
    import soundfile as sf
    sf.write(output_path, wave, sr)
    print(f"\nСохранено: {output_path}")
    print(f"Длительность: {len(wave)/sr:.1f} секунд")


# =====================================================================
# Batch Inference
# =====================================================================

def batch_inference(args):
    """
    Пакетная генерация: один референс → много текстов.
    
    Формат input.txt:
        Первая строка текста для озвучки.
        Вторая строка текста.
        И так далее.
    """
    print("=" * 60)
    print("  Пакетная генерация")
    print("=" * 60)
    
    from f5_tts.infer.enhanced_infer import (
        load_enhanced_model,
        load_vocoder,
        load_embedding_extractors,
        infer_enhanced,
    )
    import soundfile as sf
    
    # Загрузить тексты
    with open(args.input_texts, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"Текстов для генерации: {len(texts)}")
    
    # Загрузить модель (один раз)
    model = load_enhanced_model(args.checkpoint)
    vocoder = load_vocoder()
    spk_enc, emo_enc = load_embedding_extractors()
    
    # Извлечь эмбеддинги референса (один раз)
    print(f"Референс: {args.ref_audio}")
    
    output_dir = args.output_dir or "batch_output"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, text in enumerate(texts):
        print(f"\n[{i+1}/{len(texts)}] {text[:50]}...")
        
        wave, sr = infer_enhanced(
            ref_audio=args.ref_audio,
            ref_text="",
            gen_text=text,
            model=model,
            vocoder=vocoder,
            speaker_encoder=spk_enc,
            emotion_encoder=emo_enc,
            emotion_cfg_strength=args.emotion_strength,
        )
        
        out_path = os.path.join(output_dir, f"{i+1:04d}.wav")
        sf.write(out_path, wave, sr)
        print(f"  → {out_path} ({len(wave)/sr:.1f}s)")
    
    print(f"\nГотово! {len(texts)} файлов в {output_dir}/")


# =====================================================================
# Helper Functions
# =====================================================================

def _run_cmd(cmd: str):
    """Run a shell command, printing output."""
    cmd = cmd.strip()
    print(f"$ {cmd[:80]}...")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: Command returned code {result.returncode}")


def _auto_batch_size() -> int:
    """Auto-detect batch size based on available GPU memory."""
    if not torch.cuda.is_available():
        return 3200
    
    mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    if mem_gb >= 40:
        return 38400
    elif mem_gb >= 24:
        return 19200
    elif mem_gb >= 16:
        return 9600
    else:
        return 4800


def _auto_epochs(total_duration_sec: float) -> int:
    """Auto-detect number of epochs based on data size."""
    hours = total_duration_sec / 3600
    if hours < 0.1:    # < 6 min
        return 50
    elif hours < 0.5:  # < 30 min
        return 20
    elif hours < 2:
        return 10
    elif hours < 10:
        return 5
    else:
        return 3


def _select_best_reference(files: list[dict]) -> str:
    """Select the best reference audio (medium duration, good RMS)."""
    # Prefer 5-10 second clips
    scored = []
    for f in files:
        dur = f["duration"]
        # Ideal duration: 5-10 seconds
        dur_score = 1.0 - abs(dur - 7.5) / 7.5
        dur_score = max(0, dur_score)
        scored.append((dur_score, f["path"]))
    
    scored.sort(reverse=True)
    return scored[0][1] if scored else files[0]["path"]


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="F5-TTS Enhanced — Custom Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Сценарии:
  single_speaker  — клон одного голоса (5-30 мин аудио)
  emotional       — голос + эмоции (1-5 часов аудио)
  multi_speaker   — универсальная модель (10+ часов)
  test            — проверка результата
  batch           — пакетная генерация

Примеры:
  python custom_train.py single_speaker --audio_dir ./my_voice --output_dir ./my_model
  python custom_train.py test --checkpoint ./my_model/checkpoints/model_last.pt --ref_audio ref.wav --text "Тест"
""")
    
    subparsers = parser.add_subparsers(dest="scenario", required=True)
    
    # ── single_speaker ──
    sp = subparsers.add_parser("single_speaker", help="Клон одного голоса")
    sp.add_argument("--audio_dir", required=True, help="Папка с аудиофайлами")
    sp.add_argument("--transcript_dir", default=None, help="Папка с .txt транскрипциями")
    sp.add_argument("--output_dir", required=True, help="Выходная директория")
    sp.add_argument("--logger", default=None)
    
    # ── emotional ──
    ep = subparsers.add_parser("emotional", help="Голос + эмоции")
    ep.add_argument("--audio_dir", required=True)
    ep.add_argument("--metadata", required=True, help="CSV: audio_path|text|language")
    ep.add_argument("--output_dir", required=True)
    ep.add_argument("--logger", default=None)
    
    # ── multi_speaker ──
    mp = subparsers.add_parser("multi_speaker", help="Универсальная модель")
    mp.add_argument("--dataset_dir", required=True, help="Подготовленный dataset")
    mp.add_argument("--embedding_dir", required=True, help="Предвычисленные эмбеддинги")
    mp.add_argument("--output_dir", required=True)
    mp.add_argument("--logger", default=None)
    
    # ── test ──
    tp = subparsers.add_parser("test", help="Тест модели")
    tp.add_argument("--checkpoint", required=True)
    tp.add_argument("--ref_audio", required=True)
    tp.add_argument("--text", required=True)
    tp.add_argument("--emotion_strength", type=float, default=1.0)
    tp.add_argument("--output", default="test_output.wav")
    
    # ── batch ──
    bp = subparsers.add_parser("batch", help="Пакетная генерация")
    bp.add_argument("--checkpoint", required=True)
    bp.add_argument("--ref_audio", required=True)
    bp.add_argument("--input_texts", required=True, help="Файл с текстами (по строке)")
    bp.add_argument("--output_dir", default="batch_output")
    bp.add_argument("--emotion_strength", type=float, default=1.0)
    
    args = parser.parse_args()
    
    if args.scenario == "single_speaker":
        single_speaker_finetune(args)
    elif args.scenario == "emotional":
        emotional_voice_finetune(args)
    elif args.scenario == "multi_speaker":
        multi_speaker_finetune(args)
    elif args.scenario == "test":
        quick_test(args)
    elif args.scenario == "batch":
        batch_inference(args)


if __name__ == "__main__":
    main()
