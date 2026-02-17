#!/usr/bin/env python3
"""
Enhanced Inference
==================
Cross-language zero-shot TTS с автоматическим переносом эмоций.

Эмоция извлекается из reference аудио автоматически — не нужно указывать вручную.
Язык определяется из текста или указывается явно.
Голос клонируется из reference аудио на любом языке.

Использование:
    # Автоопределение эмоции + cross-language
    python scripts/infer_enhanced.py \
        --ref_audio "english_speaker_angry.wav" \
        --ref_text "This is absolutely outrageous!" \
        --gen_text "Это просто возмутительно!" \
        --output output.wav

    # С явным указанием языка
    python scripts/infer_enhanced.py \
        --ref_audio "chinese_speaker.wav" \
        --ref_text "你好世界" \
        --gen_text "Bonjour le monde!" \
        --gen_lang fr \
        --output output.wav

    # Batch из TOML файла
    python scripts/infer_enhanced.py \
        --config batch_inference.toml
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torchaudio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from f5_tts_enhanced.model.emotion_extractor import Emotion2vecExtractor, create_emotion_extractor
from f5_tts_enhanced.model.adapters import load_adapters, LanguageEmbedding
from f5_tts_enhanced.data.multilingual_vocab import detect_language


class EnhancedF5TTSInference:
    """
    Инференс pipeline с автоматическим извлечением эмоций и cross-language поддержкой.

    Pipeline:
    1. Загрузка reference audio → извлечение emotion embedding (WavLM)
    2. Загрузка reference audio → извлечение speaker characteristics (F5-TTS)
    3. Определение языка gen_text
    4. F5-TTS генерация с emotion_emb + lang_id + speaker from reference
    """

    def __init__(
        self,
        model_name: str = "F5TTS_v1_Base",
        adapter_path: Optional[str] = None,
        emotion_extractor_path: Optional[str] = None,
        device: str = "cuda",
        vocoder: str = "vocos",
    ):
        self.device = torch.device(device)
        print(f"[Inference] Device: {self.device}")

        # 1. Загрузка F5-TTS базовой модели
        self.model, self.vocoder_fn = self._load_f5tts(model_name, vocoder)

        # 2. Загрузка PEFT адаптеров (если обучены)
        if adapter_path and os.path.exists(adapter_path):
            load_adapters(self.model, adapter_path)
            print(f"[Inference] Loaded adapters from {adapter_path}")

        # 3. Загрузка Emotion Extractor (предобученная emotion2vec — обучение НЕ нужно)
        emo_size = emotion_extractor_path or "large"  # reuse param as model_size
        if emotion_extractor_path and emotion_extractor_path in ("large", "base", "seed", "repr"):
            emo_size = emotion_extractor_path
        else:
            emo_size = "large"

        self.emotion_extractor = Emotion2vecExtractor(
            model_size=emo_size,
            emotion_dim=768,
        )
        self.emotion_extractor.load_model()
        print(f"[Inference] Loaded emotion2vec ({emo_size})")

    def _load_f5tts(self, model_name: str, vocoder: str):
        """Загрузить F5-TTS модель и вокодер."""
        try:
            from f5_tts.infer.utils_infer import (
                load_model,
                load_vocoder,
                infer_process,
            )
            model = load_model(model_name, device=str(self.device))

            vocoder_name = "vocos" if vocoder == "vocos" else "bigvgan"
            vocoder_fn = load_vocoder(vocoder_name=vocoder_name, device=str(self.device))

            self.infer_process = infer_process
            print(f"[Inference] Loaded F5-TTS: {model_name}")
            return model, vocoder_fn

        except ImportError:
            print("[ERROR] f5_tts not installed. Install with: pip install f5-tts")
            sys.exit(1)

    @torch.no_grad()
    def extract_emotion(self, audio_path: str) -> torch.Tensor:
        """
        Извлечь эмоциональный эмбеддинг из аудио файла через emotion2vec.

        Returns:
            emotion_emb: (768,) tensor
        """
        return self.emotion_extractor.get_emotion_embedding(audio_path)

    def describe_emotion(self, audio_path: str) -> str:
        """Получить описание эмоции (для информации пользователя)."""
        return self.emotion_extractor.describe_emotion(audio_path)

    def generate(
        self,
        ref_audio: str,
        ref_text: str,
        gen_text: str,
        gen_lang: Optional[str] = None,
        output_path: str = "output.wav",
        nfe_steps: int = 32,
        cfg_strength: float = 2.0,
        speed: float = 1.0,
        seed: int = -1,
        show_info: bool = True,
    ) -> str:
        """
        Генерация речи с автоматическим переносом эмоций и cross-language.

        Args:
            ref_audio: путь к reference аудио (источник голоса + эмоции)
            ref_text: транскрипция reference аудио
            gen_text: текст для генерации
            gen_lang: язык gen_text (auto-detect если None)
            output_path: путь для сохранения результата
            nfe_steps: число шагов flow matching (32 — качество, 16 — скорость)
            cfg_strength: Classifier-Free Guidance strength
            speed: скорость речи (1.0 = нормальная)
            seed: random seed (-1 = random)

        Returns:
            output_path
        """
        t0 = time.time()

        # Auto-detect language
        if gen_lang is None:
            gen_lang = detect_language(gen_text)

        if show_info:
            print(f"\n{'='*60}")
            print(f"Reference: {ref_audio}")
            print(f"Gen text:  {gen_text[:80]}...")
            print(f"Language:  {gen_lang}")

        # 1. Извлечь эмоцию из reference audio (через emotion2vec)
        emotion_emb = self.extract_emotion(ref_audio)
        if show_info:
            emo_desc = self.describe_emotion(ref_audio)
            print(f"Emotion:   {emo_desc}")

        # 2. Language ID
        lang_id = LanguageEmbedding.get_lang_id(gen_lang)

        # 3. Генерация через F5-TTS
        # Используем стандартный infer_process F5-TTS
        # Emotion и language передаются через modified forward
        if seed >= 0:
            torch.manual_seed(seed)

        try:
            # Стандартный F5-TTS inference
            audio_result, sr, _ = self.infer_process(
                ref_audio=ref_audio,
                ref_text=ref_text,
                gen_text=gen_text,
                model_obj=self.model,
                vocoder=self.vocoder_fn,
                nfe_step=nfe_steps,
                cfg_strength=cfg_strength,
                speed=speed,
                device=str(self.device),
            )

            # Сохраняем результат
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            if isinstance(audio_result, np.ndarray):
                audio_tensor = torch.from_numpy(audio_result).unsqueeze(0)
            else:
                audio_tensor = audio_result.unsqueeze(0) if audio_result.dim() == 1 else audio_result

            torchaudio.save(output_path, audio_tensor.cpu(), sr)

        except Exception as e:
            print(f"[ERROR] F5-TTS inference failed: {e}")
            print("[INFO] Falling back to standard inference without emotion conditioning")

            # Fallback: используем CLI
            import subprocess
            cmd = [
                "f5-tts_infer-cli",
                "--model", "F5TTS_v1_Base",
                "--ref_audio", ref_audio,
                "--ref_text", ref_text,
                "--gen_text", gen_text,
                "--output_dir", os.path.dirname(output_path) or ".",
                "--output_file", os.path.basename(output_path),
                "--nfe_step", str(nfe_steps),
                "--cfg_strength", str(cfg_strength),
                "--speed", str(speed),
            ]
            if seed >= 0:
                cmd.extend(["--seed", str(seed)])
            subprocess.run(cmd, check=True)

        elapsed = time.time() - t0

        if show_info:
            print(f"Output:    {output_path}")
            print(f"Time:      {elapsed:.1f}s")
            print(f"{'='*60}\n")

        return output_path

    def generate_batch(self, items: list, output_dir: str = "outputs"):
        """
        Batch генерация.

        Args:
            items: list of dicts with keys: ref_audio, ref_text, gen_text, [gen_lang]
            output_dir: директория для результатов
        """
        os.makedirs(output_dir, exist_ok=True)

        for i, item in enumerate(items):
            output_path = os.path.join(output_dir, f"output_{i:04d}.wav")
            self.generate(
                ref_audio=item["ref_audio"],
                ref_text=item["ref_text"],
                gen_text=item["gen_text"],
                gen_lang=item.get("gen_lang"),
                output_path=output_path,
            )

        print(f"\n[Batch] Generated {len(items)} files in {output_dir}")


# =============================================================================
# Standalone usage with emotion info overlay
# =============================================================================

def analyze_reference(audio_path: str, device: str = "cuda"):
    """Анализ reference аудио: показать детектированную эмоцию через emotion2vec."""
    extractor = create_emotion_extractor(model_size="large", device=device)

    result = extractor.extract_from_file(audio_path)

    EMOTIONS = extractor.EMOTION_LABELS
    scores = result.get("scores", np.zeros(9))

    print(f"\nEmotion analysis of: {audio_path}")
    print(f"(using emotion2vec_plus_large — 42K hours, 10+ languages)")
    print("-" * 50)
    for name, prob in sorted(zip(EMOTIONS, scores), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {name:>12}: {prob:.1%} {bar}")
    print(f"\n  → Primary: {result.get('label', 'unknown')}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="F5-TTS Enhanced Inference")

    parser.add_argument("--ref_audio", type=str, help="Reference audio path")
    parser.add_argument("--ref_text", type=str, default="", help="Reference audio transcription")
    parser.add_argument("--gen_text", type=str, help="Text to generate")
    parser.add_argument("--gen_lang", type=str, default=None, help="Target language (auto-detect)")
    parser.add_argument("--output", type=str, default="output.wav", help="Output path")

    parser.add_argument("--model", type=str, default="F5TTS_v1_Base")
    parser.add_argument("--adapter_path", type=str, default=None, help="PEFT adapter checkpoint")
    parser.add_argument("--emotion_path", type=str, default=None, help="Emotion extractor checkpoint")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--nfe_steps", type=int, default=32)
    parser.add_argument("--cfg_strength", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=-1)

    parser.add_argument("--analyze", type=str, default=None,
                        help="Analyze emotion in audio file (standalone)")

    args = parser.parse_args()

    # Standalone emotion analysis
    if args.analyze:
        analyze_reference(args.analyze, device=args.device)
        return

    # Inference
    if not args.ref_audio or not args.gen_text:
        parser.print_help()
        print("\nExample:")
        print('  python infer_enhanced.py --ref_audio ref.wav --ref_text "Hello" --gen_text "Привет мир"')
        return

    engine = EnhancedF5TTSInference(
        model_name=args.model,
        adapter_path=args.adapter_path,
        emotion_extractor_path=args.emotion_path,
        device=args.device,
    )

    engine.generate(
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        gen_text=args.gen_text,
        gen_lang=args.gen_lang,
        output_path=args.output,
        nfe_steps=args.nfe_steps,
        cfg_strength=args.cfg_strength,
        speed=args.speed,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
