"""
Emotion Extractor (emotion2vec) + Кеширование
===============================================
Предобученная emotion2vec+ с двухуровневым кешем:

  1. Memory LRU — мгновенный доступ, ограничен по размеру
  2. Disk cache  — .npy файлы, переживают перезапуск

Ключ кеша: SHA-256 от содержимого аудио файла.
При обучении (десятки тысяч обращений к тем же ref audio) кеш
ускоряет pipeline в 50-100× — emotion2vec вызывается один раз на файл.

Зависимости:
    pip install funasr
"""

import hashlib
import json
import os
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torchaudio


# =========================================================================
# Embedding Cache
# =========================================================================

class EmbeddingCache:
    """
    Двухуровневый кеш для emotion embeddings.

    Level 1: In-memory LRU (OrderedDict, потокобезопасный)
    Level 2: Disk (.npy файлы в cache_dir)

    Ключ: SHA-256 хеш содержимого аудио файла.
    Используем хеш содержимого, а не путь, потому что:
      - один и тот же файл может быть по разным путям (symlinks, copies)
      - файл по тому же пути может измениться

    Формат disk cache:
      cache_dir/
        ab/ab3f...sha256.utterance.npy   — utterance embedding (768,)
        ab/ab3f...sha256.frame.npy       — frame embedding (T, 768)
        ab/ab3f...sha256.meta.json       — scores, label, label_idx
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_memory_items: int = 1000,
        enable_disk: bool = True,
        enable_memory: bool = True,
    ):
        # Memory LRU
        self._memory: OrderedDict[str, dict] = OrderedDict()
        self._max_memory = max_memory_items
        self._enable_memory = enable_memory
        self._lock = threading.Lock()

        # Disk
        self._enable_disk = enable_disk
        if cache_dir is None:
            cache_dir = os.path.join(
                os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
                "f5tts_emotion_cache",
            )
        self._cache_dir = Path(cache_dir)
        if enable_disk:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._hits = 0
        self._misses = 0

    # ---------- hash ----------

    @staticmethod
    def file_hash(audio_path: str) -> str:
        """SHA-256 хеш содержимого файла (быстрый, читает блоками)."""
        h = hashlib.sha256()
        with open(audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):  # 64KB blocks
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def tensor_hash(audio: torch.Tensor) -> str:
        """SHA-256 от сырых байт тензора."""
        data = audio.cpu().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    # ---------- disk paths ----------

    def _disk_dir(self, key: str) -> Path:
        """Поддиректория по первым 2 символам хеша (для разреживания)."""
        return self._cache_dir / key[:2]

    def _disk_path(self, key: str, granularity: str) -> Path:
        return self._disk_dir(key) / f"{key}.{granularity}.npy"

    def _meta_path(self, key: str) -> Path:
        return self._disk_dir(key) / f"{key}.meta.json"

    # ---------- memory ----------

    def _memory_get(self, key: str, granularity: str) -> Optional[dict]:
        if not self._enable_memory:
            return None
        cache_key = f"{key}:{granularity}"
        with self._lock:
            if cache_key in self._memory:
                self._memory.move_to_end(cache_key)
                return self._memory[cache_key]
        return None

    def _memory_put(self, key: str, granularity: str, data: dict):
        if not self._enable_memory:
            return
        cache_key = f"{key}:{granularity}"
        with self._lock:
            self._memory[cache_key] = data
            self._memory.move_to_end(cache_key)
            while len(self._memory) > self._max_memory:
                self._memory.popitem(last=False)

    # ---------- disk ----------

    def _disk_get(self, key: str, granularity: str) -> Optional[dict]:
        if not self._enable_disk:
            return None
        emb_path = self._disk_path(key, granularity)
        meta_path = self._meta_path(key)
        if not emb_path.exists():
            return None

        try:
            embedding = np.load(str(emb_path))
            meta = {}
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            return {
                "embedding": embedding,
                "scores": np.array(meta.get("scores", [])),
                "label": meta.get("label", "neutral"),
                "label_idx": meta.get("label_idx", 4),
            }
        except Exception:
            return None

    def _disk_put(self, key: str, granularity: str, data: dict):
        if not self._enable_disk:
            return
        try:
            d = self._disk_dir(key)
            d.mkdir(parents=True, exist_ok=True)

            np.save(str(self._disk_path(key, granularity)), data["embedding"])

            meta = {
                "label": data.get("label", "neutral"),
                "label_idx": int(data.get("label_idx", 4)),
            }
            if "scores" in data and data["scores"] is not None:
                scores = data["scores"]
                if isinstance(scores, np.ndarray):
                    scores = scores.tolist()
                meta["scores"] = scores

            with open(self._meta_path(key), "w") as f:
                json.dump(meta, f)
        except Exception:
            pass  # не фатально если кеш не записался

    # ---------- public API ----------

    def get(self, key: str, granularity: str = "utterance") -> Optional[dict]:
        """Получить из кеша (memory → disk). None если нет."""
        # L1: memory
        result = self._memory_get(key, granularity)
        if result is not None:
            self._hits += 1
            return result

        # L2: disk
        result = self._disk_get(key, granularity)
        if result is not None:
            self._memory_put(key, granularity, result)  # promote to L1
            self._hits += 1
            return result

        self._misses += 1
        return None

    def put(self, key: str, granularity: str, data: dict):
        """Записать в оба уровня кеша."""
        self._memory_put(key, granularity, data)
        self._disk_put(key, granularity, data)

    def clear_memory(self):
        """Очистить только memory cache."""
        with self._lock:
            self._memory.clear()

    def clear_all(self):
        """Очистить memory + disk."""
        self.clear_memory()
        if self._enable_disk and self._cache_dir.exists():
            import shutil
            shutil.rmtree(self._cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self._hits / total:.1%}" if total > 0 else "N/A",
            "memory_size": len(self._memory),
            "cache_dir": str(self._cache_dir),
        }


# =========================================================================
# Emotion Extractor
# =========================================================================

class Emotion2vecExtractor(nn.Module):
    """
    emotion2vec+ с двухуровневым кешем embeddings.

    Кеш ускоряет:
    - Обучение: ref audio повторяются → 50-100× speedup на извлечении эмоций
    - Инференс: повторные запросы с тем же reference
    - Batch: один speaker для нескольких gen_text

    Использование:
        ext = Emotion2vecExtractor(cache_dir="./cache/emotion")
        ext.load_model()

        emb = ext.get_emotion_embedding("ref.wav")   # 1й вызов: ~200ms (модель)
        emb = ext.get_emotion_embedding("ref.wav")   # 2й вызов: <1ms (memory)
        # после перезапуска:
        emb = ext.get_emotion_embedding("ref.wav")   # ~2ms (disk → memory)
    """

    EMOTION_LABELS = [
        "angry", "disgusted", "fearful", "happy", "neutral",
        "other", "sad", "surprised", "unknown",
    ]

    FEAT_DIM = 768
    NUM_CLASSES = 9

    MODELS = {
        "large": "iic/emotion2vec_plus_large",
        "base": "iic/emotion2vec_plus_base",
        "seed": "iic/emotion2vec_plus_seed",
        "repr": "iic/emotion2vec_base",
    }

    def __init__(
        self,
        model_size: str = "large",
        emotion_dim: int = 768,
        device: str = "cuda",
        hub: str = "hf",
        # --- Кеширование ---
        cache_dir: Optional[str] = None,
        max_memory_items: int = 1000,
        enable_disk_cache: bool = True,
        enable_memory_cache: bool = True,
    ):
        super().__init__()
        self.model_size = model_size
        self.native_dim = self.FEAT_DIM
        self.emotion_dim = emotion_dim
        self.device_str = device
        self._model = None
        self._hub = hub

        # Проекция если нужна другая размерность
        if emotion_dim != self.native_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.native_dim, emotion_dim),
                nn.LayerNorm(emotion_dim),
            )
        else:
            self.projection = nn.Identity()

        # Кеш
        self.cache = EmbeddingCache(
            cache_dir=cache_dir,
            max_memory_items=max_memory_items,
            enable_disk=enable_disk_cache,
            enable_memory=enable_memory_cache,
        )

    def load_model(self):
        """Загрузить предобученную emotion2vec. Вызывать один раз."""
        from funasr import AutoModel

        model_id = self.MODELS.get(self.model_size, self.MODELS["large"])
        print(f"[emotion2vec] Loading {model_id} (hub={self._hub})...")
        self._model = AutoModel(model=model_id, hub=self._hub)
        print(f"[emotion2vec] Loaded. Cache: {self.cache._cache_dir}")

    def _ensure_model(self):
        if self._model is None:
            self.load_model()

    # ---------- core extraction (no cache) ----------

    @torch.no_grad()
    def _extract_raw(self, audio_path: str, granularity: str = "utterance") -> dict:
        """Прямой вызов emotion2vec без кеша."""
        self._ensure_model()

        result = self._model.generate(
            audio_path,
            output_dir=None,
            granularity=granularity,
            extract_embedding=True,
        )

        res = result[0] if isinstance(result, list) and len(result) > 0 else result
        output = {}

        if "feats" in res:
            feats = res["feats"]
            if isinstance(feats, list):
                feats = np.array(feats)
            if isinstance(feats, torch.Tensor):
                feats = feats.cpu().numpy()
            output["embedding"] = feats
        else:
            output["embedding"] = np.zeros(self.native_dim)

        is_repr_only = (self.model_size == "repr")
        if not is_repr_only and "scores" in res:
            scores = np.array(res["scores"]) if isinstance(res["scores"], list) else res["scores"]
            output["scores"] = scores
            label_idx = int(np.argmax(scores))
            output["label_idx"] = label_idx
            output["label"] = self.EMOTION_LABELS[label_idx]
        else:
            output["label"] = "neutral"
            output["label_idx"] = 4
            output["scores"] = np.zeros(self.NUM_CLASSES)

        return output

    # ---------- cached extraction ----------

    @torch.no_grad()
    def extract_from_file(
        self,
        audio_path: str,
        granularity: str = "utterance",
    ) -> dict:
        """
        Извлечь эмоцию из файла С КЕШИРОВАНИЕМ.

        1-й вызов: emotion2vec inference → записать в memory + disk
        2+ вызовы: мгновенно из memory (или из disk после перезапуска)
        """
        key = self.cache.file_hash(audio_path)

        # Проверить кеш
        cached = self.cache.get(key, granularity)
        if cached is not None:
            return cached

        # Cache miss → извлечь
        result = self._extract_raw(audio_path, granularity)

        # Записать в кеш
        self.cache.put(key, granularity, result)

        return result

    @torch.no_grad()
    def extract_from_tensor(
        self,
        audio: torch.Tensor,
        sr: int = 16000,
    ) -> dict:
        """Извлечь эмоцию из tensor (кеш по хешу тензора)."""
        if audio.dim() > 1:
            audio = audio.squeeze(0)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)

        # Кеш по хешу содержимого тензора
        key = self.cache.tensor_hash(audio)
        cached = self.cache.get(key, "utterance")
        if cached is not None:
            return cached

        # Сохранить temp file → извлечь
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            torchaudio.save(tmp_path, audio.unsqueeze(0).cpu(), 16000)
        try:
            result = self._extract_raw(tmp_path, granularity="utterance")
        finally:
            os.unlink(tmp_path)

        self.cache.put(key, "utterance", result)
        return result

    # ---------- high-level API ----------

    @torch.no_grad()
    def get_emotion_embedding(
        self,
        audio: Union[str, torch.Tensor],
        sr: int = 16000,
    ) -> torch.Tensor:
        """
        ★ Основной метод — embedding как torch.Tensor. Кешируется.

        Returns:
            (emotion_dim,) tensor — по умолчанию (768,)
        """
        if isinstance(audio, str):
            result = self.extract_from_file(audio, granularity="utterance")
        else:
            result = self.extract_from_tensor(audio, sr=sr)

        embedding = torch.from_numpy(result["embedding"]).float()

        if self.emotion_dim != self.native_dim:
            device = next(self.projection.parameters()).device
            embedding = self.projection(embedding.to(device))

        return embedding

    @torch.no_grad()
    def get_batch_embeddings(self, audio_paths: list) -> torch.Tensor:
        """Batch: (batch, emotion_dim). Каждый файл кешируется отдельно."""
        return torch.stack([self.get_emotion_embedding(p) for p in audio_paths])

    @torch.no_grad()
    def get_frame_embeddings(
        self,
        audio: Union[str, torch.Tensor],
        sr: int = 16000,
    ) -> torch.Tensor:
        """Frame-level (50Hz, T×768). Кешируется."""
        if isinstance(audio, str):
            result = self.extract_from_file(audio, granularity="frame")
        else:
            if audio.dim() > 1:
                audio = audio.squeeze(0)
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)
            # Для frame-level — через temp file
            key = self.cache.tensor_hash(audio)
            cached = self.cache.get(key, "frame")
            if cached is not None:
                frames = torch.from_numpy(cached["embedding"]).float()
                if self.emotion_dim != self.native_dim:
                    frames = self.projection(frames)
                return frames

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
                torchaudio.save(tmp_path, audio.unsqueeze(0).cpu(), 16000)
            try:
                result = self._extract_raw(tmp_path, granularity="frame")
            finally:
                os.unlink(tmp_path)
            self.cache.put(key, "frame", result)

        frames = torch.from_numpy(result["embedding"]).float()
        if self.emotion_dim != self.native_dim:
            frames = self.projection(frames)
        return frames

    def describe_emotion(
        self,
        audio: Union[str, torch.Tensor],
        sr: int = 16000,
    ) -> str:
        """Человекочитаемое описание. Тоже кешируется."""
        if isinstance(audio, str):
            result = self.extract_from_file(audio)
        else:
            result = self.extract_from_tensor(audio, sr=sr)

        if "scores" in result and np.any(result["scores"]):
            pairs = sorted(
                zip(self.EMOTION_LABELS, result["scores"]),
                key=lambda x: -x[1]
            )
            desc = ", ".join(f"{n} ({s:.1%})" for n, s in pairs if s > 0.05)
            return f"{result['label']} → [{desc}]"
        return result.get("label", "unknown")

    def print_cache_stats(self):
        """Вывести статистику кеша."""
        s = self.cache.stats
        print(f"[emotion2vec cache] hits={s['hits']}, misses={s['misses']}, "
              f"rate={s['hit_rate']}, memory={s['memory_size']}, "
              f"dir={s['cache_dir']}")

    def warmup_cache(self, audio_paths: list):
        """
        Прогреть кеш заранее (напр. перед обучением).
        Извлекает embeddings для всех файлов и сохраняет в disk+memory.
        """
        total = len(audio_paths)
        cached = 0
        computed = 0
        for i, path in enumerate(audio_paths):
            key = self.cache.file_hash(path)
            if self.cache.get(key, "utterance") is None:
                self.extract_from_file(path, "utterance")
                computed += 1
            else:
                cached += 1
            if (i + 1) % 500 == 0 or i == total - 1:
                print(f"[warmup] {i+1}/{total} — {computed} computed, {cached} cached")
        print(f"[warmup] Done: {computed} new, {cached} from cache")


# =========================================================================
# Factory
# =========================================================================

def create_emotion_extractor(
    model_size: str = "large",
    emotion_dim: int = 768,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
) -> Emotion2vecExtractor:
    """
    Создать и загрузить emotion extractor с кешем.

    >>> ext = create_emotion_extractor("large", cache_dir="./cache/emo")
    >>> emb = ext.get_emotion_embedding("ref.wav")   # ~200ms (1-й раз)
    >>> emb = ext.get_emotion_embedding("ref.wav")   # <1ms (кеш)
    >>> ext.print_cache_stats()
    # hits=1, misses=1, rate=50.0%, memory=1
    """
    extractor = Emotion2vecExtractor(
        model_size=model_size,
        emotion_dim=emotion_dim,
        device=device,
        cache_dir=cache_dir,
    )
    extractor.load_model()
    return extractor
