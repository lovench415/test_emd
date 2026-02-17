"""
Multilingual Vocabulary
=======================
Расширенный vocab для поддержки 10+ языков без переключения токенизаторов.
Покрывает: EN, ZH (пиньинь), RU, DE, FR, ES, JA (хирагана/катакана), KO, IT, PT, AR, HI.
"""

import re
from pathlib import Path
from typing import Set


# Базовые ASCII символы (общие для всех языков)
BASE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = set("0123456789")
PUNCTUATION = set(".,!?;:'\"-()[]{}…—–·/\\@#$%&*+=<>~`|^_")
WHITESPACE = set(" ")

# Пиньинь (для китайского) — тоновые маркеры
PINYIN_TONES = set("āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ")

# Кириллица (русский, украинский, белорусский)
CYRILLIC = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
               "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
               "іїєґІЇЄҐ")  # украинские буквы

# Немецкие умлауты
GERMAN_CHARS = set("äöüßÄÖÜ")

# Французские акценты
FRENCH_CHARS = set("àâæçéèêëîïôœùûüÿ"
                   "ÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ")

# Испанские символы
SPANISH_CHARS = set("áéíóúñü¡¿"
                    "ÁÉÍÓÚÑÜ")

# Итальянские
ITALIAN_CHARS = set("àèéìíîòóùú"
                    "ÀÈÉÌÍÎÒÓÙÚ")

# Португальские
PORTUGUESE_CHARS = set("ãõâêôáéíóúàçü"
                       "ÃÕÂÊÔÁÉÍÓÚÀÇÜ")

# Японские: хирагана + катакана + некоторые символы
HIRAGANA = set(chr(c) for c in range(0x3040, 0x309F + 1))
KATAKANA = set(chr(c) for c in range(0x30A0, 0x30FF + 1))
JAPANESE_PUNCT = set("。、「」『』（）・ー〜")

# Корейские: джамо (составные элементы)
KOREAN_JAMO = set(chr(c) for c in range(0x1100, 0x11FF + 1))
# Полные слоги хангыль (0xAC00-0xD7AF) — слишком много, берём базовые джамо + совместимые
KOREAN_COMPAT = set(chr(c) for c in range(0x3130, 0x318F + 1))

# Арабские буквы
ARABIC = set(chr(c) for c in range(0x0600, 0x06FF + 1) if chr(c).isalpha() or chr(c) in "ًٌٍَُِّْ")

# Деванагари (хинди)
DEVANAGARI = set(chr(c) for c in range(0x0900, 0x097F + 1))

# Специальные токены
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[MASK]", "[SEP]", "[CLS]"]

# IPA символы для фонетической транскрипции (опционально)
IPA_SYMBOLS = set("ɑɒæɐɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɟɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɻɽɾʀʁɻʂʃʈʧ"
                  "ʉʊʋⱱɤʌɣɯʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑ̈")


def build_multilingual_vocab(
    include_ipa: bool = False,
    include_chinese_chars: bool = False,
    chinese_char_count: int = 5000,
) -> list:
    """
    Собирает полный multilingual vocab.

    Args:
        include_ipa: включить IPA символы (для фонетической транскрипции)
        include_chinese_chars: включить частые китайские иероглифы
        chinese_char_count: сколько частых иероглифов включить

    Returns:
        sorted list of vocab tokens
    """
    all_chars: Set[str] = set()

    # Базовые
    all_chars |= BASE_CHARS
    all_chars |= DIGITS
    all_chars |= PUNCTUATION
    all_chars |= WHITESPACE

    # По языкам
    all_chars |= PINYIN_TONES
    all_chars |= CYRILLIC
    all_chars |= GERMAN_CHARS
    all_chars |= FRENCH_CHARS
    all_chars |= SPANISH_CHARS
    all_chars |= ITALIAN_CHARS
    all_chars |= PORTUGUESE_CHARS
    all_chars |= HIRAGANA
    all_chars |= KATAKANA
    all_chars |= JAPANESE_PUNCT
    all_chars |= KOREAN_JAMO
    all_chars |= KOREAN_COMPAT
    all_chars |= ARABIC
    all_chars |= DEVANAGARI

    if include_ipa:
        all_chars |= IPA_SYMBOLS

    # Частые китайские иероглифы (CJK Unified)
    if include_chinese_chars:
        # Берём первые N из CJK блока (по частотности нужен отдельный список)
        for i in range(min(chinese_char_count, 20000)):
            all_chars.add(chr(0x4E00 + i))

    # Убираем пустые и невалидные
    all_chars = {c for c in all_chars if c.strip() or c == " "}

    # Сортируем для воспроизводимости
    vocab = sorted(all_chars, key=lambda c: (ord(c),))

    return vocab


def save_vocab(vocab: list, path: str):
    """Сохранить vocab.txt в формате F5-TTS (по одному символу на строку)."""
    with open(path, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")
    print(f"Saved vocab with {len(vocab)} tokens to {path}")


def load_vocab(path: str) -> list:
    """Загрузить vocab.txt."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def detect_language(text: str) -> str:
    """
    Простое определение языка по символам текста.
    Для использования при инференсе.
    """
    # Подсчёт символов по Unicode-блокам
    counters = {
        "en": 0, "zh": 0, "ru": 0, "ja": 0, "ko": 0,
        "ar": 0, "hi": 0, "de": 0, "fr": 0, "es": 0,
    }

    for char in text:
        cp = ord(char)

        if 0x0400 <= cp <= 0x04FF:
            counters["ru"] += 1
        elif 0x4E00 <= cp <= 0x9FFF:
            counters["zh"] += 1
        elif 0x3040 <= cp <= 0x30FF:
            counters["ja"] += 1
        elif (0xAC00 <= cp <= 0xD7AF) or (0x1100 <= cp <= 0x11FF):
            counters["ko"] += 1
        elif 0x0600 <= cp <= 0x06FF:
            counters["ar"] += 1
        elif 0x0900 <= cp <= 0x097F:
            counters["hi"] += 1
        elif char in GERMAN_CHARS:
            counters["de"] += 1
        elif char in FRENCH_CHARS:
            counters["fr"] += 1
        elif char in SPANISH_CHARS:
            counters["es"] += 1
        elif char.isascii() and char.isalpha():
            counters["en"] += 1

    if sum(counters.values()) == 0:
        return "en"

    # Если есть CJK/кириллица/etc — приоритет им
    non_latin = {k: v for k, v in counters.items() if k not in ("en", "de", "fr", "es")}
    if sum(non_latin.values()) > 0:
        return max(non_latin, key=non_latin.get)

    return max(counters, key=counters.get)


if __name__ == "__main__":
    vocab = build_multilingual_vocab(
        include_ipa=False,
        include_chinese_chars=True,
        chinese_char_count=5000,
    )
    print(f"Total vocab size: {len(vocab)}")
    save_vocab(vocab, "vocab_multilingual.txt")

    # Тест определения языка
    test_texts = [
        ("Hello world", "en"),
        ("Привет мир", "ru"),
        ("你好世界", "zh"),
        ("こんにちは世界", "ja"),
        ("안녕하세요 세계", "ko"),
        ("Hallo Welt, schöne Grüße", "de"),
        ("Bonjour le monde, ça va", "fr"),
    ]
    for text, expected in test_texts:
        detected = detect_language(text)
        status = "✓" if detected == expected else "✗"
        print(f"  {status} '{text[:30]}' → detected: {detected}, expected: {expected}")
