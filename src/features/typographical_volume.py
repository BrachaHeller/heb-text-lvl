"""Typographical Nikkud vocalization extractor and syllable counters via Dicta ONNX."""

import logging
import re
from typing import Optional, Tuple

from dicta_onnx import Dicta

from hebrew_readability.hf_hub import ensure_dicta_onnx_model

logger = logging.getLogger(__name__)

HEBREW_LETTERS = re.compile(r"[\u05D0-\u05EA]")
HEBREW_NIKKUD = re.compile(r"[\u0591-\u05C7]")
VOWELS_RE = re.compile(r"[\u05B1-\u05BB]")
SHURUK_RE = re.compile(r"\u05D5\u05BC")

_DICTA_ONNX_INSTANCE: Optional[Dicta] = None


def _get_dicta_onnx() -> Dicta:
    """Lazy singleton loader for Dicta ONNX model."""
    global _DICTA_ONNX_INSTANCE
    if _DICTA_ONNX_INSTANCE is None:
        onnx_path = ensure_dicta_onnx_model()
        _DICTA_ONNX_INSTANCE = Dicta(onnx_path)
    return _DICTA_ONNX_INSTANCE


def get_base_features(text: str) -> Tuple[float, str, int]:
    """Calculates Nikkud ratio, sanitizes text, and counts physical words."""
    if not isinstance(text, str) or not text.strip():
        return 0.0, "", 0

    words = text.split()
    total_hebrew_words, vocalized_words = 0, 0

    for word in words:
        if HEBREW_LETTERS.search(word):
            total_hebrew_words += 1
            if HEBREW_NIKKUD.search(word):
                vocalized_words += 1

    nikkud_ratio = (
        round(vocalized_words / total_hebrew_words, 4) if total_hebrew_words > 0 else 0.0
    )

    clean_text = HEBREW_NIKKUD.sub("", text)
    clean_text = re.sub(r"<[^>]+>", "", clean_text)
    clean_text = re.sub(r"[\u200B-\u200D\uFEFF]", "", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    num_physical_words = len(clean_text.split()) if clean_text else 0
    return nikkud_ratio, clean_text, num_physical_words


def get_syllable_features(
    clean_text: str, num_physical_words: int, stz_sentences: int
) -> Tuple[str, int, float, float]:
    """Vocalizes text via Dicta ONNX and computes syllable volume ratios."""
    if not clean_text:
        return "", 0, 0.0, 0.0

    dicta_model = _get_dicta_onnx()
    try:
        vocalized_text = dicta_model.add_diacritics(clean_text)
    except Exception as exc:
        logger.warning(f"ONNX Vocalization Error: {exc}")
        vocalized_text = clean_text

    try:
        total_syllables = 0
        for word in vocalized_text.split():
            clean_word = re.sub(r"[^\w\u0591-\u05C7]", "", word)
            if clean_word:
                vowels = len(VOWELS_RE.findall(clean_word))
                shuruks = len(SHURUK_RE.findall(clean_word))
                syllables = vowels + shuruks
                if any(c.isalpha() for c in clean_word):
                    total_syllables += max(1, syllables)

        safe_words = max(1, num_physical_words)
        safe_sents = max(1, stz_sentences)

        avg_syl_per_word = round(total_syllables / safe_words, 3)
        avg_syl_per_sent = round(total_syllables / safe_sents, 3)

        return vocalized_text, total_syllables, avg_syl_per_word, avg_syl_per_sent

    except Exception as exc:
        logger.warning(f"Syllable Math Error: {exc}")
        return vocalized_text, 0, 0.0, 0.0