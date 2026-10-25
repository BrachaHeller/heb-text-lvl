"""Typographical Nikkud vocalization extractor and syllable counters via Dicta ONNX."""

import logging
import re
from typing import List, Optional, Tuple

from dicta_onnx import Dicta

from hebrew_readability.hf_hub import ensure_dicta_onnx_model

logger = logging.getLogger(__name__)

HEBREW_LETTERS = re.compile(r"[\u05D0-\u05EA]")
HEBREW_NIKKUD = re.compile(r"[\u0591-\u05C7]")
VOWELS_RE = re.compile(r"[\u05B1-\u05BB]")
SHURUK_RE = re.compile(r"\u05D5\u05BC")

# Dicta ONNX character-level BERT has a hard positional embedding limit of 2,048 tokens.
# Keeping chunks <= 1,800 characters leaves safe headroom for [CLS], [SEP], and special chars.
MAX_ONNX_CHARS = 1800

_DICTA_ONNX_INSTANCE: Optional[Dicta] = None


def _get_dicta_onnx() -> Dicta:
    """Lazy singleton loader for Dicta ONNX model."""
    global _DICTA_ONNX_INSTANCE
    if _DICTA_ONNX_INSTANCE is None:
        onnx_path = ensure_dicta_onnx_model()
        _DICTA_ONNX_INSTANCE = Dicta(onnx_path)
    return _DICTA_ONNX_INSTANCE


def _split_into_safe_onnx_chunks(text: str, max_chars: int = MAX_ONNX_CHARS) -> List[str]:
    """Splits text > max_chars along sentence, clause, or word boundaries without dropping words."""
    if len(text) <= max_chars:
        return [text]

    # 1. Split by sentence-ending punctuation first
    raw_segments = re.split(r"(?<=[.!?])\s+", text)
    safe_segments: List[str] = []

    # 2. If any single sentence is still > max_chars, split by clauses/commas, then words
    for seg in raw_segments:
        if len(seg) <= max_chars:
            safe_segments.append(seg)
        else:
            sub_clauses = re.split(r"(?<=[;:,])\s+", seg)
            for clause in sub_clauses:
                if len(clause) <= max_chars:
                    safe_segments.append(clause)
                else:
                    # Fallback: pack by whitespace words so no word is ever truncated
                    words = clause.split()
                    word_chunk = ""
                    for w in words:
                        if len(word_chunk) + len(w) + 1 <= max_chars:
                            word_chunk = f"{word_chunk} {w}".strip()
                        else:
                            if word_chunk:
                                safe_segments.append(word_chunk)
                            # Guard against a single contiguous non-whitespace string > max_chars
                            word_chunk = w[:max_chars]
                    if word_chunk:
                        safe_segments.append(word_chunk)

    # 3. Greedily pack segments into chunks up to max_chars to preserve maximum context
    packed_chunks: List[str] = []
    current = ""
    for seg in safe_segments:
        if not current:
            current = seg
        elif len(current) + 1 + len(seg) <= max_chars:
            current = f"{current} {seg}"
        else:
            packed_chunks.append(current)
            current = seg
    if current:
        packed_chunks.append(current)

    return packed_chunks


def _vocalize_safely(clean_text: str, dicta_model: Dicta) -> str:
    """Vocalizes full paragraph if <= 1800 chars, or chunks proactively to prevent 2048-token crash."""
    if len(clean_text) <= MAX_ONNX_CHARS:
        try:
            return dicta_model.add_diacritics(clean_text)
        except Exception as exc:
            logger.warning(f"ONNX Vocalization Error on standard text: {exc}")
            return clean_text

    chunks = _split_into_safe_onnx_chunks(clean_text, max_chars=MAX_ONNX_CHARS)
    vocalized_parts: List[str] = []
    for chunk in chunks:
        try:
            vocalized_parts.append(dicta_model.add_diacritics(chunk))
        except Exception as exc:
            logger.warning(f"ONNX Vocalization Error on chunk (len={len(chunk)}): {exc}")
            vocalized_parts.append(chunk)
    return " ".join(vocalized_parts)


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
    vocalized_text = _vocalize_safely(clean_text, dicta_model)

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