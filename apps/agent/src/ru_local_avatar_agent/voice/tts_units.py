from __future__ import annotations

import re

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=\S)", flags=re.UNICODE)
MAX_UNIT_CHARS = 90


def split_tts_units(text: str) -> list[str]:
    """Split one dialogue clause into independently speakable TTS requests.

    Qwen3-TTS can stop after the first sentence when a short multi-sentence
    string is sent as one request. The fix is not phrase-specific punctuation
    rewriting; it is to make TTS serving consume sentence-level units while the
    dialogue/chunking layer still sees one logical clause.
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    units: list[str] = []
    for sentence in SENTENCE_SPLIT_RE.split(normalized):
        units.extend(_split_long_unit(sentence.strip()))
    return [unit for unit in units if unit]


def _split_long_unit(text: str) -> list[str]:
    if len(text) <= MAX_UNIT_CHARS:
        return [text]

    return _split_on_words(text)


def _split_on_words(text: str) -> list[str]:
    words = text.split(" ")
    units: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word]).strip()
        if current and len(candidate) > MAX_UNIT_CHARS:
            units.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        units.append(" ".join(current))
    return units
