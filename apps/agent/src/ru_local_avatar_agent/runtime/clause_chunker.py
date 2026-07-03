from __future__ import annotations

import re
from dataclasses import dataclass


HARD_BOUNDARIES = {".", "!", "?", "...", "…"}
SOFT_BOUNDARIES = {",", ";", ":"}
MIN_CHUNK_CHARS = 8
MIN_HARD_BOUNDARY_CHARS = 16
MIN_SOFT_BOUNDARY_CHARS = 24
MIN_TOKEN_TARGET_CHARS = 36
MIN_TOKEN_TARGET_WORDS = 6
INCOMPLETE_TOKEN_TARGET_TAILS = {
    "а",
    "был",
    "была",
    "были",
    "было",
    "в",
    "во",
    "для",
    "и",
    "как",
    "когда",
    "могу",
    "можем",
    "можешь",
    "на",
    "но",
    "ну",
    "о",
    "по",
    "поняла",
    "про",
    "с",
    "та",
    "так",
    "такая",
    "такой",
    "там",
    "тебе",
    "тебя",
    "ты",
    "у",
    "хочу",
    "что",
    "чтобы",
    "это",
    "я",
}


@dataclass(frozen=True, slots=True)
class ClauseChunk:
    text: str
    reason: str


def normalize_tts_text(text: str) -> str:
    """Normalize text only at TTS-safe typography boundaries.

    LLM streams emit text deltas, not words. We must preserve subword joins
    exactly, then clean punctuation spacing that would otherwise be spoken.
    """

    normalized = text.replace("\u00a0", " ")
    normalized = re.sub(r"\s+[—–]\s+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s+([,.;:!?…])", r"\1", normalized)
    normalized = re.sub(r"([(\[{«])\s+", r"\1", normalized)
    normalized = re.sub(r"\s+([)\]}»])", r"\1", normalized)
    normalized = re.sub(r"(?<=[^\W\d_])\s*-\s*(?=[^\W\d_])", "-", normalized)
    return normalized.strip()


class ClauseChunker:
    def __init__(self, token_target: int = 12) -> None:
        if token_target < 4:
            raise ValueError("token_target must be >= 4")
        self.token_target = token_target
        self._buffer = ""
        self._deltas_since_flush = 0

    def push(self, token: str) -> ClauseChunk | None:
        if not token:
            return None

        self._buffer += token
        self._deltas_since_flush += 1

        text = normalize_tts_text(self._buffer)
        hard_boundary = _find_hard_boundary(self._buffer)
        if hard_boundary is not None:
            return self._flush_prefix(hard_boundary, "sentence_boundary")

        if len(text) < MIN_CHUNK_CHARS:
            return None

        last = text[-1:]
        if last in SOFT_BOUNDARIES and len(text) >= MIN_SOFT_BOUNDARY_CHARS:
            return self._flush_all("clause_boundary")
        if self._deltas_since_flush >= self.token_target:
            return self._flush_at_last_word_boundary()
        return None

    def finish(self) -> ClauseChunk | None:
        if not normalize_tts_text(self._buffer):
            return None
        return self._flush_all("final")

    def _flush_all(self, reason: str) -> ClauseChunk:
        text = _finalize_tts_clause(normalize_tts_text(self._buffer))
        self._buffer = ""
        self._deltas_since_flush = 0
        return ClauseChunk(text=text, reason=reason)

    def _flush_prefix(self, raw_end: int, reason: str) -> ClauseChunk:
        text = _finalize_tts_clause(normalize_tts_text(self._buffer[:raw_end]))
        self._buffer = self._buffer[raw_end:].lstrip()
        self._deltas_since_flush = 0
        return ClauseChunk(text=text, reason=reason)

    def _flush_at_last_word_boundary(self) -> ClauseChunk | None:
        raw = self._buffer.rstrip()
        if not raw:
            return None

        boundary: re.Match[str] | None = None
        for match in re.finditer(r"\s+", raw):
            prefix = normalize_tts_text(raw[: match.start()])
            suffix = normalize_tts_text(raw[match.end() :])
            if _can_flush_token_target(prefix) and suffix:
                boundary = match

        if boundary is None:
            return None

        prefix = _finalize_tts_clause(normalize_tts_text(raw[: boundary.start()]))
        self._buffer = raw[boundary.end() :]
        self._deltas_since_flush = 0
        return ClauseChunk(text=prefix, reason="token_target")


def _finalize_tts_clause(text: str) -> str:
    if text.endswith((",", ";", ":")):
        return text[:-1].rstrip() + "."
    if text and not (text.endswith("...") or text.endswith(tuple(HARD_BOUNDARIES))):
        return text + "."
    return text


def _find_hard_boundary(raw: str) -> int | None:
    for match in re.finditer(r"\.\.\.|[.!?…]", raw):
        prefix = normalize_tts_text(raw[: match.end()])
        if len(prefix) >= MIN_HARD_BOUNDARY_CHARS:
            return match.end()
    return None


def _can_flush_token_target(text: str) -> bool:
    if len(text) < MIN_TOKEN_TARGET_CHARS:
        return False
    words = text.split()
    if len(words) < MIN_TOKEN_TARGET_WORDS:
        return False
    last = words[-1].strip("«»\"'()[]{}.,!?;:…").lower()
    if last in INCOMPLETE_TOKEN_TARGET_TAILS:
        return False
    return True
