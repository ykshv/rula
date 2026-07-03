from __future__ import annotations

import re
from difflib import SequenceMatcher

_CRITICAL_TOKENS = {
    "не",
    "нет",
    "ни",
    "нельзя",
    "можно",
    "хочу",
    "надо",
    "нужно",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\wёЁ]+", text.lower(), flags=re.UNICODE)


def _compact(text: str) -> str:
    return "".join(_tokens(text))


def speculative_transcript_matches(speculative: str, final: str) -> bool:
    """Return true when final STT is only a minor correction of speculative text.

    Speculative generation should survive harmless ASR morphology changes
    ("девка" -> "девку"), otherwise we throw away already synthesized first audio
    and miss the latency target. Short or semantically risky turns stay strict.
    """
    speculative_norm = _compact(speculative)
    final_norm = _compact(final)
    if not speculative_norm or not final_norm:
        return False
    if speculative_norm == final_norm:
        return True

    speculative_tokens = set(_tokens(speculative))
    final_tokens = set(_tokens(final))
    if speculative_tokens & _CRITICAL_TOKENS != final_tokens & _CRITICAL_TOKENS:
        return False

    max_len = max(len(speculative_norm), len(final_norm))
    if max_len < 8:
        return False

    len_delta = abs(len(speculative_norm) - len(final_norm))
    if (
        speculative_norm.startswith(final_norm) or final_norm.startswith(speculative_norm)
    ) and len_delta <= max(6, int(max_len * 0.20)):
        return True

    return SequenceMatcher(None, speculative_norm, final_norm).ratio() >= 0.92
