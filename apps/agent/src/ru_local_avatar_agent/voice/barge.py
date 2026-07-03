from __future__ import annotations

import re
from dataclasses import dataclass

EMERGENCY_INTERRUPT_WORDS = {
    "стоп",
    "стой",
    "хватит",
    "замолчи",
    "перестань",
    "тише",
}


@dataclass(frozen=True, slots=True)
class BargeInVerification:
    confirmed: bool
    reason: str
    transcript: str


def verify_barge_in_transcript(
    transcript: str,
    active_assistant_text: str,
) -> BargeInVerification:
    """Reject barge-in candidates that are only playback echo.

    VAD alone cannot distinguish the user's real interruption from the avatar's
    own speaker audio leaking into the microphone. A short STT confirmation is
    slower, but it prevents cutting phrases like "Хорошо, не буду" after the
    first word.
    """
    normalized_transcript = _normalize(transcript)
    if not normalized_transcript:
        return BargeInVerification(False, "empty_transcript", transcript)

    transcript_tokens = set(normalized_transcript.split())
    if transcript_tokens & EMERGENCY_INTERRUPT_WORDS:
        return BargeInVerification(True, "emergency_command", transcript)

    normalized_assistant = _normalize(active_assistant_text)
    if not normalized_assistant:
        return BargeInVerification(True, "no_active_assistant_text", transcript)

    compact_transcript = normalized_transcript.replace(" ", "")
    compact_assistant = normalized_assistant.replace(" ", "")
    if compact_transcript and compact_transcript in compact_assistant:
        return BargeInVerification(False, "assistant_echo_substring", transcript)

    assistant_tokens = set(normalized_assistant.split())
    overlap = transcript_tokens & assistant_tokens
    if transcript_tokens and len(overlap) / len(transcript_tokens) >= 0.60:
        return BargeInVerification(False, "assistant_echo_overlap", transcript)

    return BargeInVerification(True, "new_user_speech", transcript)


def _normalize(text: str) -> str:
    normalized = text.casefold().replace("ё", "е")
    tokens = re.findall(r"[0-9a-zа-я]+", normalized, flags=re.UNICODE)
    return " ".join(tokens)
