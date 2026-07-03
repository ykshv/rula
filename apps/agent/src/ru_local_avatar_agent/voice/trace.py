"""Per-turn trace used for latency diagnosis and UI timelines."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TurnTrace:
    session_id: str
    turn_id: int
    generation_id: int
    speech_end_wall: float | None = None
    marks: dict[str, float] = field(default_factory=dict)
    latency_tier: str = "unknown"
    playback_policy: str = "unknown"
    tts_cache_hits: int = 0
    tts_cache_misses: int = 0
    tts_unit_count: int = 0
    underruns: int = 0
    stale_drops: int = 0

    def mark(self, name: str, when: float | None = None) -> None:
        self.marks[name] = when if when is not None else time.monotonic()

    def observe_tts_unit(self, *, cache_hit: bool) -> None:
        self.tts_unit_count += 1
        if cache_hit:
            self.tts_cache_hits += 1
        else:
            self.tts_cache_misses += 1

    def summary(self) -> dict[str, Any]:
        baseline = self.speech_end_wall or self.marks.get("eot_commit")

        def delta(name: str) -> int | None:
            if baseline is None or name not in self.marks:
                return None
            return max(0, round((self.marks[name] - baseline) * 1000))

        def phase_delta(start: str, end: str) -> int | None:
            if start not in self.marks or end not in self.marks:
                return None
            return max(0, round((self.marks[end] - self.marks[start]) * 1000))

        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "generation_id": self.generation_id,
            "latency_tier": self.latency_tier,
            "playback_policy": self.playback_policy,
            "from_speech_end_ms": {
                "speculation_start": delta("speculation_start"),
                "eot_commit": delta("eot_commit"),
                "llm_first_token": delta("llm_first_token"),
                "tts_first_chunk": delta("tts_first_chunk"),
                "first_audio": delta("first_audio"),
                "audio_done": delta("audio_done"),
            },
            "phase_ms": {
                "commit_to_first_audio": phase_delta("eot_commit", "first_audio"),
                "llm_to_first_audio": phase_delta("llm_first_token", "first_audio"),
                "tts_to_first_audio": phase_delta("tts_first_chunk", "first_audio"),
                "audio_playout": phase_delta("first_audio", "audio_done"),
            },
            "tts": {
                "unit_count": self.tts_unit_count,
                "cache_hits": self.tts_cache_hits,
                "cache_misses": self.tts_cache_misses,
            },
            "reliability": {
                "underruns": self.underruns,
                "stale_drops": self.stale_drops,
            },
        }
