from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AcceptanceThresholds:
    first_audio_p50_min_ms: int = 600
    first_audio_p50_max_ms: int = 700
    first_audio_p95_max_ms: int = 1100
    first_audio_release_failure_ms: int = 1500
    avatar_visible_reaction_max_ms: int = 250
    barge_in_p95_max_ms: int = 300
    russian_asr_wer_max: float = 0.08
    speculative_hit_rate_min: float = 0.70
    audio_face_pts_drift_p95_max_ms: int = 50

    def validate_latency(self, *, first_audio_p95_ms: int, avatar_reaction_ms: int) -> list[str]:
        failures: list[str] = []
        if first_audio_p95_ms > self.first_audio_p95_max_ms:
            failures.append("first_audio_p95_ms")
        if first_audio_p95_ms > self.first_audio_release_failure_ms:
            failures.append("first_audio_release_failure_ms")
        if avatar_reaction_ms > self.avatar_visible_reaction_max_ms:
            failures.append("avatar_visible_reaction_ms")
        return failures
