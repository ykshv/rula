from __future__ import annotations

FACE_LEAD_MS = 400


def face_emit_horizon_ms(*, pushed_ms: float, flushing: bool) -> float:
    if flushing:
        return float("inf")
    return pushed_ms + FACE_LEAD_MS
