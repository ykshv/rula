"""Prometheus metrics for the realtime voice pipeline.

Histogram buckets are aligned with the runtime acceptance targets so
p50/p95 against targets can be read directly.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

LATENCY_BUCKETS_MS = (
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
    1100,
    1300,
    1500,
    2000,
    3000,
)

FIRST_AUDIO_MS = Histogram(
    "rula_first_audio_ms",
    "User speech end (VAD) to first response audio frame pushed to LiveKit, ms",
    buckets=LATENCY_BUCKETS_MS,
)
FIRST_AUDIO_BY_TIER_MS = Histogram(
    "rula_first_audio_by_tier_ms",
    "User speech end to first audio split by response latency tier, ms",
    ["latency_tier"],
    buckets=LATENCY_BUCKETS_MS,
)
EOT_COMMIT_MS = Histogram(
    "rula_eot_commit_ms",
    "User speech end (VAD) to committed EOT decision, ms",
    buckets=LATENCY_BUCKETS_MS,
)
BARGE_IN_MS = Histogram(
    "rula_barge_in_ms",
    "User voiced onset during agent speech to playback flush completed, ms",
    buckets=LATENCY_BUCKETS_MS,
)
LLM_FIRST_TOKEN_MS = Histogram(
    "rula_llm_first_token_ms",
    "LLM request start to first streamed token, ms",
    buckets=LATENCY_BUCKETS_MS,
)
TTS_FIRST_CHUNK_MS = Histogram(
    "rula_tts_first_chunk_ms",
    "First clause text ready to synthesized audio returned, ms",
    buckets=LATENCY_BUCKETS_MS,
)
TTS_UNIT_WALL_MS = Histogram(
    "rula_tts_unit_wall_ms",
    "One TTS unit synthesis wall time split by latency tier and cache state, ms",
    ["latency_tier", "cache_state"],
    buckets=LATENCY_BUCKETS_MS,
)
TTS_UNIT_AUDIO_MS = Histogram(
    "rula_tts_unit_audio_ms",
    "One TTS unit audio duration split by latency tier and cache state, ms",
    ["latency_tier", "cache_state"],
    buckets=LATENCY_BUCKETS_MS,
)
TTS_CACHE_EVENTS = Counter(
    "rula_tts_cache_events_total",
    "TTS cache hits, misses, writes, and disk reads",
    ["event"],
)
TTS_UNIT_TIMEOUTS = Counter(
    "rula_tts_unit_timeouts_total",
    "TTS units that exceeded their bounded synthesis deadline",
)
STT_PARTIAL_MS = Histogram(
    "rula_stt_partial_ms",
    "Partial transcription wall time, ms",
    buckets=LATENCY_BUCKETS_MS,
)
STT_FINAL_MS = Histogram(
    "rula_stt_final_ms",
    "Final transcription wall time, ms",
    buckets=LATENCY_BUCKETS_MS,
)
A2F_WINDOW_MS = Histogram(
    "rula_a2f_window_ms",
    "Audio2Face one-window inference wall time, ms",
    buckets=(5, 10, 20, 30, 50, 75, 100, 150, 250, 500),
)

SPECULATIVE_STARTS = Counter("rula_speculative_starts_total", "Speculative generations started")
SPECULATIVE_HITS = Counter("rula_speculative_hits_total", "Speculative generations kept at commit")
SPECULATIVE_DISCARDS = Counter(
    "rula_speculative_discards_total",
    "Speculative generations discarded (user resumed or mismatch)",
)
TURNS_COMPLETED = Counter("rula_turns_completed_total", "Voice turns answered to completion")
INTERRUPTS = Counter("rula_interrupts_total", "Barge-in interrupts executed")
BARGE_DUCKS = Counter(
    "rula_barge_ducks_total", "Playback ducks triggered by early barge-in detection"
)
BARGE_DUCK_ABORTS = Counter(
    "rula_barge_duck_aborts_total", "Ducks reverted because the voiced burst was not speech"
)
BARGE_CONFIRM_MS = Histogram(
    "rula_barge_confirm_ms",
    "User voiced onset to transcript-confirmed cancellation, ms",
    buckets=LATENCY_BUCKETS_MS,
)
BARGE_IN_REJECTS = Counter(
    "rula_barge_in_rejects_total",
    "Barge-in candidates rejected before playback cancellation",
    ["reason"],
)
PIPELINE_ERRORS = Counter(
    "rula_pipeline_errors_total",
    "Unhandled voice pipeline errors",
    ["stage"],
)
STALE_DROPS = Counter("rula_stale_drops_total", "Artifacts dropped due to stale generation_id")
AUDIO_UNDERRUNS = Counter(
    "rula_audio_underruns_total", "Playback pauses caused by TTS falling behind realtime"
)
AUDIO_PLAYOUT_TIMEOUTS = Counter(
    "rula_audio_playout_timeouts_total",
    "Audio playout waits that exceeded the bounded turn-release budget",
)

ACTIVE_SESSIONS = Gauge("rula_active_voice_sessions", "Voice session workers currently running")
