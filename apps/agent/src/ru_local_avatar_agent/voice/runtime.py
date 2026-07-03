"""VoiceRuntime: owns the GPU engines and per-session workers.

Engines load once at process start (in a background thread); readiness is
fail-closed — the voice pipeline reports ready only when every component is
loaded and LiveKit credentials/server are present.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from ru_local_avatar_agent.domain.session import SessionStateMachine
from ru_local_avatar_agent.runtime.config import RuntimeConfig
from ru_local_avatar_agent.runtime.vllm_client import VllmDialogueModel
from ru_local_avatar_agent.voice.audit import SQLiteConversationAuditStore
from ru_local_avatar_agent.voice.tts import estimate_max_new_tokens
from ru_local_avatar_agent.voice.tts_cache import TtsUnitCache, normalize_tts_cache_text
from ru_local_avatar_agent.voice.turn import TurnTuning

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EngineStatus:
    loaded: bool = False
    detail: str = "not_loaded"
    load_seconds: float = 0.0


@dataclass(slots=True)
class VoiceStatus:
    enabled: bool
    stt: EngineStatus = field(default_factory=EngineStatus)
    tts: EngineStatus = field(default_factory=EngineStatus)
    face: EngineStatus = field(default_factory=EngineStatus)
    vad: EngineStatus = field(default_factory=EngineStatus)
    error: str = ""

    @property
    def ready(self) -> bool:
        return (
            self.enabled
            and self.stt.loaded
            and self.tts.loaded
            and self.face.loaded
            and self.vad.loaded
        )


@dataclass(frozen=True, slots=True)
class PlaybackSettings:
    latency_tier: str
    playback_policy: str
    cache_hit: bool
    prebuffer_ms: int
    rebuffer_ms: int
    unit_start_min_ms: int
    start_after_response: bool


class VoiceRuntime:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.status = VoiceStatus(enabled=True)
        self.turn_tuning = _turn_tuning_from_profile(config.profile)

        profile_tts = config.profile.get("tts", {})
        profile_stt = config.profile.get("stt", {})
        profile_face = config.profile.get("face", {})
        profile_llm = config.profile.get("llm", {})
        self.llm_max_tokens = int(profile_llm.get("max_tokens", 64))
        self.llm_temperature = float(profile_llm.get("temperature", 0.6))
        self.llm_presence_penalty = float(profile_llm.get("presence_penalty", 0.0))
        self.first_chunk_token_target = int(
            profile_tts.get("first_chunk", {}).get("token_target", 8)
        )
        self.later_chunk_token_target = int(
            profile_tts.get("later_chunks", {}).get("token_target", 18)
        )
        self.tts_fast_code_predictor_enabled = bool(
            profile_tts.get("fast_code_predictor", {}).get("enabled", False)
        )
        import os

        self.tts_compile_enabled = (
            os.getenv("RULA_TTS_COMPILE", "1") == "1"
            and bool(profile_tts.get("streaming", {}).get("compile", True))
        )
        self.tts_unit_timeout_ms = int(
            profile_tts.get("streaming", {}).get("unit_timeout_ms", 8_000)
        )
        self.tts_chunk_timeout_ms = int(
            profile_tts.get("streaming", {}).get("chunk_timeout_ms", 3_500)
        )
        playback_config = profile_tts.get("playback", {})
        self.tts_playback_start_mode = str(playback_config.get("start_mode", "streaming"))
        self.tts_playback_start_after_response = (
            self.tts_playback_start_mode == "after_response"
        )
        self.tts_playback_prebuffer_ms = int(playback_config.get("prebuffer_ms", 220))
        self.tts_playback_rebuffer_ms = int(playback_config.get("rebuffer_ms", 140))
        self.tts_playback_unit_start_min_ms = int(
            playback_config.get("unit_start_min_ms", 600)
        )
        self.tts_cached_playback_prebuffer_ms = int(
            playback_config.get("cached_prebuffer_ms", 220)
        )
        self.tts_cached_playback_rebuffer_ms = int(
            playback_config.get("cached_rebuffer_ms", 180)
        )
        self.tts_cached_playback_unit_start_min_ms = int(
            playback_config.get("cached_unit_start_min_ms", 0)
        )
        self.tts_speculative_enabled = bool(
            profile_tts.get("speculative_tts", {}).get("enabled", False)
        )
        profile_tts_generation = profile_tts.get("generation", {})
        self.tts_min_new_tokens = int(profile_tts_generation.get("min_new_tokens", 18))
        self.tts_max_new_tokens = int(profile_tts_generation.get("max_new_tokens", 384))
        self.tts_chars_per_token = float(profile_tts_generation.get("chars_per_token", 1.35))
        self.tts_padding_tokens = int(profile_tts_generation.get("padding_tokens", 6))
        self.tts_generation_config = dict(profile_tts_generation)
        self.tts_generation_kwargs = {
            key: value
            for key, value in profile_tts_generation.items()
            if key
            not in {
                "min_new_tokens",
                "max_new_tokens",
                "chars_per_token",
                "padding_tokens",
            }
        }
        tts_cache_config = profile_tts.get("cache", {})
        deterministic_tts = (
            not bool(self.tts_generation_kwargs.get("do_sample", False))
            and not bool(self.tts_generation_kwargs.get("subtalker_do_sample", False))
        )
        self.tts_cache_enabled = bool(tts_cache_config.get("enabled", deterministic_tts))
        self.tts_cache_max_units = int(tts_cache_config.get("max_units", 64))
        self.tts_cache_max_chars = int(tts_cache_config.get("max_chars", 96))
        self.tts_cache_persistent = bool(tts_cache_config.get("persistent", True))
        self.tts_cache_warmup_enabled = bool(tts_cache_config.get("warmup_direct", True))
        cache_path = tts_cache_config.get("path", "data/tts_cache")
        tts_cache_root = self.config.root / cache_path
        self.stt_device = str(profile_stt.get("device", "cuda"))
        self.face_device = str(profile_face.get("device", "cuda"))
        self.livekit_ws_url = config.livekit_ws_url

        self.stt = None
        self.tts = None
        self.face_engine = None
        self.conversation_audit = SQLiteConversationAuditStore(config.conversation_audit_path)
        self._vad_factory = None
        self._stt_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        self._face_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="a2f")
        self._tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        self._tts_cache: OrderedDict[str, list[Any]] = OrderedDict()
        self._tts_cache_lock = threading.Lock()
        self._tts_disk_cache = TtsUnitCache(
            root=tts_cache_root,
            voice=self.config.voice_preset or "Serena",
            generation_config=self.tts_generation_config,
            max_chars=self.tts_cache_max_chars,
            enabled=self.tts_cache_enabled and self.tts_cache_persistent,
        )
        self._workers: dict[str, Any] = {}
        self._worker_tasks: dict[str, asyncio.Task] = {}
        self._load_thread: threading.Thread | None = None
        self.gpu_headroom_after_load: dict[str, Any] | None = None

    # --------------------------------------------------------------- loading

    def start_loading(self) -> None:
        if self._load_thread is not None:
            return
        self._load_thread = threading.Thread(target=self._load_all, name="voice-load", daemon=True)
        self._load_thread.start()

    def _load_all(self) -> None:
        models_root = self.config.root / "models" / "hf"
        try:
            started = time.monotonic()
            from ru_local_avatar_agent.voice.vad import SileroVad

            self._vad_factory = SileroVad
            probe = SileroVad()
            probe.probability(np.zeros(512, dtype=np.float32))
            self.status.vad = EngineStatus(True, "silero", time.monotonic() - started)
            logger.info("VAD loaded in %.1fs", self.status.vad.load_seconds)

            started = time.monotonic()
            from ru_local_avatar_agent.voice.stt import GigaAMTranscriber

            self.stt = GigaAMTranscriber(models_root / "ai-sage__GigaAM-v3", device=self.stt_device)
            self.stt.transcribe(np.zeros(16_000, dtype=np.float32))  # warmup
            self.status.stt = EngineStatus(
                True, f"gigaam-v3-e2e-rnnt ({self.stt_device})", time.monotonic() - started
            )
            logger.info("STT loaded in %.1fs", self.status.stt.load_seconds)

            started = time.monotonic()
            from ru_local_avatar_agent.voice.face import A2FEngine, A2FTurnStream

            self.face_engine = A2FEngine(
                models_root / "nvidia__Audio2Face-3D-v3.0",
                prefer_gpu=self.face_device == "cuda",
            )
            warm = A2FTurnStream(self.face_engine)
            warm.push_audio(np.zeros(16_000, dtype=np.float32))
            self.status.face = EngineStatus(
                True, f"a2f-3d-v3 ({self.face_engine.provider})", time.monotonic() - started
            )
            logger.info("A2F loaded in %.1fs", self.status.face.load_seconds)

            started = time.monotonic()
            from ru_local_avatar_agent.voice.streaming_tts import StreamingQwenTts

            tts_dir = models_root / "Qwen__Qwen3-TTS-12Hz-1.7B-CustomVoice"
            speaker = self.config.voice_preset if self.config.voice_preset else "Serena"
            self.tts = StreamingQwenTts(
                tts_dir,
                speaker=speaker.capitalize(),
                compile_steps=self.tts_compile_enabled,
                generation_kwargs=self.tts_generation_config,
            )
            # Two passes so torch.compile captures CUDA graphs for every
            # step shape before the first real turn.
            self.tts.warmup()
            self.tts.warmup()
            self._tts_sample_rate = self.tts.sample_rate
            sample_mode = "sampled" if self.tts.generation_settings.do_sample else "greedy"
            self.status.tts = EngineStatus(
                True,
                f"qwen3-tts-streaming {speaker} @{self.tts.sample_rate}Hz"
                f" (compiled={self.tts._compiled}, mode={sample_mode})",
                time.monotonic() - started,
            )
            logger.info("TTS loaded in %.1fs", self.status.tts.load_seconds)
            self._warm_tts_direct_responses()

            self._warm_llm()

            # Snapshot the post-load headroom once: this is the profile's
            # actual invariant. Measured with torch.cuda.mem_get_info from
            # INSIDE this process: under WSL2 the global nvidia-smi number
            # includes evictable Windows desktop allocations (browser, DWM)
            # and would flap readiness whenever the user opens a tab; what
            # bounds our OOM risk is memory available to the CUDA context.
            import torch

            free_bytes, total_bytes = torch.cuda.mem_get_info()
            min_free_gb = float(
                self.config.profile.get("runtime", {}).get("min_free_vram_gb_after_load", 1.5)
            )
            free_gb = round(free_bytes / 1024**3, 3)
            self.gpu_headroom_after_load = {
                "ok": free_gb >= min_free_gb,
                "detail": "headroom_ok" if free_gb >= min_free_gb else "insufficient_vram_headroom",
                "free_gb": free_gb,
                "total_gb": round(total_bytes / 1024**3, 3),
                "min_free_gb": min_free_gb,
                "source": "torch.cuda.mem_get_info(after_load)",
            }
            logger.info(
                "GPU headroom after load: %s (cuda-free %.2f GB, min %.2f GB)",
                self.gpu_headroom_after_load["detail"],
                free_gb,
                min_free_gb,
            )
        except Exception as exc:  # fail closed, report honestly
            logger.exception("voice runtime failed to load")
            self.status.error = f"{type(exc).__name__}: {exc}"

    def _warm_llm(self) -> None:
        """Best-effort vLLM warmup so turn 0 does not pay prefix-cache misses."""
        import httpx

        from ru_local_avatar_agent.voice.llm import build_messages

        try:
            started = time.monotonic()
            response = httpx.post(
                f"{self.config.vllm_base_url}/chat/completions",
                json={
                    "model": self.config.llm_model,
                    "messages": build_messages([], "Привет!"),
                    "max_tokens": 8,
                    "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=30.0,
            )
            response.raise_for_status()
            logger.info("LLM warmup ok in %.1fs", time.monotonic() - started)
        except Exception as exc:
            logger.warning("LLM warmup skipped: %s", exc)

    @property
    def tts_sample_rate(self) -> int:
        return getattr(self, "_tts_sample_rate", 24_000)

    @property
    def dialogue_model(self) -> VllmDialogueModel:
        return VllmDialogueModel(
            base_url=self.config.vllm_base_url,
            model=self.config.llm_model,
            max_tokens=self.llm_max_tokens,
            temperature=self.llm_temperature,
            presence_penalty=self.llm_presence_penalty,
        )

    def create_vad(self):
        if self._vad_factory is None:
            raise RuntimeError("voice runtime not loaded")
        return self._vad_factory()

    async def run_stt(self, pcm_16k: np.ndarray) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._stt_executor, self.stt.transcribe, pcm_16k)

    def _tts_cache_key(self, text: str) -> str:
        voice = self.config.voice_preset or "Serena"
        payload = {
            "voice": voice,
            "generation_config": self.tts_generation_config,
            "text": normalize_tts_cache_text(text),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _get_tts_cache(self, text: str) -> list[Any] | None:
        if not self.tts_cache_enabled:
            return None
        key = self._tts_cache_key(text)
        with self._tts_cache_lock:
            cached = self._tts_cache.get(key)
            if cached is not None:
                self._tts_cache.move_to_end(key)
                return list(cached)
        disk_cached = self._tts_disk_cache.get(text)
        if disk_cached is not None:
            from ru_local_avatar_agent.voice import metrics

            metrics.TTS_CACHE_EVENTS.labels(event="disk_hit").inc()
            with self._tts_cache_lock:
                self._tts_cache[key] = list(disk_cached)
                self._tts_cache.move_to_end(key)
                while len(self._tts_cache) > self.tts_cache_max_units:
                    self._tts_cache.popitem(last=False)
            return list(disk_cached)
        return None

    def _put_tts_cache(self, text: str, chunks: list[Any]) -> None:
        if (
            not self.tts_cache_enabled
            or not chunks
            or len(text.strip()) > self.tts_cache_max_chars
        ):
            return
        key = self._tts_cache_key(text)
        with self._tts_cache_lock:
            self._tts_cache[key] = list(chunks)
            self._tts_cache.move_to_end(key)
            while len(self._tts_cache) > self.tts_cache_max_units:
                self._tts_cache.popitem(last=False)
        self._tts_disk_cache.put(text, chunks)

    def is_tts_cached(self, text: str) -> bool:
        if not self.tts_cache_enabled:
            return False
        key = self._tts_cache_key(text)
        with self._tts_cache_lock:
            if key in self._tts_cache:
                return True
        return self._tts_disk_cache.exists(text)

    def playback_settings_for_text(
        self,
        text: str,
        *,
        planned_latency_tier: str,
        planned_playback_policy: str,
    ) -> PlaybackSettings:
        cache_hit = bool(text and self.is_tts_cached(text))
        cached_policy = cache_hit and planned_playback_policy == "cached"
        if cached_policy:
            return PlaybackSettings(
                latency_tier=planned_latency_tier or "cached_tts",
                playback_policy="cached",
                cache_hit=True,
                prebuffer_ms=self.tts_cached_playback_prebuffer_ms,
                rebuffer_ms=self.tts_cached_playback_rebuffer_ms,
                unit_start_min_ms=self.tts_cached_playback_unit_start_min_ms,
                start_after_response=False,
            )
        return PlaybackSettings(
            latency_tier="qwen_tts",
            playback_policy="qwen_tts",
            cache_hit=False,
            prebuffer_ms=self.tts_playback_prebuffer_ms,
            rebuffer_ms=self.tts_playback_rebuffer_ms,
            unit_start_min_ms=self.tts_playback_unit_start_min_ms,
            start_after_response=self.tts_playback_start_after_response,
        )

    def _warm_tts_direct_responses(self) -> None:
        if not self.tts_cache_enabled or not self.tts_cache_warmup_enabled or self.tts is None:
            return
        from ru_local_avatar_agent.voice.brain import DIRECT_TTS_WARMUP_TEXTS

        for text in DIRECT_TTS_WARMUP_TEXTS:
            if self.is_tts_cached(text):
                continue
            started = time.monotonic()
            try:
                max_frames = estimate_max_new_tokens(
                    text,
                    min_new_tokens=self.tts_min_new_tokens,
                    max_new_tokens=self.tts_max_new_tokens,
                    chars_per_token=self.tts_chars_per_token,
                    padding_tokens=self.tts_padding_tokens,
                )
                chunks = list(self.tts.stream(text, max_frames=max_frames))
                self._put_tts_cache(text, chunks)
                logger.info(
                    "TTS direct warm cache text=%r chunks=%d wall_ms=%.0f",
                    text[:80],
                    len(chunks),
                    (time.monotonic() - started) * 1000,
                )
            except Exception:
                logger.warning("TTS direct warm cache failed text=%r", text[:80], exc_info=True)

    def stream_tts(self, text: str):
        """Async iterator over TtsChunk, produced on the dedicated TTS thread.

        Closing the iterator (or cancelling the consuming task) flips a stop
        flag that the synthesis loop polls every codec frame (~30 ms), so
        barge-in aborts synthesis almost immediately.
        """
        import threading

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        stop = threading.Event()
        SENTINEL = object()

        def producer() -> None:
            try:
                cached = self._get_tts_cache(text)
                if cached is not None:
                    from ru_local_avatar_agent.voice import metrics

                    metrics.TTS_CACHE_EVENTS.labels(event="hit").inc()
                    logger.info("tts cache hit chars=%d chunks=%d", len(text), len(cached))
                    for chunk in cached:
                        if stop.is_set():
                            return
                        while not stop.is_set():
                            try:
                                asyncio.run_coroutine_threadsafe(
                                    queue.put(chunk), loop
                                ).result(timeout=1.0)
                                break
                            except TimeoutError:
                                continue
                            except Exception:
                                return
                    return
                from ru_local_avatar_agent.voice import metrics

                metrics.TTS_CACHE_EVENTS.labels(event="miss").inc()

                max_frames = estimate_max_new_tokens(
                    text,
                    min_new_tokens=self.tts_min_new_tokens,
                    max_new_tokens=self.tts_max_new_tokens,
                    chars_per_token=self.tts_chars_per_token,
                    padding_tokens=self.tts_padding_tokens,
                )
                produced: list[Any] = []
                for chunk in self.tts.stream(
                    text,
                    max_frames=max_frames,
                    should_stop=stop.is_set,
                ):
                    if stop.is_set():
                        break
                    produced.append(chunk)
                    # Block the TTS thread if the consumer is behind; the
                    # queue is small so cancellation stays responsive.
                    while not stop.is_set():
                        try:
                            asyncio.run_coroutine_threadsafe(
                                queue.put(chunk), loop
                            ).result(timeout=1.0)
                            break
                        except TimeoutError:
                            continue
                        except Exception:
                            return
                if not stop.is_set():
                    self._put_tts_cache(text, produced)
                    if produced:
                        metrics.TTS_CACHE_EVENTS.labels(event="write").inc()
            except Exception as exc:  # surfaced to the consumer
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop)
                return
            finally:
                with contextlib.suppress(Exception):
                    asyncio.run_coroutine_threadsafe(queue.put(SENTINEL), loop)

        future = self._tts_executor.submit(producer)

        class _Stream:
            def __aiter__(self):
                return self

            async def __anext__(_self):
                item = await queue.get()
                if item is SENTINEL:
                    raise StopAsyncIteration
                if isinstance(item, Exception):
                    raise item
                return item

            async def aclose(_self, timeout_s: float = 0.5) -> None:
                stop.set()
                # Drain so the producer is never stuck on queue.put.
                deadline = time.monotonic() + max(0.0, timeout_s)
                while not future.done() and time.monotonic() < deadline:
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0.01)
                if not future.done():
                    future.cancel()

        return _Stream()

    async def run_face(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._face_executor, fn, *args)

    # --------------------------------------------------------------- tokens

    def mint_token(
        self,
        *,
        identity: str,
        name: str,
        room_name: str,
        ttl_seconds: int = 3600,
    ) -> str:
        from datetime import timedelta

        from livekit import api

        return (
            api.AccessToken(self.config.livekit_api_key, self.config.livekit_api_secret)
            .with_identity(identity)
            .with_name(name)
            .with_ttl(timedelta(seconds=ttl_seconds))
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
            )
            .to_jwt()
        )

    # --------------------------------------------------------------- workers

    def start_worker(
        self,
        session: SessionStateMachine,
        room_name: str,
        *,
        on_stop: Callable[[str], None] | None = None,
        barge_in_enabled: bool = True,
    ) -> bool:
        from ru_local_avatar_agent.voice.worker import VoiceSessionWorker

        if session.session_id in self._workers:
            return False
        worker = VoiceSessionWorker(
            self,
            session=session,
            room_name=room_name,
            barge_in_enabled=barge_in_enabled,
        )
        self._workers[session.session_id] = worker

        async def _run() -> None:
            try:
                await worker.run()
            except Exception:
                logger.exception("voice worker crashed for %s", room_name)
            finally:
                self._workers.pop(session.session_id, None)
                self._worker_tasks.pop(session.session_id, None)
                if on_stop is not None:
                    on_stop(session.session_id)

        self._worker_tasks[session.session_id] = asyncio.create_task(_run())
        return True

    def get_worker(self, session_id: str):
        return self._workers.get(session_id)

    async def stop_worker(self, session_id: str, *, timeout_seconds: float = 5.0) -> bool:
        worker = self._workers.get(session_id)
        task = self._worker_tasks.get(session_id)
        if worker is None and task is None:
            return False

        if worker is not None:
            worker.close()

        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=timeout_seconds)
            except TimeoutError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        self._workers.pop(session_id, None)
        self._worker_tasks.pop(session_id, None)
        return True

    async def shutdown(self) -> None:
        for worker in list(self._workers.values()):
            worker.close()
        for task in list(self._worker_tasks.values()):
            task.cancel()

    # --------------------------------------------------------------- status

    def snapshot(self) -> dict[str, Any]:
        return {
            "ok": self.status.ready,
            "detail": self.status.error
            or (
                "loaded" if self.status.ready else "voice engines loading or unavailable"
            ),
            "engines": {
                "vad": asdict(self.status.vad),
                "stt": asdict(self.status.stt),
                "tts": asdict(self.status.tts),
                "face": asdict(self.status.face),
            },
            "active_sessions": len(self._workers),
        }


def _turn_tuning_from_profile(profile: dict[str, Any]) -> TurnTuning:
    defaults = TurnTuning()
    reactive = profile.get("turn_detection", {}).get("reactive", {})
    return replace(
        defaults,
        voiced_on_probability=float(
            reactive.get("voiced_on_probability", defaults.voiced_on_probability)
        ),
        voiced_off_probability=float(
            reactive.get("voiced_off_probability", defaults.voiced_off_probability)
        ),
        speculative_silence_ms=int(
            reactive.get("speculative_silence_ms", defaults.speculative_silence_ms)
        ),
        commit_silence_ms=int(reactive.get("commit_silence_ms", defaults.commit_silence_ms)),
        punctuated_commit_silence_ms=int(
            reactive.get(
                "punctuated_commit_silence_ms",
                defaults.punctuated_commit_silence_ms,
            )
        ),
        incomplete_commit_silence_ms=int(
            reactive.get(
                "incomplete_commit_silence_ms",
                defaults.incomplete_commit_silence_ms,
            )
        ),
        punctuated_commit_min_words=int(
            reactive.get(
                "punctuated_commit_min_words",
                defaults.punctuated_commit_min_words,
            )
        ),
        barge_in_probability=float(
            reactive.get("barge_in_probability", defaults.barge_in_probability)
        ),
        barge_in_voiced_ms=int(
            reactive.get("barge_in_voiced_ms", defaults.barge_in_voiced_ms)
        ),
        barge_in_min_speaking_ms=int(
            reactive.get(
                "barge_in_min_speaking_ms",
                defaults.barge_in_min_speaking_ms,
            )
        ),
    )
