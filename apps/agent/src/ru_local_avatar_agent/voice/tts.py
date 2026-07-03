"""Qwen3-TTS clause-level synthesis.

The 12 Hz CustomVoice model synthesizes faster than realtime on the RTX 5090,
so clause-sized requests keep the audio queue ahead of playback while leaving
barge-in cancellation points between clauses. A single worker thread owns the
model; results for stale generations are dropped by the caller.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SynthesizedClause:
    pcm: np.ndarray  # float32 mono, model sample rate
    sample_rate: int
    text: str


def estimate_max_new_tokens(
    text: str,
    *,
    min_new_tokens: int = 18,
    max_new_tokens: int = 384,
    chars_per_token: float = 1.35,
    padding_tokens: int = 6,
) -> int:
    if min_new_tokens < 1:
        raise ValueError("min_new_tokens must be >= 1")
    if max_new_tokens < min_new_tokens:
        raise ValueError("max_new_tokens must be >= min_new_tokens")
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be > 0")
    if padding_tokens < 0:
        raise ValueError("padding_tokens must be >= 0")

    chars = len(text.strip())
    estimated = math.ceil(chars * chars_per_token + padding_tokens)
    return min(max(estimated, min_new_tokens), max_new_tokens)


class QwenTtsEngine:
    def __init__(
        self,
        model_dir: Path,
        *,
        speaker: str = "Serena",
        language: str = "Russian",
        device: str = "cuda:0",
        fast_code_predictor: bool = False,
        min_new_tokens: int = 18,
        max_new_tokens_limit: int = 384,
        chars_per_token: float = 1.35,
        padding_tokens: int = 6,
        generation_kwargs: dict | None = None,
    ) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        self._min_new_tokens = min_new_tokens
        self._max_new_tokens_limit = max_new_tokens_limit
        self._chars_per_token = chars_per_token
        self._padding_tokens = padding_tokens
        self._generation_kwargs = generation_kwargs or {}
        self._model = Qwen3TTSModel.from_pretrained(
            str(model_dir),
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.fast_code_predictor_enabled = False
        if fast_code_predictor:
            from ru_local_avatar_agent.voice.fast_tts import install_fast_code_predictor

            install_fast_code_predictor(self._model)
            self.fast_code_predictor_enabled = True
        self.speaker = speaker
        self.language = language
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-tts")
        self._lock = threading.Lock()

    def synthesize_blocking(self, text: str, *, instruct: str | None = None) -> SynthesizedClause:
        text = text.strip()
        if not text:
            raise ValueError("empty TTS text")
        # Qwen3-TTS-12Hz treats this as the upper budget for generated audio
        # codes. A large floor makes even short clauses synthesize many seconds
        # of audio before the first packet can be pushed.
        max_new_tokens = estimate_max_new_tokens(
            text,
            min_new_tokens=self._min_new_tokens,
            max_new_tokens=self._max_new_tokens_limit,
            chars_per_token=self._chars_per_token,
            padding_tokens=self._padding_tokens,
        )
        kwargs: dict = {
            "text": text,
            "language": self.language,
            "speaker": self.speaker,
            "max_new_tokens": max_new_tokens,
        }
        kwargs.update(self._generation_kwargs)
        if instruct:
            kwargs["instruct"] = instruct
        started = time.monotonic()
        with self._lock:
            wavs, sample_rate = self._model.generate_custom_voice(**kwargs)
        wall_ms = (time.monotonic() - started) * 1000
        pcm = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        audio_ms = len(pcm) * 1000 / int(sample_rate)
        logger.info(
            (
                "tts synthesized chars=%d max_new_tokens=%d audio_ms=%.0f "
                "wall_ms=%.0f rtf=%.2f fast=%s"
            ),
            len(text),
            max_new_tokens,
            audio_ms,
            wall_ms,
            wall_ms / max(audio_ms, 1.0),
            self.fast_code_predictor_enabled,
        )
        return SynthesizedClause(pcm=pcm, sample_rate=int(sample_rate), text=text)

    async def synthesize(self, text: str, *, instruct: str | None = None) -> SynthesizedClause:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, lambda: self.synthesize_blocking(text, instruct=instruct)
        )
