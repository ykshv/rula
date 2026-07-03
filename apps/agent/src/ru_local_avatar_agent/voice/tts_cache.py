"""Persistent cache for deterministic TTS chunks.

The cache stores only locally generated chunks. It deliberately avoids pickle:
metadata is JSON, audio arrays are stored as NumPy arrays in an `.npz` file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from ru_local_avatar_agent.voice.streaming_tts import TtsChunk

logger = logging.getLogger(__name__)


def normalize_tts_cache_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class TtsUnitCache:
    def __init__(
        self,
        *,
        root: Path,
        voice: str,
        generation_config: dict[str, Any],
        max_chars: int,
        enabled: bool = True,
    ) -> None:
        self.root = root
        self.voice = voice
        self.generation_config = dict(generation_config)
        self.max_chars = max_chars
        self.enabled = enabled
        self._lock = threading.Lock()
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def key(self, text: str) -> str:
        payload = {
            "voice": self.voice,
            "generation_config": self.generation_config,
            "text": normalize_tts_cache_text(text),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def path_for(self, text: str) -> Path:
        return self.root / f"{self.key(text)}.npz"

    def get(self, text: str) -> list[TtsChunk] | None:
        normalized = normalize_tts_cache_text(text)
        if not self._can_cache(normalized):
            return None
        path = self.path_for(normalized)
        if not path.exists():
            return None
        with self._lock:
            try:
                with np.load(path, allow_pickle=False) as data:
                    metadata = json.loads(str(data["metadata"].item()))
                    chunks = []
                    for index, item in enumerate(metadata["chunks"]):
                        chunks.append(
                            TtsChunk(
                                pcm=np.asarray(data[f"pcm_{index}"], dtype=np.float32),
                                sample_rate=int(item["sample_rate"]),
                                pts_ms=int(item["pts_ms"]),
                                is_final=bool(item["is_final"]),
                            )
                        )
                    return chunks
            except Exception:
                logger.warning("failed to read TTS cache %s", path, exc_info=True)
                with suppress(OSError):
                    path.unlink()
                return None

    def put(self, text: str, chunks: list[TtsChunk]) -> None:
        normalized = normalize_tts_cache_text(text)
        if not self._can_cache(normalized) or not chunks:
            return
        path = self.path_for(normalized)
        metadata = {
            "voice": self.voice,
            "text": normalized,
            "chunks": [
                {
                    "sample_rate": chunk.sample_rate,
                    "pts_ms": chunk.pts_ms,
                    "is_final": chunk.is_final,
                }
                for chunk in chunks
            ],
        }
        arrays: dict[str, Any] = {
            "metadata": np.array(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
        }
        arrays.update(
            {
                f"pcm_{index}": chunk.pcm.astype(np.float32)
                for index, chunk in enumerate(chunks)
            }
        )
        fd, tmp_name = tempfile.mkstemp(prefix=f"{path.stem}.", suffix=".tmp", dir=self.root)
        os.close(fd)
        tmp = Path(tmp_name)
        npz_tmp = tmp.with_suffix(tmp.suffix + ".npz")
        try:
            np.savez_compressed(npz_tmp, **arrays)
            npz_tmp.replace(path)
        finally:
            with suppress(OSError):
                tmp.unlink()
            with suppress(OSError):
                npz_tmp.unlink()

    def exists(self, text: str) -> bool:
        normalized = normalize_tts_cache_text(text)
        return self._can_cache(normalized) and self.path_for(normalized).exists()

    def _can_cache(self, text: str) -> bool:
        return self.enabled and bool(text) and len(text) <= self.max_chars
