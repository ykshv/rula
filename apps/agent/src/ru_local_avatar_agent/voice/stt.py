"""GigaAM-v3 e2e_rnnt speech recognition over in-memory PCM.

Bypasses the upstream file/ffmpeg path and feeds tensors directly, so partial
transcriptions of the live utterance buffer are cheap enough to run every few
hundred milliseconds on the RTX 5090.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

MAX_UTTERANCE_SECONDS = 29  # GigaAM shortform ceiling is ~30 s


class GigaAMTranscriber:
    def __init__(self, model_dir: Path, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModel

        self._torch = torch
        self._device = device
        model = AutoModel.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=True,
        )
        model.to(device)
        model.eval()
        # GigaAMModel wraps GigaAMASR as `.model`; we need the tensor path.
        self._asr = model.model
        self._keepalive = model
        self._lock = threading.Lock()

    def transcribe(self, pcm_16k: np.ndarray) -> str:
        """Transcribe mono float32 16 kHz PCM. Thread-safe, blocking."""
        if pcm_16k.size == 0:
            return ""
        max_samples = MAX_UTTERANCE_SECONDS * 16_000
        if pcm_16k.shape[0] > max_samples:
            pcm_16k = pcm_16k[-max_samples:]
        torch = self._torch
        with self._lock, torch.inference_mode():
            wav = torch.from_numpy(np.ascontiguousarray(pcm_16k, dtype=np.float32))
            wav = wav.to(self._device).unsqueeze(0)
            length = torch.full([1], wav.shape[-1], device=self._device)
            encoded, encoded_len = self._asr.forward(wav, length)
            texts = self._asr.decoding.decode(self._asr.head, encoded, encoded_len)
        return texts[0].strip()
