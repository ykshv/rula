"""Silero VAD wrapper.

Runs on CPU so the GPU stays free for STT/TTS/A2F. Consumes 512-sample
(32 ms) frames of 16 kHz mono float32 audio and yields speech probabilities.
"""

from __future__ import annotations

import numpy as np

VAD_FRAME_SAMPLES = 512
VAD_SAMPLE_RATE = 16_000
VAD_FRAME_MS = 32


class SileroVad:
    def __init__(self) -> None:
        import torch
        from silero_vad import load_silero_vad

        self._torch = torch
        self._model = load_silero_vad()
        self._model.eval()

    def reset(self) -> None:
        self._model.reset_states()

    def probability(self, frame: np.ndarray) -> float:
        """Speech probability for one 512-sample float32 frame."""
        if frame.shape[0] != VAD_FRAME_SAMPLES:
            raise ValueError(f"VAD frame must be {VAD_FRAME_SAMPLES} samples, got {frame.shape[0]}")
        tensor = self._torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
        with self._torch.inference_mode():
            return float(self._model(tensor, VAD_SAMPLE_RATE).item())
