"""Small PCM helpers shared by the voice pipeline."""

from __future__ import annotations

import numpy as np


def int16_to_float32(pcm: np.ndarray) -> np.ndarray:
    return (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def float32_to_int16(pcm: np.ndarray) -> np.ndarray:
    return (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16)


def resample(pcm: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample mono float32 PCM on CPU via torchaudio's polyphase kernel."""
    if source_rate == target_rate:
        return pcm
    import torch
    import torchaudio.functional as taf

    tensor = torch.from_numpy(np.ascontiguousarray(pcm, dtype=np.float32)).unsqueeze(0)
    out = taf.resample(tensor, source_rate, target_rate)
    return out.squeeze(0).numpy()
