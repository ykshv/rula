"""Pre-generate Russian user-turn audio for the E2E latency probe.

Runs inside the agent image while the agent is NOT holding the GPU TTS
(or with enough free VRAM). Writes 16 kHz mono WAVs plus a manifest.

  docker run --rm --gpus all --network host \
    -v <repo-root>/models/hf:/app/models/hf:ro \
    -v <repo-root>/scripts:/app/scripts:ro \
    -v <repo-root>/runtime:/app/runtime \
    --entrypoint python3 wsl-agent:latest /app/scripts/evals/make_probe_audio.py
"""

from __future__ import annotations

import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app/apps/agent/src")

OUT_DIR = Path(os.getenv("RULA_RUNTIME_DIR", "/runtime")) / "probe_audio"
PHRASES = [
    "Привет! Как у тебя дела сегодня?",
    "Расскажи мне что-нибудь интересное про космос.",
    "Какой твой любимый цвет и почему?",
    "Посоветуй, что приготовить на ужин.",
    "Сколько будет двадцать семь плюс пятнадцать?",
    "Ты умеешь рассказывать анекдоты?",
    "Объясни простыми словами, что такое нейросеть.",
    "Какая столица у Австралии?",
    "Мне сегодня немного грустно, подбодри меня.",
    "Назови три интересных факта о дельфинах.",
    "Что ты думаешь о современной музыке?",
    "Помоги придумать имя для котёнка.",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=len(PHRASES))
    args = parser.parse_args()

    import soundfile as sf

    from ru_local_avatar_agent.voice.audio import resample
    from ru_local_avatar_agent.voice.tts import QwenTtsEngine

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # A different premium voice than the avatar's, acting as the "user".
    tts = QwenTtsEngine(
        Path("/app/models/hf/Qwen__Qwen3-TTS-12Hz-1.7B-CustomVoice"), speaker="Vivian"
    )
    manifest = []
    for index, phrase in enumerate(PHRASES[: max(1, args.limit)]):
        clause = tts.synthesize_blocking(phrase)
        pcm16 = resample(clause.pcm, clause.sample_rate, 16_000)
        name = f"turn_{index:02d}.wav"
        sf.write(str(OUT_DIR / name), pcm16.astype(np.float32), 16_000)
        manifest.append({"file": name, "text": phrase, "seconds": len(pcm16) / 16_000})
        print(f"{name}: {len(pcm16)/16_000:.2f}s  {phrase}")
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(manifest)} turns to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
