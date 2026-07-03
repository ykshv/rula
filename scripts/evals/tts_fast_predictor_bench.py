"""Benchmark the experimental Qwen3-TTS fast code-predictor path.

This script loads Qwen3-TTS once, measures the stock qwen-tts generation path,
installs the manual code-predictor loop, then measures the patched path.

Run it with the main agent stopped so the benchmark can own GPU memory:

  docker compose -f infra/wsl/docker-compose.yml stop agent
  docker compose -f infra/wsl/docker-compose.yml run --rm --entrypoint python3 agent /app/scripts/evals/tts_fast_predictor_bench.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/app/apps/agent/src")

from ru_local_avatar_agent.voice.fast_tts import install_fast_code_predictor


DEFAULT_MODEL_DIR = Path("/app/models/hf/Qwen__Qwen3-TTS-12Hz-1.7B-CustomVoice")
DEFAULT_TEXTS = [
    "Привет! Я локальный аватар и отвечаю по-русски.",
    "Мне нравится разговаривать с тобой быстро и естественно.",
    "Сейчас проверим, насколько ускорился первый речевой кусок.",
]


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(model_dir: Path):
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(
        str(model_dir),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )


def synthesize_once(model, text: str, *, speaker: str, language: str) -> dict[str, float]:
    input_ids = model._tokenize_texts([model._build_assistant_text(text)])
    generate_kwargs = model._merge_generate_kwargs()

    sync_cuda()
    t0 = time.perf_counter()
    codes, _ = model.model.generate(
        input_ids=input_ids,
        instruct_ids=[None],
        languages=[language],
        speakers=[speaker],
        non_streaming_mode=True,
        **generate_kwargs,
    )
    sync_cuda()
    generated_at = time.perf_counter()

    wavs, sample_rate = model.model.speech_tokenizer.decode(
        [{"audio_codes": code} for code in codes]
    )
    sync_cuda()
    decoded_at = time.perf_counter()

    pcm = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    audio_seconds = len(pcm) / int(sample_rate)
    talker_seconds = generated_at - t0
    total_seconds = decoded_at - t0
    frames = int(codes[0].shape[0])
    rms = float(np.sqrt(np.mean(pcm**2)))

    return {
        "talker_ms": talker_seconds * 1000,
        "total_ms": total_seconds * 1000,
        "audio_seconds": audio_seconds,
        "rtf": total_seconds / max(audio_seconds, 1e-6),
        "frames": frames,
        "ms_per_frame": talker_seconds * 1000 / max(frames, 1),
        "rms": rms,
    }


def run_case(model, label: str, texts: list[str], *, speaker: str, language: str) -> None:
    rows: list[dict[str, float]] = []
    for index, text in enumerate(texts, start=1):
        row = synthesize_once(model, text, speaker=speaker, language=language)
        rows.append(row)
        print(
            f"{label} run {index}: talker={row['talker_ms']:.0f}ms "
            f"total={row['total_ms']:.0f}ms audio={row['audio_seconds']:.2f}s "
            f"RTF={row['rtf']:.2f} frames={row['frames']:.0f} "
            f"ms/frame={row['ms_per_frame']:.1f} rms={row['rms']:.3f}",
            flush=True,
        )

    print(
        f"{label} summary: talker_p50={statistics.median(row['talker_ms'] for row in rows):.0f}ms "
        f"total_p50={statistics.median(row['total_ms'] for row in rows):.0f}ms "
        f"rtf_p50={statistics.median(row['rtf'] for row in rows):.2f}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--speaker", default="Serena")
    parser.add_argument("--language", default="Russian")
    parser.add_argument("--text", action="append", help="Text to synthesize; may be passed multiple times")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    texts = args.text if args.text else DEFAULT_TEXTS

    print(f"loading {args.model_dir}", flush=True)
    t0 = time.perf_counter()
    model = load_model(args.model_dir)
    sync_cuda()
    print(f"loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    print("\n=== baseline qwen-tts generate() ===", flush=True)
    run_case(model, "baseline", texts, speaker=args.speaker, language=args.language)

    print("\n=== fast code predictor ===", flush=True)
    install_fast_code_predictor(model)
    run_case(model, "fast", texts, speaker=args.speaker, language=args.language)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
