from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("empty series")
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * p))
    return ordered[index]


def main() -> int:
    path = Path("evals/results/latency_samples.jsonl")
    if not path.exists():
        print("missing evals/results/latency_samples.jsonl", file=sys.stderr)
        return 1

    first_audio: list[float] = []
    avatar_reaction: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        first_audio.append(float(item["speech_end_to_first_audio_ms"]))
        avatar_reaction.append(float(item["speech_end_to_reaction_ms"]))

    report = {
        "first_audio_p50": statistics.median(first_audio),
        "first_audio_p95": percentile(first_audio, 0.95),
        "avatar_reaction_p95": percentile(avatar_reaction, 0.95),
    }
    print(json.dumps(report, indent=2))

    if report["first_audio_p95"] > 1100 or report["avatar_reaction_p95"] > 250:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
