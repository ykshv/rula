from __future__ import annotations

import json
import sys
from pathlib import Path


REQUIRED_EVIDENCE = {
    "nvidia_open_model_license",
    "audio2face_3d_sdk",
    "audio2emotion_if_enabled",
    "qwen3",
    "qwen3_tts",
    "gigaam_v3",
    "vap",
    "livekit",
    "vllm",
    "voice_cloning_consent_policy",
}


def main() -> int:
    evidence_path = Path("models/legal_evidence.json")
    if not evidence_path.exists():
        print("missing models/legal_evidence.json; legal gate fails closed", file=sys.stderr)
        return 1

    data = json.loads(evidence_path.read_text(encoding="utf-8"))
    present = set(data.get("evidence", {}).keys())
    missing = sorted(REQUIRED_EVIDENCE - present)
    if missing:
        print(f"missing legal evidence: {', '.join(missing)}", file=sys.stderr)
        return 1

    print("legal gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
