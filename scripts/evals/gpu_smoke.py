from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def main() -> int:
    report: dict[str, object] = {"checks": {}, "metrics": {}}
    if not shutil.which("nvidia-smi"):
        print("nvidia-smi is required for GPU smoke", file=sys.stderr)
        return 1

    gpu_name = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).splitlines()[0]
    vram_mb = int(
        run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]).splitlines()[0]
    )
    report["checks"] = {"nvidia_smi": True, "gpu_name": gpu_name, "vram_mb": vram_mb}

    manifest = Path("models/artifacts.manifest.json")
    if not manifest.exists():
        print("missing models/artifacts.manifest.json; real model smoke fails closed", file=sys.stderr)
        print(json.dumps(report, indent=2))
        return 1

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
