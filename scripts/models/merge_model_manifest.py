from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--local-dir", required=True)
    args = parser.parse_args()

    manifest_path = Path("models/artifacts.manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"models": {}}

    info = HfApi().model_info(args.repo_id)
    manifest.setdefault("models", {})[args.repo_id] = {
        "revision": info.sha,
        "local_dir": str(Path(args.local_dir).resolve()),
        "gated": info.gated,
        "download_method": "curl_parallel",
    }
    manifest["download_root"] = str(Path("models/hf").resolve())
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
