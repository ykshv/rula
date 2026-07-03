from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from huggingface_hub import HfApi, snapshot_download


DEFAULT_MODELS = [
    "Qwen/Qwen3-14B-FP8",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "ai-sage/GigaAM-v3",
    "nvidia/Audio2Face-3D-v3.0",
]


def safe_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download local avatar model artifacts from HF.")
    parser.add_argument("--models-dir", default="models/hf")
    parser.add_argument(
        "--model",
        action="append",
        dest="selected_models",
        help="Specific HF repo id to download. Can be repeated.",
    )
    parser.add_argument("--include-fallback-llm", action="store_true")
    parser.add_argument(
        "--disable-xet",
        action="store_true",
        help="Disable hf-xet and use regular Hub transfer path. Useful when Xet stalls.",
    )
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()

    models = list(args.selected_models or DEFAULT_MODELS)
    if args.include_fallback_llm:
        models.insert(1, "Qwen/Qwen3-14B")

    root = Path(args.models_dir).resolve()
    if args.disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    manifest_path = Path("models/artifacts.manifest.json")
    if manifest_path.exists():
        manifest: dict[str, object] = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("models", {})
    else:
        manifest = {"models": {}}
    manifest["download_root"] = str(root)

    for repo_id in models:
        started = perf_counter()
        info = api.model_info(repo_id)
        local_dir = root / safe_repo_id(repo_id)
        print(f"Downloading {repo_id} -> {local_dir}")
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            max_workers=args.max_workers,
            resume_download=True,
        )
        elapsed = round(perf_counter() - started, 2)
        manifest["models"][repo_id] = {
            "revision": info.sha,
            "local_dir": str(Path(path).resolve()),
            "elapsed_seconds": elapsed,
            "gated": info.gated,
        }

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    raise SystemExit(main())
