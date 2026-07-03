from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_FP8_FILES = [
    ".gitattributes",
    "LICENSE",
    "README.md",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def headers(token: str | None, extra: dict[str, str] | None = None) -> dict[str, str]:
    result = {"User-Agent": "ru-local-avatar-downloader/0.1"}
    if token:
        result["Authorization"] = f"Bearer {token}"
    if extra:
        result.update(extra)
    return result


def request(url: str, token: str | None, method: str = "GET", extra: dict[str, str] | None = None):
    return urllib.request.Request(url, method=method, headers=headers(token, extra))


def remote_size(url: str, token: str | None) -> int | None:
    try:
        with urllib.request.urlopen(request(url, token, method="HEAD"), timeout=60) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value else None
    except Exception:
        try:
            with urllib.request.urlopen(
                request(url, token, extra={"Range": "bytes=0-0"}),
                timeout=60,
            ) as response:
                content_range = response.headers.get("Content-Range")
                if content_range and "/" in content_range:
                    return int(content_range.rsplit("/", 1)[1])
                value = response.headers.get("Content-Length")
                return int(value) if value else None
        except Exception:
            return None


def repo_file_sizes(repo_id: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in HfApi().list_repo_tree(repo_id, recursive=True, repo_type="model"):
        path = getattr(item, "path", None)
        size = getattr(item, "size", None)
        if isinstance(path, str) and isinstance(size, int):
            result[path] = size
    return result


def resolve_files(repo_id: str, requested_files: list[str]) -> tuple[list[str], dict[str, int]]:
    sizes = repo_file_sizes(repo_id)
    if requested_files:
        return requested_files, sizes
    if repo_id == "Qwen/Qwen3-14B-FP8":
        return DEFAULT_FP8_FILES, sizes
    return sorted(sizes), sizes


def download_file(
    repo_id: str,
    file_name: str,
    local_dir: Path,
    token: str | None,
    expected_size: int | None,
) -> tuple[str, int]:
    url = f"https://huggingface.co/{repo_id}/resolve/main/{file_name}"
    target = local_dir / file_name
    target.parent.mkdir(parents=True, exist_ok=True)

    expected = expected_size if expected_size is not None else remote_size(url, token)
    existing = target.stat().st_size if target.exists() else 0
    if expected is not None and existing == expected:
        return file_name, existing
    if expected is not None and existing > expected:
        target.unlink()
        existing = 0

    extra_headers = {"Range": f"bytes={existing}-"} if existing > 0 else None
    req = request(url, token, extra=extra_headers)
    mode = "ab" if existing > 0 else "wb"

    started = time.monotonic()
    last_log = started
    downloaded = existing
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            if existing > 0 and response.status == 200:
                mode = "wb"
                downloaded = 0
            with target.open(mode + "") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_log >= 10:
                        mbps = (downloaded - existing) / 1024 / 1024 / max(now - started, 0.001)
                        total = f"{expected / 1024 / 1024:.0f}MB" if expected else "unknown"
                        print(f"{file_name}: {downloaded / 1024 / 1024:.0f}MB/{total} {mbps:.2f}MB/s", flush=True)
                        last_log = now
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and expected is not None and target.exists() and target.stat().st_size == expected:
            return file_name, expected
        raise

    final_size = target.stat().st_size
    if expected is not None and final_size != expected:
        raise RuntimeError(f"{file_name}: size mismatch {final_size} != {expected}")
    return file_name, final_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Qwen/Qwen3-14B-FP8")
    parser.add_argument("--local-dir", default="models/hf/Qwen__Qwen3-14B-FP8")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--file", action="append", default=[])
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id} to {local_dir.resolve()}", flush=True)
    if not token:
        print("HF_TOKEN is not set; public download will run without auth.", file=sys.stderr, flush=True)

    files, sizes = resolve_files(args.repo_id, args.file)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [
            pool.submit(download_file, args.repo_id, file_name, local_dir, token, sizes.get(file_name))
            for file_name in files
        ]
        for future in as_completed(futures):
            file_name, size = future.result()
            print(f"done {file_name} {size / 1024 / 1024:.1f}MB", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
