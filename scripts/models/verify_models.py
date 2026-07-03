from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest_path = Path("models/artifacts.manifest.json")
    if not manifest_path.exists():
        print("missing models/artifacts.manifest.json", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksums: dict[str, dict[str, str]] = {}
    failures: list[str] = []

    for repo_id, info in manifest.get("models", {}).items():
        local_dir = Path(info["local_dir"])
        if not local_dir.exists():
            failures.append(f"{repo_id}:missing_dir")
            continue
        repo_checksums: dict[str, str] = {}
        for path in sorted(local_dir.rglob("*")):
            relative = path.relative_to(local_dir)
            parts = set(relative.parts)
            if ".cache" in parts:
                continue
            if path.suffix in {".incomplete", ".metadata", ".lock"}:
                continue
            if path.is_file():
                repo_checksums[str(relative).replace("\\", "/")] = sha256_file(path)
        checksums[repo_id] = repo_checksums

    checksum_path = Path("models/artifacts.checksums.json")
    checksum_path.write_text(json.dumps(checksums, indent=2, ensure_ascii=False), encoding="utf-8")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Wrote {checksum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
