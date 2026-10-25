"""Check repository-local Markdown file links, including scoped agent instructions."""

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.md"],
            cwd=ROOT,
        )
        .decode()
        .split("\0")
    )
    failures = []
    count = 0
    for name in sorted(set(filter(None, names))):
        path = ROOT / name
        if not path.is_file():
            continue  # A staged deletion is not a document to check.
        count += 1
        content = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
        links = re.findall(r"\[[^\]\n]*\]\(([^)\n]+)\)", content)
        links += re.findall(r"^\[[^\]\n]+\]:\s*(\S+)", content, flags=re.MULTILINE)
        for link in links:
            url = urlsplit(link.strip().strip("<>"))
            if url.scheme or url.netloc or not url.path:
                continue
            target = (path.parent / unquote(url.path)).resolve()
            if not target.is_relative_to(ROOT) or not target.exists():
                failures.append(f"{name}: missing or out-of-repository link {link}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Repository Markdown file links verified: {count} documents")


if __name__ == "__main__":
    main()
