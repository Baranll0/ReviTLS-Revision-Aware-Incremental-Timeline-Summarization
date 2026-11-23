"""Record the software/hardware environment of a result run into outputs/final/env.json."""

import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({exc.__class__.__name__})"


def version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else None


def main() -> None:
    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_dirty_files": [line for line in run(["git", "status", "--porcelain"]).splitlines() if line.strip()],
        "packages": {p: version(p) for p in ("torch", "sentence-transformers", "transformers", "openai", "numpy", "rouge-score", "pydantic", "scipy")},
        "gpu": run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "ollama_version": run(["ollama", "--version"]),
        "ollama_models": run(["ollama", "list"]),
        "config_hashes": {str(p): sha(p) for p in sorted(Path("configs").glob("*.yaml"))},
    }
    path = Path("outputs/final/env.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
