"""Check the staged publication surface without reading any runtime directories."""
from pathlib import Path
import json
import re
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    paths = [p for p in paths if p]
    assert paths, "Stage the intended source files before checking"
    assert [p for p in paths if p.lower().endswith(".md")] == ["README.md"]
    readme = subprocess.check_output(["git", "show", ":README.md"], cwd=root).decode().strip()
    assert readme.startswith("# vlm-decide")
    result_files = {f"results/imagenette-20261014/{name}" for name in
                    ("summary.json", "config.json", "dataset.json", "manifest.json", "predictions.jsonl")}
    result_files |= {f"results/cv-20261023/{name}" for name in ("metrics.json", "protocol.json")}
    for path in paths:
        assert (Path(path).suffix == ".py" or path in result_files
                or path in {"README.md", "LICENSE", ".gitignore", "pyproject.toml"}), path
        text = subprocess.check_output(["git", "show", ":"+path], cwd=root).decode()
        if path in result_files:
            if path.endswith(".jsonl"):
                for line in text.splitlines():
                    json.loads(line)
            else:
                json.loads(text)
        assert not re.search(r"https?://(?:www\.)?(?:youtube\.com|youtu\.be)/", text, re.I), path
        assert not re.search(r"(?im)^\s*(?:password|secret|api_key)\s*=\s*['\"][^'\"]+['\"]", text), path
    print(f"Publication check passed: {len(paths)} text source files")


if __name__ == "__main__":
    main()
