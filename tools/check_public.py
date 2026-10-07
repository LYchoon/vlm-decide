"""Check the staged publication surface without reading any runtime directories."""
from pathlib import Path
import re
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    paths = [p for p in paths if p]
    assert paths, "Stage the intended source files before checking"
    assert [p for p in paths if p.lower().endswith(".md")] == ["README.md"]
    readme = subprocess.check_output(["git", "show", ":README.md"], cwd=root).decode().strip()
    assert len(readme.splitlines()) == 1 and readme.endswith(".")
    for path in paths:
        assert Path(path).suffix == ".py" or path in {"README.md", "LICENSE", ".gitignore", "pyproject.toml"}, path
        text = subprocess.check_output(["git", "show", ":"+path], cwd=root).decode()
        assert not re.search(r"https?://(?:www\.)?(?:youtube\.com|youtu\.be)/", text, re.I), path
        assert not re.search(r"(?im)^\s*(?:password|secret|api_key)\s*=\s*['\"][^'\"]+['\"]", text), path
    print(f"Publication check passed: {len(paths)} text source files")


if __name__ == "__main__":
    main()
