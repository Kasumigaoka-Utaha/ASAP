"""Create a source-only archive without touching existing Git history or data."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "CITATION.cff", ".gitignore")
SOURCE_DIRS = ("ASAP-3DBPP", "ASAP-Routing", "docs", "tools", "tests")
EXCLUDED_DIRS = {".git", "__pycache__", ".venv", ".pytest_cache", "logs", "eval_logs",
                 "runs", "data", "dataset", "test_dataset", "checkpoints", "dist", "swanlog"}
EXTENSIONS = {".py", ".md", ".yaml", ".yml", ".sh", ".txt", ".cff"}


def main():
    paths = [ROOT / name for name in ROOT_FILES]
    for name in SOURCE_DIRS:
        directory = ROOT / name
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing source directory: {directory}")
        paths.extend(path for path in directory.rglob("*") if path.is_file()
                     and not path.is_symlink()
                     and not any(part in EXCLUDED_DIRS for part in path.relative_to(ROOT).parts)
                     and (path.suffix in EXTENSIONS or path.name in {"LICENSE", ".gitignore"}))
    destination = ROOT / "dist" / "ASAP-source.zip"
    destination.parent.mkdir(exist_ok=True)
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, "ASAP/" + path.relative_to(ROOT).as_posix())
    with ZipFile(destination) as archive:
        if archive.testzip():
            raise RuntimeError("Archive integrity check failed")
        print(f"Created {destination} ({len(archive.namelist())} source files)")


if __name__ == "__main__":
    main()
