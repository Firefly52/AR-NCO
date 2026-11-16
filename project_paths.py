"""Portable repository-relative paths for the release copy."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def resolve_path(value, base=None):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if (ROOT / path).exists():
        return ROOT / path
    if base is not None and (Path(base) / path).exists():
        return Path(base) / path
    return ROOT / path
