"""Check that release archives contain runtime code, not research corpora."""

from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from pathlib import Path

MAX_SDIST_BYTES = 1_000_000
ROOT_FILES = {".gitignore", "CHANGELOG.md", "LICENSE", "PKG-INFO", "README.md",
              "pyproject.toml"}
REQUIRED_MODULES = {"__init__.py", "__main__.py", "analyzer.py", "cli.py",
                    "structured.py"}
WHEEL_METADATA = {"METADATA", "WHEEL", "entry_points.txt", "RECORD",
                  "licenses/LICENSE"}


def check_sdist(path: Path) -> tuple[int, int]:
    if path.stat().st_size >= MAX_SDIST_BYTES:
        raise ValueError("Source distribution exceeds the 1 MB runtime/docs cap")
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
    if not members:
        raise ValueError("Source distribution is empty")
    root = members[0].name.split("/", 1)[0]
    modules = set()
    for member in members:
        parts = member.name.split("/")
        if not member.isfile() or parts[0] != root or any(
            part in ("", ".", "..") for part in parts
        ):
            raise ValueError("Unexpected source distribution path")
        relative = Path(*parts[1:])
        if len(relative.parts) == 1 and relative.name in ROOT_FILES:
            continue
        if relative.parts[:1] == ("docs",) and relative.suffix == ".md":
            continue
        if relative.parts[:2] == ("src", "ko_pii_guard") and relative.suffix == ".py":
            modules.add(relative.name)
            continue
        raise ValueError("Source distribution contains an unexpected file category")
    if not REQUIRED_MODULES <= modules:
        raise ValueError("Source distribution is missing required runtime modules")
    return path.stat().st_size, len(members)


def check_wheel(path: Path) -> tuple[int, int]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Wheel contains duplicate archive entries")
        modules = set()
        dist_info = None
        for name in names:
            parts = name.split("/")
            if any(part in ("", ".", "..") for part in parts):
                raise ValueError("Unexpected wheel path")
            if len(parts) == 2 and parts[0] == "ko_pii_guard" and name.endswith(".py"):
                modules.add(parts[1])
                continue
            if len(parts) >= 2 and parts[0].startswith("ko_pii_guard-") and parts[0].endswith(
                ".dist-info"
            ) and "/".join(parts[1:]) in WHEEL_METADATA:
                if dist_info is not None and dist_info != parts[0]:
                    raise ValueError("Wheel contains multiple metadata directories")
                dist_info = parts[0]
                continue
            raise ValueError("Wheel contains an unexpected file category")
        entry_points = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        if len(entry_points) != 1 or b"ko-pii-guard = ko_pii_guard.cli:main" not in archive.read(
            entry_points[0]
        ):
            raise ValueError("Wheel CLI entry point is missing")
        if dist_info is None or any(f"{dist_info}/{name}" not in names for name in WHEEL_METADATA):
            raise ValueError("Wheel metadata is incomplete")
    if not REQUIRED_MODULES <= modules:
        raise ValueError("Wheel is missing required runtime modules")
    return path.stat().st_size, len(names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    sdists = list(args.dist.glob("ko_pii_guard-*.tar.gz"))
    wheels = list(args.dist.glob("ko_pii_guard-*.whl"))
    if len(sdists) != 1 or len(wheels) != 1:
        parser.error("dist must contain exactly one ko-pii-guard sdist and wheel")
    sdist_bytes, sdist_files = check_sdist(sdists[0])
    wheel_bytes, wheel_files = check_wheel(wheels[0])
    print(json.dumps({"sdist_bytes": sdist_bytes, "sdist_files": sdist_files,
                      "wheel_bytes": wheel_bytes, "wheel_files": wheel_files}))


if __name__ == "__main__":
    main()
