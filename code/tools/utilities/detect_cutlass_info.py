#!/usr/bin/env python3
"""
Print CUTLASS installation details for Makefile consumption.

Outputs two lines:
1. Root path containing include/cutlass (if available)
2. Installed nvidia-cutlass-dsl version (blank if unavailable)
"""

import os
import sys
from pathlib import Path
from importlib import metadata as importlib_metadata
from packaging.version import Version


def main() -> None:
    candidates: list[Path] = []

    env_path = os.environ.get("CUTLASS_PATH")
    if env_path:
        candidates.append(Path(env_path))

    search_roots = [Path(p) for p in sys.path if p]
    search_roots.extend(
        [
            Path("/usr/local/include"),
            Path("/usr/include"),
            Path.home() / ".local" / "include",
        ]
    )

    cutlass_root = ""
    include_suffixes = [
        Path("include/cutlass/cutlass.h"),
        Path("cutlass/cutlass.h"),
    ]

    def probe_from_base(base: Path) -> Path | None:
        for suffix in include_suffixes:
            candidate = base / suffix
            if candidate.is_file():
                return candidate.parent.parent
        alt = base / "cutlass_library" / "source" / "include" / "cutlass" / "cutlass.h"
        if alt.is_file():
            return alt.parent.parent.parent
        return None

    for root in candidates + search_roots:
        hit = probe_from_base(root)
        if hit is not None:
            cutlass_root = str(hit)
            break

    cutlass_version = ""
    discovered: dict[str, str] = {}
    for dist_name in ("nvidia-cutlass-dsl", "nvidia-cutlass"):
        try:
            discovered[dist_name] = importlib_metadata.version(dist_name)
        except importlib_metadata.PackageNotFoundError:
            continue
        except Exception:
            continue

    if discovered:
        preferred = max(discovered.items(), key=lambda item: Version(item[1]))
        cutlass_version = preferred[1]

    print(cutlass_root)
    print(cutlass_version)


if __name__ == "__main__":
    main()
