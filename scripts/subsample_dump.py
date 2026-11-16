"""Subsample a LAMMPS dump to a manageable size for in-repo tutorials.

Usage:
    python scripts/subsample_dump.py \
        --in /path/to/dump_1.trj \
        --out data/cdp_525K_dump_1.trj \
        --n_frames 1500 \
        --stride 1

A standard LAMMPS text dump has frames separated by 'ITEM: TIMESTEP' lines.
This script does NOT parse the file; it copies frames as opaque blocks, so it
preserves any unusual column ordering or metadata exactly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def split_frames(path: Path):
    """Yield each frame as a list of lines (still including the ITEM headers)."""
    current: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("ITEM: TIMESTEP") and current:
                yield current
                current = [line]
            else:
                current.append(line)
        if current:
            yield current


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="src", required=True, type=Path)
    p.add_argument("--out", dest="dst", required=True, type=Path)
    p.add_argument(
        "--n_frames",
        type=int,
        default=1500,
        help="Maximum number of frames to keep (default: 1500).",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Take every Nth frame (default: 1; the dump's saved cadence).",
    )
    args = p.parse_args(argv)

    if not args.src.exists():
        print(f"Input not found: {args.src}", file=sys.stderr)
        return 1
    args.dst.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.dst.open("w", encoding="utf-8") as out:
        for i, frame in enumerate(split_frames(args.src)):
            if i % args.stride != 0:
                continue
            out.writelines(frame)
            written += 1
            if written >= args.n_frames:
                break

    bytes_written = args.dst.stat().st_size
    print(
        f"Wrote {written} frames to {args.dst} ({bytes_written / 1024**2:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
