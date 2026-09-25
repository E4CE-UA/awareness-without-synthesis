#!/usr/bin/env python3
"""
Normalise demo cluster identifiers to the canonical ``C<n>`` convention.

Why this exists
---------------
``data/demo/rbo_fragmentation.csv`` labels its clusters ``C2..C7`` while the
other bundled demo tables labelled them ``2..7``. Because the cluster id is a
pure join key between the insularity table, the divergence table and the
cluster-size table, the mismatch made ``aws-align diagnose`` and the CSC test
suite fail with:

    ValueError: Divergence table contains clusters absent from insularity:
    ['C2', 'C3', 'C4', 'C5', 'C6', 'C7']

This script rewrites only the identifier columns, and only when the value is a
bare integer-like token. It does not touch numeric data, formulas or any other
column, so the canonical CSC value is unchanged (it is recomputed and asserted
by tests/test_csc.py).

Idempotent: running it twice is a no-op ("C2" is left as "C2", never "CC2").

Usage
-----
    python tools/fix_cluster_labels.py            # apply
    python tools/fix_cluster_labels.py --check    # exit 1 if a fix is needed
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "data" / "demo"

# file -> identifier columns that must carry the C-prefix
TARGETS: dict[str, tuple[str, ...]] = {
    "citation_cluster_insularity.csv": ("cluster",),
    "paper_topics.csv": ("cluster",),
    "semantic_topics.csv": ("cluster",),
    "alignment_raw.csv": ("cluster_a", "cluster_b"),
}

_BARE_INT = re.compile(r"^\d+$")


def canonical(value: object) -> str:
    """``2`` / ``2.0`` / ``"2"`` -> ``"C2"``; ``"C2"`` -> ``"C2"``."""
    text = str(value).strip()
    if text.endswith(".0") and _BARE_INT.match(text[:-2] or "x"):
        text = text[:-2]
    if _BARE_INT.match(text):
        return f"C{int(text)}"
    return text


def process(path: Path, columns: tuple[str, ...], apply: bool) -> int:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    changed = 0
    for column in columns:
        if column not in frame.columns:
            raise SystemExit(f"{path.name}: expected column {column!r}")
        new = frame[column].map(canonical)
        changed += int((new != frame[column]).sum())
        frame[column] = new
    if changed and apply:
        frame.to_csv(path, index=False)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report without writing; exit 1 if any file needs a fix",
    )
    args = parser.parse_args()

    total = 0
    for name, columns in TARGETS.items():
        path = DEMO / name
        if not path.is_file():
            raise SystemExit(f"missing demo file: {path}")
        changed = process(path, columns, apply=not args.check)
        total += changed
        verb = "needs" if args.check else "rewrote"
        print(f"{name:40s} {verb} {changed:6d} cell(s)")

    print(f"total: {total} cell(s)")
    if args.check and total:
        print("cluster identifiers are NOT canonical", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
