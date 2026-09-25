#!/usr/bin/env python3
"""
Generate ``provenance.json``: a machine-checkable record of what this
repository contains and what it reproduces.

The file is generated, never hand-edited, so it cannot drift from the data.
It records, for every tracked input and source file, a SHA-256 digest; plus
the git commit, the interpreter, the resolved dependency versions and the
canonical CSC quantities recomputed from the bundled demo tables.

Usage
-----
    python tools/make_provenance.py            # write provenance.json
    python tools/make_provenance.py --check    # verify, exit 1 on mismatch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUT = REPO / "provenance.json"

# Files whose digests pin the reproduction. Globs are resolved sorted.
TRACKED = (
    "aws_align/*.py",
    "tests/*.py",
    "tools/*.py",
    "data/demo/*.csv",
    "pyproject.toml",
    "requirements.txt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ("git", *args),
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip() or None


def manifest() -> dict[str, str]:
    entries: dict[str, str] = {}
    for pattern in TRACKED:
        for path in sorted(REPO.glob(pattern)):
            if path.is_file():
                entries[path.relative_to(REPO).as_posix()] = sha256(path)
    return entries


def canonical_csc() -> dict[str, object]:
    sys.path.insert(0, str(REPO))
    from aws_align import compute_csc, load_divergence, load_insularity

    insularity = REPO / "data" / "demo" / "citation_cluster_insularity.csv"
    divergence = REPO / "data" / "demo" / "rbo_fragmentation.csv"
    result = compute_csc(
        load_insularity(insularity),
        load_divergence(divergence),
        corpus="Plastic recycling",
    )
    return {
        "inputs": {
            "insularity": insularity.relative_to(REPO).as_posix(),
            "divergence": divergence.relative_to(REPO).as_posix(),
        },
        "n_clusters": int(result.n_clusters),
        "n_pairs": int(result.n_pairs),
        "E_within_corpus": int(result.E_within_corpus),
        "E_intra_cluster": int(result.E_intra_cluster),
        "E_cross_cluster": int(result.E_cross_cluster),
        "S_cross": float(result.S_cross),
        "D_bar": float(result.D_bar),
        "D_bar_w": float(result.D_bar_w),
        "fragmentation": float(result.fragmentation),
        "CSC": float(result.csc),
        "identity": "CSC = 1 - S_cross * D_bar_w",
        "published_value": 0.402,
    }


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "pytest"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", None)
        except ImportError:
            versions[name] = None
    return versions


def build() -> dict[str, object]:
    return {
        "schema": "aws-align/provenance/1",
        "generated_by": "tools/make_provenance.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "repository": (
            "https://github.com/E4CE-UA/awareness-without-synthesis"
        ),
        "git": {
            "commit": git("rev-parse", "HEAD"),
            "describe": git("describe", "--tags", "--always", "--dirty"),
            "dirty": bool(git("status", "--porcelain")),
            "note": (
                "'commit' is HEAD at generation time. A record committed into "
                "the repository necessarily names its own parent commit, "
                "because writing the file changes the tree. Content integrity "
                "is pinned by 'file_digests_sha256', which "
                "'make verify-provenance' checks independently of git."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "dependencies": dependency_versions(),
        },
        "cluster_id_convention": (
            "Cluster identifiers are canonical 'C<n>' strings in every bundled "
            "demo table; see tools/fix_cluster_labels.py."
        ),
        "canonical_demo": canonical_csc(),
        "file_digests_sha256": manifest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify digests and CSC against the committed provenance.json",
    )
    args = parser.parse_args()

    current = build()

    if not args.check:
        OUTPUT.write_text(json.dumps(current, indent=2) + "\n")
        digests = current["file_digests_sha256"]
        csc = current["canonical_demo"]["CSC"]
        print(f"wrote {OUTPUT.name}: {len(digests)} files, CSC = {csc:.7f}")
        return 0

    if not OUTPUT.is_file():
        print(f"missing {OUTPUT.name}; run without --check", file=sys.stderr)
        return 1

    stored = json.loads(OUTPUT.read_text())
    problems: list[str] = []

    old_digests = stored.get("file_digests_sha256", {})
    new_digests = current["file_digests_sha256"]
    for name in sorted(set(old_digests) | set(new_digests)):
        before, after = old_digests.get(name), new_digests.get(name)
        if before != after:
            problems.append(f"digest changed: {name}")

    before_csc = stored.get("canonical_demo", {}).get("CSC")
    after_csc = current["canonical_demo"]["CSC"]
    if before_csc is None or abs(before_csc - after_csc) > 1e-12:
        problems.append(f"CSC changed: {before_csc} -> {after_csc}")

    if problems:
        for line in problems:
            print(line, file=sys.stderr)
        return 1

    print(
        f"provenance OK: {len(new_digests)} digests match, "
        f"CSC = {after_csc:.7f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
