"""
Subcommands
-----------
aws-align diagnose
    Compute the canonical CSC diagnostic:

        fragmentation = S_cross * D_bar_w
        CSC = 1 - fragmentation

    The optional size-mixing null applies only to S_cross.

aws-align map
    Render the pairwise vocabulary-fragmentation map.

aws-align align
    Run or reproduce the LLM cross-cluster vocabulary-alignment analysis.
    ``--dry-run`` rebuilds the summary from an archived raw CSV without API
    calls.

Run ``aws-align <subcommand> --help`` for command-specific options.
With no subcommand, the CLI prints a short banner.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union


DISTRIBUTION_NAME = "awareness-without-synthesis"

try:
    __version__ = version(DISTRIBUTION_NAME)
except PackageNotFoundError:
    # Useful when running directly from an uninstalled source checkout.
    __version__ = "1.0.0"


# Repository layout:
#   awareness-without-synthesis/
#       aws_align/
#       data/demo/
DEMO = Path(__file__).resolve().parent.parent / "data" / "demo"


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def _resolve(
    path: Optional[Union[str, Path]],
    demo_name: str,
) -> str:
    """Return the explicit path, or the corresponding bundled demo file."""
    if path:
        candidate = Path(path).expanduser()
        if not candidate.is_file():
            raise SystemExit(f"error: input file does not exist: {candidate}")
        return str(candidate)

    demo_path = DEMO / demo_name
    if not demo_path.is_file():
        raise SystemExit(
            "error: no input file was supplied and the bundled demo file is "
            f"missing: {demo_path}"
        )
    return str(demo_path)


def _parse_cluster_id(value: Any) -> Union[int, str]:
    """
    Preserve labels such as C2 while converting plain integer strings to int.
    """
    text = str(value).strip()
    if not text:
        raise ValueError("cluster identifier cannot be empty")

    try:
        numeric = float(text)
    except ValueError:
        return text

    if numeric.is_integer() and text.replace("+", "").replace("-", "").replace(".", "").isdigit():
        return int(numeric)

    return text


def _load_labels(path: Optional[Union[str, Path]]) -> Optional[Dict[Union[int, str], str]]:
    """Load a JSON object mapping cluster identifiers to display labels."""
    if not path:
        return None

    labels_path = Path(path).expanduser()
    if not labels_path.is_file():
        raise SystemExit(f"error: labels file does not exist: {labels_path}")

    try:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: invalid labels JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise SystemExit("error: labels JSON must be an object")

    return {
        _parse_cluster_id(key): str(value)
        for key, value in payload.items()
    }


def _prepare_output(path: Union[str, Path]) -> Path:
    """Create the parent directory for an output file and return its Path."""
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _format_optional_float(
    value: Any,
    digits: int = 4,
    missing: str = "NA",
) -> str:
    if value is None:
        return missing
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return missing


# -----------------------------------------------------------------------------
# diagnose
# -----------------------------------------------------------------------------
def cmd_diagnose(args: argparse.Namespace) -> int:
    from .csc import compute_csc
    from .io import load_cluster_sizes, load_divergence, load_insularity

    insularity_path = _resolve(
        args.insularity,
        "citation_cluster_insularity.csv",
    )
    divergence_path = _resolve(
        args.divergence,
        "rbo_fragmentation.csv",
    )

    insularity = load_insularity(insularity_path)
    divergence = load_divergence(divergence_path)

    cluster_sizes = None
    effective_n_nulls = int(args.n_nulls)

    if effective_n_nulls > 0:
        sizes_path: Optional[str] = None

        if args.sizes:
            sizes_path = _resolve(args.sizes, "paper_topics.csv")
        else:
            demo_sizes = DEMO / "paper_topics.csv"
            if demo_sizes.is_file():
                sizes_path = str(demo_sizes)

        if sizes_path is None:
            print(
                "[warn] n_nulls > 0 but no cluster-size file is available; "
                "the auxiliary S_cross null will be skipped",
                file=sys.stderr,
            )
            effective_n_nulls = 0
        else:
            try:
                cluster_sizes = load_cluster_sizes(sizes_path)
            except Exception as exc:
                print(
                    "[warn] could not load cluster sizes "
                    f"({type(exc).__name__}: {exc}); "
                    "the auxiliary S_cross null will be skipped",
                    file=sys.stderr,
                )
                effective_n_nulls = 0
                cluster_sizes = None

    result = compute_csc(
        insularity,
        divergence,
        corpus=args.corpus,
        cluster_sizes=cluster_sizes,
        n_nulls=effective_n_nulls,
        seed=args.seed,
    )

    print("=" * 68)
    print(f"AWS-align diagnostic — corpus: {result.corpus}")
    print("=" * 68)
    print(f"  clusters                  : {result.n_clusters}")
    print(f"  unordered cluster pairs   : {result.n_pairs}")
    print(f"  within-corpus edges       : {result.E_within_corpus}")
    print(f"  intra-cluster edges       : {result.E_intra_cluster}")
    print(f"  cross-cluster edges       : {result.E_cross_cluster}")
    print(f"  S_cross                   : {result.S_cross:.4f}")
    print(f"  D_bar                     : {result.D_bar:.4f}")
    print(f"  D_bar_w                   : {result.D_bar_w:.4f}")
    print(f"  fragmentation             : {result.fragmentation:.4f}")
    print(f"  CSC                       : {result.csc:.4f}")

    if getattr(result, "n_nulls", 0) > 0:
        print("-" * 68)
        print("  Auxiliary size-mixing null for S_cross only")
        print(
            "  null mean S_cross         : "
            f"{_format_optional_float(result.mu_null_scross)}"
        )
        print(
            "  null SD S_cross           : "
            f"{_format_optional_float(result.sigma_null_scross)}"
        )
        print(
            "  z(S_cross)                : "
            f"{_format_optional_float(result.z_scross, digits=2)}"
        )
        print(
            "  empirical p               : "
            f"{_format_optional_float(result.p_empirical_scross, digits=4)}"
        )
        print(f"  null draws                : {result.n_nulls}")

    print("=" * 68)

    if args.out:
        output = _prepare_output(args.out)
        result.pairwise.to_csv(output, index=False)
        print(f"pairwise table -> {output}")

    return 0


# -----------------------------------------------------------------------------
# map
# -----------------------------------------------------------------------------
def cmd_map(args: argparse.Namespace) -> int:
    from .fragmentation import plot_fragmentation_map
    from .io import load_divergence, load_insularity

    insularity = load_insularity(
        _resolve(args.insularity, "citation_cluster_insularity.csv")
    )
    divergence = load_divergence(
        _resolve(args.divergence, "rbo_fragmentation.csv")
    )
    labels = _load_labels(args.labels)

    csc_value = None
    if not args.no_csc:
        from .csc import compute_csc

        csc_value = compute_csc(
            insularity,
            divergence,
            corpus=args.corpus,
            n_nulls=0,
        ).csc

    output = _prepare_output(args.out)

    plot_fragmentation_map(
        insularity,
        divergence,
        cluster_labels=labels,
        corpus=args.corpus,
        csc=csc_value,
        outfile=str(output),
        dpi=args.dpi,
    )

    print(f"fragmentation map -> {output}")
    return 0


# -----------------------------------------------------------------------------
# align
# -----------------------------------------------------------------------------
def cmd_align(args: argparse.Namespace) -> int:
    from .align import (
        alignment_report,
        consensus_from_raw,
        load_cluster_terms,
        load_env,
    )

    if args.dry_run:
        raw_path = _resolve(args.raw, "alignment_raw.csv")

        summary = consensus_from_raw(
            raw_path,
            min_confidence=args.min_confidence,
            min_models=args.min_models,
        )

        print(
            alignment_report(
                summary,
                min_confidence=args.min_confidence,
                min_models=args.min_models,
                top_k=args.top_k,
            )
        )

    else:
        from .align import align_vocabulary

        load_env()

        terms = load_cluster_terms(
            _resolve(args.terms, "semantic_topics.csv"),
            top_k=args.top_k,
        )
        labels = _load_labels(args.labels)

        raw, summary = align_vocabulary(
            terms,
            cluster_labels=labels,
            max_pairs=args.max_pairs,
            sleep=args.sleep,
            min_confidence=args.min_confidence,
            min_models=args.min_models,
            checkpoint_path=args.checkpoint,
            resume=not args.no_resume,
            retry_failed=args.retry_failed,
            max_retries=args.max_retries,
            timeout=args.timeout,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        if args.raw_out:
            raw_output = _prepare_output(args.raw_out)
            raw.to_csv(raw_output, index=False)
            print(f"raw responses -> {raw_output}")

        print(
            alignment_report(
                summary,
                min_confidence=args.min_confidence,
                min_models=args.min_models,
                top_k=args.top_k,
            )
        )

    if args.out:
        summary_output = _prepare_output(args.out)
        summary.to_csv(summary_output, index=False)
        print(f"summary table -> {summary_output}")

    return 0


# -----------------------------------------------------------------------------
# Parser
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aws-align",
        description=(
            "Diagnose hidden vocabulary fragmentation in scholarly "
            "knowledge graphs."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"aws-align {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    # diagnose
    diagnose = subparsers.add_parser(
        "diagnose",
        help="compute the canonical CSC diagnostic",
        description=(
            "Compute CSC = 1 - (S_cross * D_bar_w). "
            "The optional size-mixing null applies only to S_cross."
        ),
    )
    diagnose.add_argument(
        "-i",
        "--insularity",
        help="citation-insularity CSV (default: bundled demo)",
    )
    diagnose.add_argument(
        "-d",
        "--divergence",
        help="pairwise vocabulary-divergence CSV (default: bundled demo)",
    )
    diagnose.add_argument(
        "-s",
        "--sizes",
        help=(
            "paper-level cluster CSV for the auxiliary S_cross null "
            "(default: bundled demo when available)"
        ),
    )
    diagnose.add_argument(
        "--corpus",
        default="corpus",
        help="corpus name used in output labels",
    )
    diagnose.add_argument(
        "--n-nulls",
        type=int,
        default=0,
        dest="n_nulls",
        help="number of auxiliary S_cross null draws (default: 0)",
    )
    diagnose.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for the auxiliary null",
    )
    diagnose.add_argument(
        "-o",
        "--out",
        help="write the pairwise CSC audit table to CSV",
    )
    diagnose.set_defaults(func=cmd_diagnose)

    # map
    map_parser = subparsers.add_parser(
        "map",
        help="render the pairwise fragmentation map",
    )
    map_parser.add_argument(
        "-i",
        "--insularity",
        help="citation-insularity CSV (default: bundled demo)",
    )
    map_parser.add_argument(
        "-d",
        "--divergence",
        help="pairwise vocabulary-divergence CSV (default: bundled demo)",
    )
    map_parser.add_argument(
        "--labels",
        help="JSON object mapping cluster identifiers to display labels",
    )
    map_parser.add_argument(
        "--corpus",
        default="corpus",
        help="corpus name used in the figure",
    )
    map_parser.add_argument(
        "--no-csc",
        action="store_true",
        help="omit the CSC value from the figure subtitle",
    )
    map_parser.add_argument(
        "-o",
        "--out",
        default="fragmentation_map.png",
        help="output image path",
    )
    map_parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="output resolution",
    )
    map_parser.set_defaults(func=cmd_map)

    # align
    align = subparsers.add_parser(
        "align",
        help="LLM cross-cluster vocabulary alignment",
    )
    align.add_argument(
        "--dry-run",
        action="store_true",
        help="rebuild the summary from a raw CSV without API calls",
    )
    align.add_argument(
        "--raw",
        help="raw alignment CSV used by --dry-run (default: bundled demo)",
    )
    align.add_argument(
        "-t",
        "--terms",
        help="semantic_topics CSV containing cluster and top_terms",
    )
    align.add_argument(
        "--labels",
        help="JSON object mapping cluster identifiers to display labels",
    )
    align.add_argument(
        "--top-k",
        type=int,
        default=20,
        dest="top_k",
        help="maximum ranked terms loaded per cluster",
    )
    align.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        dest="max_pairs",
        help="limit the number of unordered cluster pairs",
    )
    align.add_argument(
        "--min-confidence",
        type=int,
        choices=range(1, 6),
        default=3,
        dest="min_confidence",
        help="minimum confidence included in consensus",
    )
    align.add_argument(
        "--min-models",
        type=int,
        default=2,
        dest="min_models",
        help="minimum number of distinct agreeing models",
    )
    align.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="seconds to wait after each provider call",
    )
    align.add_argument(
        "--checkpoint",
        help="JSONL checkpoint path for live alignment",
    )
    align.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore an existing checkpoint and run all requested calls",
    )
    align.add_argument(
        "--retry-failed",
        action="store_true",
        help="retry checkpointed calls whose previous status was failed",
    )
    align.add_argument(
        "--max-retries",
        type=int,
        default=3,
        dest="max_retries",
        help="maximum provider attempts per model and cluster pair",
    )
    align.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="HTTP timeout in seconds",
    )
    align.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="provider sampling temperature",
    )
    align.add_argument(
        "--max-tokens",
        type=int,
        default=1800,
        dest="max_tokens",
        help="maximum response tokens",
    )
    align.add_argument(
        "--raw-out",
        dest="raw_out",
        help="write live raw responses to CSV",
    )
    align.add_argument(
        "-o",
        "--out",
        help="write the consensus summary to CSV",
    )
    align.set_defaults(func=cmd_align)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        print(
            f"aws-align {__version__} — "
            "vocabulary-fragmentation diagnostic"
        )
        print(
            "subcommands: diagnose | map | align "
            "(try 'aws-align <subcommand> --help')"
        )
        return 0

    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
