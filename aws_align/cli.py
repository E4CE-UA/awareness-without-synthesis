"""
aws_align.cli — command-line interface.

Subcommands
-----------
    aws-align diagnose  compute CSC = 1 - (S_cross * D_bar) + null model
    aws-align map       render the fragmentation map figure
    aws-align align     LLM cross-cluster vocabulary alignment table
                        (--dry-run reproduces the table offline from a raw CSV)

Run `aws-align <subcommand> --help` for options. With no arguments it prints
a short banner. Pointing any subcommand at the bundled demo data reproduces
the plastics/recycling canonical numbers (CSC = 0.402).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__

# Bundled demo directory (…/aws_align/../data/demo)
DEMO = Path(__file__).resolve().parent.parent / "data" / "demo"


# ----------------------------------------------------------------------------- 
def _resolve(path, demo_name):
    """Return `path` if given, else the bundled demo file."""
    if path:
        return path
    p = DEMO / demo_name
    if not p.is_file():
        raise SystemExit(f"error: no --input given and demo file missing: {p}")
    return str(p)


def cmd_diagnose(args):
    from .csc import compute_csc
    from .io import load_cluster_sizes, load_divergence, load_insularity

    ins = load_insularity(_resolve(args.insularity, "citation_cluster_insularity.csv"))
    div = load_divergence(_resolve(args.divergence, "rbo_fragmentation.csv"))
    sizes = None
    sizes_path = args.sizes or (str(DEMO / "paper_topics.csv") if (DEMO / "paper_topics.csv").is_file() else None)
    if sizes_path:
        try:
            sizes = load_cluster_sizes(sizes_path)
        except Exception as e:
            print(f"[warn] could not load cluster sizes ({e}); null model skipped", file=sys.stderr)

    res = compute_csc(ins, div, corpus=args.corpus, cluster_sizes=sizes,
                      n_nulls=args.n_nulls, seed=args.seed)

    print("=" * 60)
    print(f"AWS-align diagnostic — corpus: {res.corpus}")
    print("=" * 60)
    print(f"  clusters           : {res.n_clusters}")
    print(f"  E_within / E_cross : {res.E_within} / {res.E_cross}")
    print(f"  S_cross (coupling) : {res.S_cross:.4f}")
    print(f"  D_bar   (divergence): {res.D_bar:.4f}")
    print(f"  AWS = S_cross*D_bar : {res.aws_score:.4f}")
    print(f"  CSC = 1 - AWS       : {res.csc:.4f}")
    if res.mu_null is not None:
        print(f"  null S_cross        : {res.mu_null:.4f} +/- {res.sigma_null:.4f}"
              f"  (z = {res.z_scross:.1f}, p = {res.p_value:.2g}, n={res.n_nulls})")
    print("=" * 60)
    if args.out:
        res.pairwise.to_csv(args.out, index=False)
        print(f"pairwise table -> {args.out}")


def cmd_map(args):
    from .fragmentation import plot_fragmentation_map
    from .io import load_divergence, load_insularity

    ins = load_insularity(_resolve(args.insularity, "citation_cluster_insularity.csv"))
    div = load_divergence(_resolve(args.divergence, "rbo_fragmentation.csv"))
    labels = None
    if args.labels:
        import json
        labels = {int(k): v for k, v in json.loads(Path(args.labels).read_text()).items()}
    csc = None
    if not args.no_csc:
        from .csc import compute_csc
        csc = compute_csc(ins, div, corpus=args.corpus, n_nulls=0).csc
    plot_fragmentation_map(ins, div, cluster_labels=labels, corpus=args.corpus,
                           csc=csc, outfile=args.out, dpi=args.dpi)
    print(f"fragmentation map -> {args.out}")


def cmd_align(args):
    from .align import (alignment_report, consensus_from_raw, load_cluster_terms,
                        load_env)

    if args.dry_run:
        raw = _resolve(args.raw, "alignment_raw.csv")
        summary = consensus_from_raw(raw, min_confidence=args.min_confidence,
                                     min_models=args.min_models)
        print(alignment_report(summary))
    else:
        from .align import align_vocabulary
        load_env()
        terms = load_cluster_terms(_resolve(args.terms, "semantic_topics.csv"),
                                   top_k=args.top_k)
        labels = None
        if args.labels:
            import json
            labels = {int(k): v for k, v in json.loads(Path(args.labels).read_text()).items()}
        raw, summary = align_vocabulary(terms, cluster_labels=labels,
                                        max_pairs=args.max_pairs,
                                        min_confidence=args.min_confidence,
                                        min_models=args.min_models)
        if args.raw_out:
            raw.to_csv(args.raw_out, index=False)
            print(f"raw responses -> {args.raw_out}")
        print(alignment_report(summary))
    if args.out:
        summary.to_csv(args.out, index=False)
        print(f"summary table -> {args.out}")


# ----------------------------------------------------------------------------- 
def build_parser():
    p = argparse.ArgumentParser(
        prog="aws-align",
        description="Diagnose hidden vocabulary fragmentation in scholarly knowledge graphs.",
    )
    p.add_argument("--version", action="version", version=f"aws-align {__version__}")
    sub = p.add_subparsers(dest="command")

    # diagnose
    d = sub.add_parser("diagnose", help="compute CSC + null model")
    d.add_argument("-i", "--insularity", help="citation insularity CSV (default: demo)")
    d.add_argument("-d", "--divergence", help="vocabulary divergence CSV (default: demo)")
    d.add_argument("-s", "--sizes", help="cluster-sizes CSV for null model (default: demo)")
    d.add_argument("--corpus", default="corpus", help="corpus name for labels")
    d.add_argument("--n-nulls", type=int, default=1000, dest="n_nulls")
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("-o", "--out", help="write pairwise table to CSV")
    d.set_defaults(func=cmd_diagnose)

    # map
    m = sub.add_parser("map", help="render fragmentation map figure")
    m.add_argument("-i", "--insularity", help="citation insularity CSV (default: demo)")
    m.add_argument("-d", "--divergence", help="vocabulary divergence CSV (default: demo)")
    m.add_argument("--labels", help="JSON {cluster_id: label} for annotations")
    m.add_argument("--corpus", default="corpus")
    m.add_argument("--no-csc", action="store_true", help="skip CSC subtitle")
    m.add_argument("-o", "--out", default="fragmentation_map.png")
    m.add_argument("--dpi", type=int, default=300)
    m.set_defaults(func=cmd_map)

    # align
    a = sub.add_parser("align", help="LLM cross-cluster vocabulary alignment")
    a.add_argument("--dry-run", action="store_true",
                   help="reproduce table offline from a raw CSV (no API calls)")
    a.add_argument("--raw", help="raw LLM responses CSV for --dry-run (default: demo)")
    a.add_argument("-t", "--terms", help="semantic_topics CSV (cluster, top_terms)")
    a.add_argument("--labels", help="JSON {cluster_id: label}")
    a.add_argument("--top-k", type=int, default=20, dest="top_k")
    a.add_argument("--max-pairs", type=int, default=None, dest="max_pairs")
    a.add_argument("--min-confidence", type=int, default=3, dest="min_confidence")
    a.add_argument("--min-models", type=int, default=2, dest="min_models")
    a.add_argument("--raw-out", dest="raw_out", help="write raw responses to CSV")
    a.add_argument("-o", "--out", help="write consensus summary to CSV")
    a.set_defaults(func=cmd_align)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        print(f"aws-align {__version__} — vocabulary-fragmentation diagnostic")
        print("subcommands: diagnose | map | align   (try 'aws-align <cmd> --help')")
        return 0
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
