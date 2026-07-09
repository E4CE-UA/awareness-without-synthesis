"""
aws_align.align — LLM-proposed vocabulary alignment across clusters.

Given the top characteristic terms of two clusters, one or more LLMs propose
term pairs that refer to related/equivalent concepts despite different wording.
A consensus alignment is a term pair proposed by >= `min_models` models at
confidence >= `min_confidence`.

The module has two entry points:

  align_vocabulary(...)   run the LLMs live (needs OPENROUTER_API_KEY)
  consensus_from_raw(...) recompute the alignment table from a saved raw CSV
                          — the reproducible / offline path (no API calls).

"""
from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- 
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_MODELS = [
    {"key": "gpt4o", "model": "openai/gpt-4o", "label": "GPT-4o"},
    {"key": "claude_haiku", "model": "anthropic/claude-haiku-4.5", "label": "Claude Haiku"},
    {"key": "llama_70b", "model": "meta-llama/llama-3.1-70b-instruct", "label": "Llama-3.1-70B"},
]

SYSTEM_PROMPT = (
    "You are an expert in scientific terminology and interdisciplinary research. "
    "Your task is to identify terminological equivalences and near-equivalences "
    "between two research subfields that study related phenomena but use "
    "different vocabulary."
)

USER_PROMPT_TEMPLATE = """
Two research clusters within one literature corpus use different specialized vocabularies.

Cluster A: "{label_a}"
Top characteristic terms (ranked by c-TF-IDF importance):
{terms_a}

Cluster B: "{label_b}"
Top characteristic terms (ranked by c-TF-IDF importance):
{terms_b}

Task: Identify pairs of terms (one from each cluster) that refer to related or
equivalent scientific concepts, even though they use different vocabulary.

For each proposed alignment, rate confidence (1-5):
1 = weak analogy   2 = partial overlap   3 = functional equivalent
4 = near-synonym   5 = exact equivalent

Respond ONLY with a valid JSON object and nothing else:
{{
  "alignments": [
    {{"term_a": "<term A>", "term_b": "<term B>", "confidence": <1-5>, "reasoning": "<brief>"}}
  ],
  "n_alignable": <integer>,
  "barrier_assessment": "<one sentence>"
}}

List up to 10 alignments, ordered by confidence (highest first).
""".strip()


# ----------------------------------------------------------------------------- 
# Environment / credentials
# -----------------------------------------------------------------------------
def load_env(start: Optional[Path] = None, max_up: int = 4) -> Optional[Path]:
    """
    Load a .env file by walking up from `start` (default: cwd) up to `max_up`
    parent directories. Returns the path loaded, or None. Uses python-dotenv if
    available, else a minimal parser. Never prints key values.
    """
    start = Path(start or Path.cwd()).resolve()
    candidates = [start] + list(start.parents)[:max_up]
    for d in candidates:
        env = d / ".env"
        if env.is_file():
            try:
                from dotenv import load_dotenv

                load_dotenv(env, override=False)
            except Exception:
                for line in env.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return env
    return None


def _call_openrouter(model: str, system: str, user: str, max_retries: int = 3):
    import requests

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY not set (call load_env() or export it)")
    for attempt in range(max_retries):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1500,
                },
                timeout=60,
            )
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"].strip()
                s, e = content.find("{"), content.rfind("}") + 1
                if s >= 0 and e > s:
                    content = content[s:e]
                return json.loads(content)
            elif r.status_code == 429:
                time.sleep(2 ** (attempt + 1))
            else:
                time.sleep(2)
        except Exception:
            time.sleep(3)
    return None


# ----------------------------------------------------------------------------- 
# Term loading
# -----------------------------------------------------------------------------
def load_cluster_terms(semantic_topics_csv, top_k: int = 20) -> dict:
    """Load {cluster_id: [terms]} from a semantic_topics.csv (cluster, top_terms)."""
    df = pd.read_csv(semantic_topics_csv)
    out = {}
    for _, row in df.iterrows():
        cid = int(row["cluster"])
        raw = str(row["top_terms"])
        terms = [t.strip() for t in raw.replace(";", ",").split(",") if t.strip()]
        out[cid] = terms[:top_k]
    return out


# ----------------------------------------------------------------------------- 
# Consensus computation (shared by live + offline paths)
# -----------------------------------------------------------------------------
def consensus_from_raw(
    raw: "pd.DataFrame | str | Path",
    min_confidence: int = 3,
    min_models: int = 2,
) -> pd.DataFrame:
    """
    Compute the per-pair consensus alignment table from raw LLM responses.

    A consensus alignment = a (term_a, term_b) pair proposed by >= min_models
    distinct models at confidence >= min_confidence.

    Returns one row per cluster pair with counts, mean confidence, mean number
    of alignable terms, and the top consensus alignments as a string.
    """
    df = raw if isinstance(raw, pd.DataFrame) else pd.read_csv(raw)
    df = df[df["confidence"].notna() & (df["confidence"] != "") & (df["reasoning"] != "FAILED")].copy()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["confidence"])

    rows = []
    for (ca, cb), grp in df.groupby(["cluster_a", "cluster_b"]):
        strong = grp[grp["confidence"] >= min_confidence]
        if len(strong):
            counts = (
                strong.groupby(["term_a", "term_b"])
                .agg(
                    n_models=("model_key", "nunique"),
                    mean_conf=("confidence", "mean"),
                    models=("model_label", lambda x: "; ".join(sorted(set(x)))),
                )
                .reset_index()
            )
            consensus = counts[counts["n_models"] >= min_models].sort_values(
                "mean_conf", ascending=False
            )
        else:
            consensus = pd.DataFrame()

        n_alignable = grp.drop_duplicates("model_key")["n_alignable"].dropna()
        rows.append(
            {
                "cluster_a": int(ca),
                "cluster_b": int(cb),
                "label_a": grp["label_a"].iloc[0],
                "label_b": grp["label_b"].iloc[0],
                "total_alignments": len(grp),
                "strong_alignments": len(strong),
                "consensus_alignments": len(consensus),
                "mean_confidence": round(float(grp["confidence"].mean()), 2),
                "mean_n_alignable": round(float(n_alignable.mean()), 1) if len(n_alignable) else 0.0,
                "top_consensus": "; ".join(
                    f"{r.term_a}<->{r.term_b}({r.mean_conf:.1f})"
                    for r in consensus.head(3).itertuples()
                )
                if len(consensus)
                else "NONE",
                "barrier_sample": grp.drop_duplicates("model_key")["barrier_assessment"].iloc[0]
                if len(grp)
                else "",
            }
        )
    return pd.DataFrame(rows).sort_values("consensus_alignments").reset_index(drop=True)


def alignment_report(summary: pd.DataFrame, models: Optional[Sequence[dict]] = None) -> str:
    """Render a text report from a consensus summary table."""
    models = models or DEFAULT_MODELS
    total = len(summary)
    no_cons = int((summary["consensus_alignments"] == 0).sum())
    with_cons = total - no_cons
    mean_conf = summary["mean_confidence"].mean()
    mean_align = summary["mean_n_alignable"].mean()
    lines = [
        "=" * 70,
        "LLM VOCABULARY ALIGNMENT: CROSS-CLUSTER TERMINOLOGICAL BRIDGES",
        "=" * 70,
        "",
        f"Cluster pairs analysed: {total}",
        f"Models: {', '.join(m['label'] for m in models)}",
        "",
        f"Pairs with consensus (>= 2 models agree): {with_cons}/{total}",
        f"Pairs with NO consensus:                  {no_cons}/{total}",
        f"Mean confidence across alignments:        {mean_conf:.2f}/5",
        f"Mean alignable terms per pair:            {mean_align:.1f}/20",
        "",
        "── RESULTS BY PAIR ──",
        "",
        summary[
            ["cluster_a", "cluster_b", "consensus_alignments", "mean_confidence",
             "mean_n_alignable", "top_consensus"]
        ].to_string(index=False),
        "",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- 
# Live path
# -----------------------------------------------------------------------------
def align_vocabulary(
    cluster_terms: dict,
    cluster_labels: Optional[dict] = None,
    models: Optional[Sequence[dict]] = None,
    max_pairs: Optional[int] = None,
    sleep: float = 1.0,
    min_confidence: int = 3,
    min_models: int = 2,
):
    """
    Run LLM vocabulary alignment live over all cluster pairs.

    Returns (raw_df, summary_df). Requires OPENROUTER_API_KEY in the environment
    and the optional 'llm' extra (requests). For a reproducible offline run,
    use consensus_from_raw() on a saved raw CSV instead.
    """
    models = list(models or DEFAULT_MODELS)
    cluster_labels = cluster_labels or {}
    ids = sorted(cluster_terms)
    pairs = list(itertools.combinations(ids, 2))
    if max_pairs:
        pairs = pairs[:max_pairs]

    records = []
    for ca, cb in pairs:
        ta = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(cluster_terms.get(ca, [])))
        tb = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(cluster_terms.get(cb, [])))
        la = cluster_labels.get(ca, f"Cluster {ca}")
        lb = cluster_labels.get(cb, f"Cluster {cb}")
        user = USER_PROMPT_TEMPLATE.format(label_a=la, label_b=lb, terms_a=ta, terms_b=tb)
        for m in models:
            res = _call_openrouter(m["model"], SYSTEM_PROMPT, user)
            if res is None:
                records.append(dict(cluster_a=ca, cluster_b=cb, label_a=la, label_b=lb,
                                    model_key=m["key"], model_label=m["label"], term_a="",
                                    term_b="", confidence="", reasoning="FAILED",
                                    n_alignable="", barrier_assessment="FAILED"))
            else:
                aligns = res.get("alignments", []) or [{}]
                for a in aligns:
                    records.append(dict(
                        cluster_a=ca, cluster_b=cb, label_a=la, label_b=lb,
                        model_key=m["key"], model_label=m["label"],
                        term_a=a.get("term_a", ""), term_b=a.get("term_b", ""),
                        confidence=a.get("confidence", 0),
                        reasoning=a.get("reasoning", "No alignments found" if not a else ""),
                        n_alignable=res.get("n_alignable", len(res.get("alignments", []))),
                        barrier_assessment=res.get("barrier_assessment", ""),
                    ))
            time.sleep(sleep)
    raw = pd.DataFrame(records)
    summary = consensus_from_raw(raw, min_confidence=min_confidence, min_models=min_models)
    return raw, summary
