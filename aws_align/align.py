"""
aws_align.align — LLM-proposed vocabulary alignment across clusters.

Given the ranked characteristic terms of two clusters, multiple LLMs propose
cross-cluster term relations. Consensus is defined as the same observed term
pair being proposed by at least ``min_models`` distinct models at confidence
greater than or equal to ``min_confidence``.

Public entry points
-------------------
align_vocabulary(...)
    Run the configured LLMs live through OpenRouter. Returns the historical
    two-object API: ``(raw_df, summary_df)``.

consensus_from_raw(...)
    Rebuild the consensus summary from a saved raw CSV or DataFrame without
    making API calls. Historical raw CSVs without ``relation_type`` remain
    supported.

alignment_report(...)
    Render a compact text report from the summary table.

load_cluster_terms(...)
    Load ranked cluster vocabularies from ``semantic_topics.csv``.

Interpretation
--------------
Cross-model consensus is evidence of agreement among the configured models; it
does not by itself establish that two expressions are lexically
interchangeable. ``query_expansion_safe`` is therefore restricted to
high-confidence exact equivalents and near-synonyms.

Reproducibility
---------------
Hosted model outputs are best-effort reproducible rather than bit-exact.
Temperature 0, exact model identifiers, prompt hashes, timestamps, raw provider
content, and optional JSONL checkpoints are recorded for auditability.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd


PROMPT_VERSION = "aws-align-v2.0"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1800

ALLOWED_RELATION_TYPES = {
    "exact_equivalent",
    "near_synonym",
    "functional_equivalent",
    "related_but_not_interchangeable",
}

QUERY_EXPANSION_SAFE_RELATIONS = {
    "exact_equivalent",
    "near_synonym",
}


# Keep dictionaries rather than a custom class so the existing CLI and user
# code can continue to access m["key"], m["model"], and m["label"].
DEFAULT_MODELS: List[Dict[str, str]] = [
    {
        "key": "gpt4o_mini",
        "model": "openai/gpt-4o-mini",
        "label": "GPT-4o-mini",
    },
    {
        "key": "claude_haiku",
        "model": "anthropic/claude-haiku-4.5",
        "label": "Claude Haiku",
    },
    {
        "key": "llama_70b",
        "model": "meta-llama/llama-3.1-70b-instruct",
        "label": "Llama-3.1-70B",
    },
]


SYSTEM_PROMPT = (
    "You are an expert in scientific terminology and interdisciplinary "
    "research. Compare only the terms explicitly supplied in the two ranked "
    "lists. Do not invent, paraphrase, translate, stem, merge, or introduce "
    "terms absent from those lists. Classify semantic relations conservatively."
)


USER_PROMPT_TEMPLATE = """
Two research clusters within one literature corpus use different specialised
vocabularies.

Cluster A: "{label_a}"
Top characteristic terms from Cluster A:
{terms_a}

Cluster B: "{label_b}"
Top characteristic terms from Cluster B:
{terms_b}

Identify up to 10 meaningful cross-cluster term pairs. Each term must be copied
exactly from the corresponding list above.

For every proposed pair, assign exactly one relation type:

- exact_equivalent:
  The terms denote the same concept and are normally interchangeable.
- near_synonym:
  The terms denote almost the same concept, with only a narrow contextual or
  disciplinary difference.
- functional_equivalent:
  The terms play a similar role or function but are not lexical substitutes.
- related_but_not_interchangeable:
  The concepts are scientifically related but should not replace one another
  in a query.

Rate confidence from 1 to 5:

1 = weak or uncertain relation
2 = plausible but limited overlap
3 = defensible functional relation
4 = strong near-equivalence
5 = exact or effectively exact equivalence

Return only one valid JSON object:

{{
  "alignments": [
    {{
      "term_a": "<exact term copied from Cluster A>",
      "term_b": "<exact term copied from Cluster B>",
      "relation_type": "<one allowed relation type>",
      "confidence": <integer from 1 to 5>,
      "reasoning": "<brief domain-grounded explanation>"
    }}
  ],
  "n_alignable": <integer>,
  "barrier_assessment": "<one concise sentence>"
}}

Return an empty "alignments" list when no defensible pair exists.
""".strip()


RAW_COLUMNS = [
    "cluster_a",
    "cluster_b",
    "label_a",
    "label_b",
    "model_key",
    "model_label",
    "model_requested",
    "model_returned",
    "provider",
    "request_timestamp_utc",
    "prompt_version",
    "prompt_sha256",
    "temperature",
    "max_tokens",
    "http_status",
    "attempts",
    "response_id",
    "status",
    "valid_alignment",
    "validation_error",
    "term_a_raw",
    "term_b_raw",
    "term_a",
    "term_b",
    "term_a_norm",
    "term_b_norm",
    "relation_type",
    "confidence",
    "reasoning",
    "query_expansion_safe",
    "n_alignable",
    "barrier_assessment",
    "raw_content",
    "error",
]


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_term(value: Any) -> str:
    """Return a conservative Unicode/case/whitespace normalisation."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.casefold().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _coerce_cluster_id(value: Any) -> Union[int, str]:
    """
    Preserve ordinary string identifiers while converting integer-like values.

    Examples:
        2, "2", "2.0" -> 2
        "C2"          -> "C2"
    """
    if pd.isna(value):
        raise ValueError("cluster identifier cannot be missing")

    text = str(value).strip()
    if not text:
        raise ValueError("cluster identifier cannot be empty")

    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)

    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return int(float(text))

    return text


def _relation_from_confidence(confidence: Any) -> str:
    """
    Backward-compatible relation inference for historical CSVs.

    Historical data encoded relation strength only through confidence:
      5 -> exact equivalent
      4 -> near-synonym
      3 -> functional equivalent
      1-2 -> related but not interchangeable
    """
    try:
        value = int(float(confidence))
    except (TypeError, ValueError):
        return "related_but_not_interchangeable"

    if value >= 5:
        return "exact_equivalent"
    if value >= 4:
        return "near_synonym"
    if value >= 3:
        return "functional_equivalent"
    return "related_but_not_interchangeable"


def _normalise_relation_type(value: Any, confidence: Any) -> str:
    relation = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "exact": "exact_equivalent",
        "equivalent": "exact_equivalent",
        "exact_synonym": "exact_equivalent",
        "near_equivalent": "near_synonym",
        "near_synonymy": "near_synonym",
        "functional": "functional_equivalent",
        "related": "related_but_not_interchangeable",
        "partial_overlap": "related_but_not_interchangeable",
        "weak_analogy": "related_but_not_interchangeable",
    }
    relation = aliases.get(relation, relation)

    if relation not in ALLOWED_RELATION_TYPES:
        relation = _relation_from_confidence(confidence)

    return relation


def _model_value(model: Mapping[str, Any], field: str) -> str:
    value = model.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(f"model specification is missing {field!r}: {model}")
    return str(value).strip()


def _validate_models(
    models: Optional[Sequence[Mapping[str, Any]]],
) -> List[Dict[str, str]]:
    selected = list(models or DEFAULT_MODELS)
    if not selected:
        raise ValueError("at least one model must be configured")

    normalised: List[Dict[str, str]] = []
    seen_keys = set()

    for model in selected:
        item = {
            "key": _model_value(model, "key"),
            "model": _model_value(model, "model"),
            "label": _model_value(model, "label"),
        }
        if item["key"] in seen_keys:
            raise ValueError(f"duplicate model key: {item['key']}")
        seen_keys.add(item["key"])
        normalised.append(item)

    return normalised


def _model_labels(models: Sequence[Mapping[str, Any]]) -> List[str]:
    return [str(model.get("label", model.get("model", "unknown"))) for model in models]


def _extract_json_object(content: str) -> Dict[str, Any]:
    """
    Parse a JSON object, tolerating provider-added Markdown fences or preamble.
    """
    text = str(content or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("model response JSON must be an object")
        return parsed
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response did not contain a JSON object")

        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("model response JSON must be an object")
        return parsed


def _prompt_hash(system_prompt: str, user_prompt: str) -> str:
    payload = (system_prompt + "\n\n" + user_prompt).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


# -----------------------------------------------------------------------------
# Environment / credentials
# -----------------------------------------------------------------------------
def load_env(start: Optional[Path] = None, max_up: int = 4) -> Optional[Path]:
    """
    Load the nearest .env file without printing secret values.

    python-dotenv is used when installed. The fallback parser supports ordinary
    KEY=VALUE lines and optional leading ``export``.
    """
    start_path = Path(start or Path.cwd()).resolve()
    candidates = [start_path] + list(start_path.parents)[:max_up]

    for directory in candidates:
        env_path = directory / ".env"
        if not env_path.is_file():
            continue

        try:
            from dotenv import load_dotenv

            load_dotenv(env_path, override=False)
        except ImportError:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)

        return env_path

    return None


# -----------------------------------------------------------------------------
# OpenRouter
# -----------------------------------------------------------------------------
def _call_openrouter(
    model: str,
    system: str,
    user: str,
    max_retries: int = 3,
    timeout: float = 90.0,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:
    """
    Call OpenRouter and return an auditable result dictionary.

    This function never returns or logs the API key.
    """
    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "Live alignment requires requests. Install the optional LLM "
            "dependencies before calling align_vocabulary()."
        ) from exc

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Call load_env() or export it."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    title = os.getenv("OPENROUTER_APP_TITLE", "awareness-without-synthesis")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    last_error = ""
    last_status: Optional[int] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            last_status = int(response.status_code)

            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    raise ValueError("provider response contained no choices")

                content = choices[0].get("message", {}).get("content", "")
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)

                parsed = _extract_json_object(content)

                return {
                    "status": "ok",
                    "http_status": last_status,
                    "attempts": attempt,
                    "parsed": parsed,
                    "raw_content": content,
                    "response_id": _safe_text(data.get("id")),
                    "model_returned": _safe_text(data.get("model")),
                    "provider": _safe_text(data.get("provider")),
                    "usage": data.get("usage") or {},
                    "error": "",
                }

            if response.status_code == 429 or response.status_code >= 500:
                last_error = (
                    f"HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )
                if attempt < max_retries:
                    time.sleep(min(2 ** attempt, 30))
                continue

            # Non-retryable 4xx response.
            return {
                "status": "failed",
                "http_status": last_status,
                "attempts": attempt,
                "parsed": None,
                "raw_content": response.text[:5000],
                "response_id": "",
                "model_returned": "",
                "provider": "",
                "usage": {},
                "error": (
                    f"non-retryable HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                ),
            }

        except Exception as exc:  # network, JSON, and schema failures
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 30))

    return {
        "status": "failed",
        "http_status": last_status,
        "attempts": max_retries,
        "parsed": None,
        "raw_content": "",
        "response_id": "",
        "model_returned": "",
        "provider": "",
        "usage": {},
        "error": last_error or "unknown provider error",
    }


# -----------------------------------------------------------------------------
# Term loading
# -----------------------------------------------------------------------------
def load_cluster_terms(
    semantic_topics_csv: Union[str, Path],
    top_k: int = 20,
) -> Dict[Union[int, str], List[str]]:
    """
    Load ``{cluster_id: [ranked terms]}`` from semantic_topics.csv.

    Required columns:
        cluster, top_terms

    ``top_terms`` may be separated by semicolons or commas. Duplicate terms are
    removed after conservative normalisation while preserving rank order.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    frame = pd.read_csv(semantic_topics_csv)
    required = {"cluster", "top_terms"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"semantic topics file is missing columns: {sorted(missing)}"
        )

    output: Dict[Union[int, str], List[str]] = {}

    for _, row in frame.iterrows():
        cluster_id = _coerce_cluster_id(row["cluster"])
        raw_terms = _safe_text(row["top_terms"])
        candidates = [
            term.strip()
            for term in re.split(r"[;,]", raw_terms)
            if term.strip()
        ]

        unique_terms: List[str] = []
        seen = set()
        for term in candidates:
            norm = _normalise_term(term)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            unique_terms.append(term)
            if len(unique_terms) >= top_k:
                break

        if not unique_terms:
            raise ValueError(f"cluster {cluster_id!r} has no valid terms")

        output[cluster_id] = unique_terms

    if len(output) < 2:
        raise ValueError("at least two clusters are required for alignment")

    return output


# -----------------------------------------------------------------------------
# Alignment validation
# -----------------------------------------------------------------------------
def _validate_one_alignment(
    alignment: Mapping[str, Any],
    terms_a_lookup: Mapping[str, str],
    terms_b_lookup: Mapping[str, str],
) -> Dict[str, Any]:
    term_a_raw = _safe_text(alignment.get("term_a"))
    term_b_raw = _safe_text(alignment.get("term_b"))
    term_a_norm = _normalise_term(term_a_raw)
    term_b_norm = _normalise_term(term_b_raw)

    errors: List[str] = []

    if not term_a_norm:
        errors.append("term_a is empty")
    elif term_a_norm not in terms_a_lookup:
        errors.append("term_a is absent from Cluster A list")

    if not term_b_norm:
        errors.append("term_b is empty")
    elif term_b_norm not in terms_b_lookup:
        errors.append("term_b is absent from Cluster B list")

    try:
        confidence = int(float(alignment.get("confidence")))
    except (TypeError, ValueError):
        confidence = 0
        errors.append("confidence is not numeric")

    if confidence < 1 or confidence > 5:
        errors.append("confidence is outside 1-5")

    relation_type = _normalise_relation_type(
        alignment.get("relation_type"),
        confidence,
    )

    reasoning = _safe_text(alignment.get("reasoning"))
    if not reasoning:
        errors.append("reasoning is empty")

    valid = not errors

    canonical_a = terms_a_lookup.get(term_a_norm, term_a_raw)
    canonical_b = terms_b_lookup.get(term_b_norm, term_b_raw)

    return {
        "valid_alignment": valid,
        "validation_error": "; ".join(errors),
        "term_a_raw": term_a_raw,
        "term_b_raw": term_b_raw,
        "term_a": canonical_a if valid else term_a_raw,
        "term_b": canonical_b if valid else term_b_raw,
        "term_a_norm": term_a_norm,
        "term_b_norm": term_b_norm,
        "relation_type": relation_type,
        "confidence": confidence if confidence else "",
        "reasoning": reasoning,
        "query_expansion_safe": bool(
            valid
            and confidence >= 4
            and relation_type in QUERY_EXPANSION_SAFE_RELATIONS
        ),
    }


def _empty_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=RAW_COLUMNS)


def _response_base(
    cluster_a: Union[int, str],
    cluster_b: Union[int, str],
    label_a: str,
    label_b: str,
    model: Mapping[str, str],
    request_timestamp: str,
    prompt_sha256: str,
    temperature: float,
    max_tokens: int,
    call_result: Mapping[str, Any],
    n_alignable: Any,
    barrier_assessment: str,
) -> Dict[str, Any]:
    return {
        "cluster_a": cluster_a,
        "cluster_b": cluster_b,
        "label_a": label_a,
        "label_b": label_b,
        "model_key": model["key"],
        "model_label": model["label"],
        "model_requested": model["model"],
        "model_returned": _safe_text(call_result.get("model_returned")),
        "provider": _safe_text(call_result.get("provider")),
        "request_timestamp_utc": request_timestamp,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "http_status": call_result.get("http_status"),
        "attempts": call_result.get("attempts"),
        "response_id": _safe_text(call_result.get("response_id")),
        "n_alignable": n_alignable,
        "barrier_assessment": barrier_assessment,
        "raw_content": _safe_text(call_result.get("raw_content")),
        "error": _safe_text(call_result.get("error")),
    }


def _flatten_call_result(
    cluster_a: Union[int, str],
    cluster_b: Union[int, str],
    label_a: str,
    label_b: str,
    model: Mapping[str, str],
    terms_a: Sequence[str],
    terms_b: Sequence[str],
    request_timestamp: str,
    prompt_sha256: str,
    temperature: float,
    max_tokens: int,
    call_result: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    parsed = call_result.get("parsed")
    if not isinstance(parsed, dict):
        parsed = {}

    n_alignable_raw = parsed.get("n_alignable", "")
    try:
        n_alignable: Union[int, str] = max(0, int(float(n_alignable_raw)))
    except (TypeError, ValueError):
        n_alignable = ""

    barrier_assessment = _safe_text(parsed.get("barrier_assessment"))

    base = _response_base(
        cluster_a=cluster_a,
        cluster_b=cluster_b,
        label_a=label_a,
        label_b=label_b,
        model=model,
        request_timestamp=request_timestamp,
        prompt_sha256=prompt_sha256,
        temperature=temperature,
        max_tokens=max_tokens,
        call_result=call_result,
        n_alignable=n_alignable,
        barrier_assessment=barrier_assessment,
    )

    if call_result.get("status") != "ok":
        row = dict(base)
        row.update(
            {
                "status": "failed",
                "valid_alignment": False,
                "validation_error": _safe_text(call_result.get("error")),
                "term_a_raw": "",
                "term_b_raw": "",
                "term_a": "",
                "term_b": "",
                "term_a_norm": "",
                "term_b_norm": "",
                "relation_type": "",
                "confidence": "",
                "reasoning": "FAILED",
                "query_expansion_safe": False,
            }
        )
        return [row]

    alignments = parsed.get("alignments", [])
    if not isinstance(alignments, list):
        alignments = []

    terms_a_lookup = {_normalise_term(term): term for term in terms_a}
    terms_b_lookup = {_normalise_term(term): term for term in terms_b}

    rows: List[Dict[str, Any]] = []
    for proposed in alignments[:10]:
        if not isinstance(proposed, dict):
            proposed = {}

        validated = _validate_one_alignment(
            proposed,
            terms_a_lookup=terms_a_lookup,
            terms_b_lookup=terms_b_lookup,
        )

        row = dict(base)
        row.update(validated)
        row["status"] = (
            "ok_alignment"
            if validated["valid_alignment"]
            else "rejected_alignment"
        )
        rows.append(row)

    if not rows:
        row = dict(base)
        row.update(
            {
                "status": "ok_no_alignments",
                "valid_alignment": False,
                "validation_error": "",
                "term_a_raw": "",
                "term_b_raw": "",
                "term_a": "",
                "term_b": "",
                "term_a_norm": "",
                "term_b_norm": "",
                "relation_type": "",
                "confidence": "",
                "reasoning": "No alignments proposed",
                "query_expansion_safe": False,
            }
        )
        rows.append(row)

    return rows


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------
def _checkpoint_key(record: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        str(record.get("cluster_a")),
        str(record.get("cluster_b")),
        str(record.get("model_key")),
    )


def _load_checkpoint_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL checkpoint at line {line_number}: {exc}"
                ) from exc

            if not isinstance(payload, dict):
                raise ValueError(
                    f"checkpoint line {line_number} is not a JSON object"
                )

            raw_rows = payload.get("rows")
            if not isinstance(raw_rows, list):
                continue

            for row in raw_rows:
                if isinstance(row, dict):
                    rows.append(row)

    return rows


def _append_checkpoint(
    path: Path,
    cluster_a: Union[int, str],
    cluster_b: Union[int, str],
    model_key: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cluster_a": cluster_a,
        "cluster_b": cluster_b,
        "model_key": model_key,
        "saved_at_utc": _utc_now(),
        "rows": list(rows),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------------
# Historical/offline raw normalisation
# -----------------------------------------------------------------------------
def _load_raw(
    raw: Union[pd.DataFrame, str, Path],
) -> pd.DataFrame:
    frame = raw.copy() if isinstance(raw, pd.DataFrame) else pd.read_csv(raw)

    required = {
        "cluster_a",
        "cluster_b",
        "model_key",
        "term_a",
        "term_b",
        "confidence",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"raw alignment table is missing columns: {sorted(missing)}"
        )

    output = frame.copy()

    defaults: Dict[str, Any] = {
        "label_a": "",
        "label_b": "",
        "model_label": "",
        "reasoning": "",
        "n_alignable": "",
        "barrier_assessment": "",
        "status": "",
        "valid_alignment": "",
        "validation_error": "",
        "relation_type": "",
        "query_expansion_safe": "",
        "term_a_raw": "",
        "term_b_raw": "",
    }
    for column, default in defaults.items():
        if column not in output.columns:
            output[column] = default

    output["cluster_a"] = output["cluster_a"].map(_coerce_cluster_id)
    output["cluster_b"] = output["cluster_b"].map(_coerce_cluster_id)

    output["confidence"] = pd.to_numeric(
        output["confidence"],
        errors="coerce",
    )

    output["term_a"] = output["term_a"].fillna("").astype(str).str.strip()
    output["term_b"] = output["term_b"].fillna("").astype(str).str.strip()
    output["term_a_norm"] = output["term_a"].map(_normalise_term)
    output["term_b_norm"] = output["term_b"].map(_normalise_term)

    output["relation_type"] = [
        _normalise_relation_type(relation, confidence)
        for relation, confidence in zip(
            output["relation_type"],
            output["confidence"],
        )
    ]

    # Historical rows did not have an explicit status/valid flag.
    historical_valid = (
        output["confidence"].notna()
        & output["confidence"].between(1, 5)
        & output["term_a_norm"].ne("")
        & output["term_b_norm"].ne("")
        & output["reasoning"].fillna("").astype(str).ne("FAILED")
    )

    explicit_valid = output["valid_alignment"].map(
        lambda value: (
            value
            if isinstance(value, bool)
            else str(value).strip().casefold() in {"true", "1", "yes"}
        )
    )

    has_explicit_flag = output["valid_alignment"].astype(str).str.strip().ne("")
    output["valid_alignment"] = historical_valid.where(
        ~has_explicit_flag,
        explicit_valid,
    )

    empty_status = output["status"].fillna("").astype(str).str.strip().eq("")
    output.loc[
        empty_status & output["valid_alignment"],
        "status",
    ] = "historical_alignment"
    output.loc[
        empty_status & ~output["valid_alignment"],
        "status",
    ] = "historical_non_alignment"

    output["query_expansion_safe"] = (
        output["valid_alignment"]
        & output["confidence"].fillna(0).ge(4)
        & output["relation_type"].isin(QUERY_EXPANSION_SAFE_RELATIONS)
    )

    return output


def _deduplicate_model_alignments(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the strongest duplicate from one model for one normalised pair.
    """
    if frame.empty:
        return frame.copy()

    return (
        frame.sort_values(
            ["confidence", "query_expansion_safe"],
            ascending=[False, False],
        )
        .drop_duplicates(
            [
                "cluster_a",
                "cluster_b",
                "model_key",
                "term_a_norm",
                "term_b_norm",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )


def _modal_relation(group: pd.DataFrame) -> str:
    valid = group["relation_type"].dropna().astype(str)
    valid = valid[valid.isin(ALLOWED_RELATION_TYPES)]
    if valid.empty:
        return ""

    counts = valid.value_counts()
    top_count = counts.iloc[0]
    candidates = set(counts[counts == top_count].index)

    # Conservative tie-breaking: prefer the less interchangeable relation.
    priority = [
        "related_but_not_interchangeable",
        "functional_equivalent",
        "near_synonym",
        "exact_equivalent",
    ]
    for relation in priority:
        if relation in candidates:
            return relation
    return str(counts.index[0])


# -----------------------------------------------------------------------------
# Consensus computation
# -----------------------------------------------------------------------------
def consensus_details_from_raw(
    raw: Union[pd.DataFrame, str, Path],
    min_confidence: int = 3,
    min_models: int = 2,
) -> pd.DataFrame:
    """
    Return one row per consensus term pair.

    Historical compatibility is preserved by defining consensus on the
    normalised term pair, as the original implementation did. Relation type is
    reported separately using a conservative modal relation. Safe query
    expansion additionally requires at least ``min_models`` models to support
    an exact equivalent or near-synonym at confidence >= 4.
    """
    if not 1 <= min_confidence <= 5:
        raise ValueError("min_confidence must be between 1 and 5")
    if min_models <= 0:
        raise ValueError("min_models must be positive")

    frame = _load_raw(raw)
    valid = frame[
        frame["valid_alignment"]
        & frame["confidence"].ge(min_confidence)
    ].copy()

    valid = _deduplicate_model_alignments(valid)

    columns = [
        "cluster_a",
        "cluster_b",
        "label_a",
        "label_b",
        "term_a",
        "term_b",
        "term_a_norm",
        "term_b_norm",
        "n_models",
        "mean_confidence",
        "min_confidence",
        "max_confidence",
        "models",
        "relation_type",
        "relation_agreement_models",
        "safe_models",
        "query_expansion_safe",
    ]

    if valid.empty:
        return pd.DataFrame(columns=columns)

    details: List[Dict[str, Any]] = []

    group_columns = [
        "cluster_a",
        "cluster_b",
        "term_a_norm",
        "term_b_norm",
    ]

    for keys, group in valid.groupby(group_columns, sort=True, dropna=False):
        cluster_a, cluster_b, term_a_norm, term_b_norm = keys
        n_models = int(group["model_key"].nunique())
        if n_models < min_models:
            continue

        relation_type = _modal_relation(group)
        relation_agreement_models = int(
            group.loc[
                group["relation_type"].eq(relation_type),
                "model_key",
            ].nunique()
        )

        safe_models = int(
            group.loc[
                group["query_expansion_safe"],
                "model_key",
            ].nunique()
        )

        display_row = group.sort_values(
            ["confidence", "model_key"],
            ascending=[False, True],
        ).iloc[0]

        labels_a = group["label_a"].dropna().astype(str)
        labels_b = group["label_b"].dropna().astype(str)

        details.append(
            {
                "cluster_a": cluster_a,
                "cluster_b": cluster_b,
                "label_a": labels_a.iloc[0] if len(labels_a) else "",
                "label_b": labels_b.iloc[0] if len(labels_b) else "",
                "term_a": display_row["term_a"],
                "term_b": display_row["term_b"],
                "term_a_norm": term_a_norm,
                "term_b_norm": term_b_norm,
                "n_models": n_models,
                "mean_confidence": float(group["confidence"].mean()),
                "min_confidence": float(group["confidence"].min()),
                "max_confidence": float(group["confidence"].max()),
                "models": "; ".join(
                    sorted(
                        set(
                            group["model_label"]
                            .fillna("")
                            .astype(str)
                            .replace("", pd.NA)
                            .dropna()
                        )
                    )
                ),
                "relation_type": relation_type,
                "relation_agreement_models": relation_agreement_models,
                "safe_models": safe_models,
                "query_expansion_safe": bool(safe_models >= min_models),
            }
        )

    result = pd.DataFrame(details, columns=columns)
    if result.empty:
        return result

    return result.sort_values(
        [
            "cluster_a",
            "cluster_b",
            "query_expansion_safe",
            "n_models",
            "mean_confidence",
            "term_a_norm",
            "term_b_norm",
        ],
        ascending=[True, True, False, False, False, True, True],
    ).reset_index(drop=True)


def consensus_from_raw(
    raw: Union[pd.DataFrame, str, Path],
    min_confidence: int = 3,
    min_models: int = 2,
) -> pd.DataFrame:
    """
    Compute the historical per-cluster-pair consensus summary.

    This function remains compatible with the original CLI and archived raw
    CSVs. It returns one row per cluster pair.
    """
    if not 1 <= min_confidence <= 5:
        raise ValueError("min_confidence must be between 1 and 5")
    if min_models <= 0:
        raise ValueError("min_models must be positive")

    frame = _load_raw(raw)
    details = consensus_details_from_raw(
        frame,
        min_confidence=min_confidence,
        min_models=min_models,
    )

    summary_rows: List[Dict[str, Any]] = []

    for (cluster_a, cluster_b), group in frame.groupby(
        ["cluster_a", "cluster_b"],
        sort=True,
        dropna=False,
    ):
        valid = group[group["valid_alignment"]].copy()
        strong = valid[valid["confidence"].ge(min_confidence)].copy()
        strong = _deduplicate_model_alignments(strong)

        pair_details = details[
            details["cluster_a"].eq(cluster_a)
            & details["cluster_b"].eq(cluster_b)
        ]

        n_alignable = pd.to_numeric(
            group.drop_duplicates("model_key")["n_alignable"],
            errors="coerce",
        ).dropna()

        model_status = group.drop_duplicates("model_key")
        models_attempted = int(model_status["model_key"].nunique())
        failed_models = int(
            model_status["status"].fillna("").astype(str).eq("failed").sum()
        )

        confidence_mean = (
            float(valid["confidence"].mean())
            if not valid.empty
            else float("nan")
        )

        top_consensus = "NONE"
        if not pair_details.empty:
            top_rows = pair_details.head(3)
            top_consensus = "; ".join(
                (
                    f"{row.term_a}<->{row.term_b}"
                    f"[{row.relation_type}]"
                    f"({row.mean_confidence:.1f}; n={row.n_models})"
                )
                for row in top_rows.itertuples()
            )

        barrier_values = (
            group.drop_duplicates("model_key")["barrier_assessment"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        barrier_values = barrier_values[barrier_values.ne("")]
        barrier_assessments = " || ".join(barrier_values.tolist())

        labels_a = group["label_a"].dropna().astype(str)
        labels_b = group["label_b"].dropna().astype(str)

        summary_rows.append(
            {
                "cluster_a": cluster_a,
                "cluster_b": cluster_b,
                "label_a": labels_a.iloc[0] if len(labels_a) else "",
                "label_b": labels_b.iloc[0] if len(labels_b) else "",
                "models_attempted": models_attempted,
                "failed_models": failed_models,
                "total_alignments": int(len(valid)),
                "strong_alignments": int(len(strong)),
                "consensus_alignments": int(len(pair_details)),
                "query_expansion_safe_alignments": int(
                    pair_details["query_expansion_safe"].sum()
                    if not pair_details.empty
                    else 0
                ),
                "mean_confidence": (
                    round(confidence_mean, 2)
                    if pd.notna(confidence_mean)
                    else float("nan")
                ),
                "mean_n_alignable": (
                    round(float(n_alignable.mean()), 1)
                    if len(n_alignable)
                    else 0.0
                ),
                "top_consensus": top_consensus,
                "barrier_assessments": barrier_assessments,
            }
        )

    columns = [
        "cluster_a",
        "cluster_b",
        "label_a",
        "label_b",
        "models_attempted",
        "failed_models",
        "total_alignments",
        "strong_alignments",
        "consensus_alignments",
        "query_expansion_safe_alignments",
        "mean_confidence",
        "mean_n_alignable",
        "top_consensus",
        "barrier_assessments",
    ]

    summary = pd.DataFrame(summary_rows, columns=columns)
    if summary.empty:
        return summary

    return summary.sort_values(
        ["consensus_alignments", "cluster_a", "cluster_b"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def alignment_report(
    summary: pd.DataFrame,
    models: Optional[Sequence[Mapping[str, Any]]] = None,
    min_confidence: int = 3,
    min_models: int = 2,
    top_k: int = 20,
) -> str:
    """Render a text report from a consensus summary table."""
    selected_models = _validate_models(models)
    total = int(len(summary))

    if total == 0:
        return (
            "=" * 70
            + "\nLLM VOCABULARY ALIGNMENT\n"
            + "=" * 70
            + "\n\nNo cluster pairs were available.\n"
        )

    no_consensus = int(summary["consensus_alignments"].eq(0).sum())
    with_consensus = total - no_consensus
    safe_pairs = int(
        summary["query_expansion_safe_alignments"].gt(0).sum()
        if "query_expansion_safe_alignments" in summary.columns
        else 0
    )

    mean_confidence = pd.to_numeric(
        summary["mean_confidence"],
        errors="coerce",
    ).mean()
    mean_alignable = pd.to_numeric(
        summary["mean_n_alignable"],
        errors="coerce",
    ).mean()

    mean_confidence_text = (
        f"{mean_confidence:.2f}/5"
        if pd.notna(mean_confidence)
        else "NA"
    )
    mean_alignable_text = (
        f"{mean_alignable:.1f}/{top_k}"
        if pd.notna(mean_alignable)
        else "NA"
    )

    display_columns = [
        "cluster_a",
        "cluster_b",
        "consensus_alignments",
        "query_expansion_safe_alignments",
        "mean_confidence",
        "mean_n_alignable",
        "top_consensus",
    ]
    display_columns = [
        column for column in display_columns if column in summary.columns
    ]

    lines = [
        "=" * 70,
        "LLM VOCABULARY ALIGNMENT: CROSS-CLUSTER TERMINOLOGICAL BRIDGES",
        "=" * 70,
        "",
        f"Cluster pairs analysed: {total}",
        f"Models: {', '.join(_model_labels(selected_models))}",
        (
            f"Consensus threshold: >= {min_models} models at "
            f"confidence >= {min_confidence}"
        ),
        "",
        f"Pairs with consensus:                    {with_consensus}/{total}",
        f"Pairs with no consensus:                 {no_consensus}/{total}",
        f"Pairs with query-expansion-safe bridge:  {safe_pairs}/{total}",
        f"Mean confidence across valid alignments: {mean_confidence_text}",
        f"Mean alignable terms per pair:           {mean_alignable_text}",
        "",
        "RESULTS BY PAIR",
        "",
        summary[display_columns].to_string(index=False),
        "",
    ]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Live path
# -----------------------------------------------------------------------------
def align_vocabulary(
    cluster_terms: Mapping[Union[int, str], Sequence[str]],
    cluster_labels: Optional[Mapping[Union[int, str], str]] = None,
    models: Optional[Sequence[Mapping[str, Any]]] = None,
    max_pairs: Optional[int] = None,
    sleep: float = 1.0,
    min_confidence: int = 3,
    min_models: int = 2,
    checkpoint_path: Optional[Union[str, Path]] = None,
    resume: bool = True,
    retry_failed: bool = False,
    max_retries: int = 3,
    timeout: float = 90.0,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run live vocabulary alignment over all unordered cluster pairs.

    The return value remains backward-compatible:

        raw_df, summary_df = align_vocabulary(...)

    Parameters added beyond the historical API are optional. When
    ``checkpoint_path`` is provided, one JSONL record is written after every
    model × cluster-pair call. A resumed run skips completed calls.

    ``retry_failed=True`` re-runs checkpointed calls whose previous status was
    ``failed``.
    """
    if not 1 <= min_confidence <= 5:
        raise ValueError("min_confidence must be between 1 and 5")
    if min_models <= 0:
        raise ValueError("min_models must be positive")
    if sleep < 0:
        raise ValueError("sleep cannot be negative")
    if max_pairs is not None and max_pairs < 0:
        raise ValueError("max_pairs cannot be negative")
    if max_retries <= 0:
        raise ValueError("max_retries must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    selected_models = _validate_models(models)
    labels = dict(cluster_labels or {})

    terms: Dict[Union[int, str], List[str]] = {}
    for cluster_id, values in cluster_terms.items():
        clean_values = [
            _safe_text(value)
            for value in values
            if _safe_text(value)
        ]
        if not clean_values:
            raise ValueError(f"cluster {cluster_id!r} has no valid terms")
        terms[cluster_id] = clean_values

    cluster_ids = sorted(terms, key=lambda value: str(value))
    if len(cluster_ids) < 2:
        raise ValueError("at least two clusters are required")

    pairs = list(itertools.combinations(cluster_ids, 2))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    raw_rows: List[Dict[str, Any]] = []
    completed = set()

    checkpoint: Optional[Path] = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else None
    )

    if checkpoint is not None and resume and checkpoint.is_file():
        checkpoint_rows = _load_checkpoint_rows(checkpoint)
        raw_rows.extend(checkpoint_rows)

        checkpoint_frame = pd.DataFrame(checkpoint_rows)
        if not checkpoint_frame.empty:
            for key, group in checkpoint_frame.groupby(
                ["cluster_a", "cluster_b", "model_key"],
                dropna=False,
            ):
                statuses = set(group["status"].fillna("").astype(str))
                failed_only = statuses == {"failed"}
                if retry_failed and failed_only:
                    continue
                completed.add(tuple(str(value) for value in key))

    for cluster_a, cluster_b in pairs:
        terms_a = terms[cluster_a]
        terms_b = terms[cluster_b]

        label_a = str(labels.get(cluster_a, f"Cluster {cluster_a}"))
        label_b = str(labels.get(cluster_b, f"Cluster {cluster_b}"))

        formatted_a = "\n".join(
            f"{index + 1}. {term}"
            for index, term in enumerate(terms_a)
        )
        formatted_b = "\n".join(
            f"{index + 1}. {term}"
            for index, term in enumerate(terms_b)
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            label_a=label_a,
            label_b=label_b,
            terms_a=formatted_a,
            terms_b=formatted_b,
        )
        prompt_sha256 = _prompt_hash(SYSTEM_PROMPT, user_prompt)

        for model in selected_models:
            key = (
                str(cluster_a),
                str(cluster_b),
                model["key"],
            )
            if key in completed:
                continue

            request_timestamp = _utc_now()
            call_result = _call_openrouter(
                model=model["model"],
                system=SYSTEM_PROMPT,
                user=user_prompt,
                max_retries=max_retries,
                timeout=timeout,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            rows = _flatten_call_result(
                cluster_a=cluster_a,
                cluster_b=cluster_b,
                label_a=label_a,
                label_b=label_b,
                model=model,
                terms_a=terms_a,
                terms_b=terms_b,
                request_timestamp=request_timestamp,
                prompt_sha256=prompt_sha256,
                temperature=temperature,
                max_tokens=max_tokens,
                call_result=call_result,
            )
            raw_rows.extend(rows)

            if checkpoint is not None:
                _append_checkpoint(
                    checkpoint,
                    cluster_a=cluster_a,
                    cluster_b=cluster_b,
                    model_key=model["key"],
                    rows=rows,
                )

            if sleep:
                time.sleep(sleep)

    raw = pd.DataFrame(raw_rows)
    for column in RAW_COLUMNS:
        if column not in raw.columns:
            raw[column] = ""

    raw = raw[RAW_COLUMNS].copy() if not raw.empty else _empty_raw_frame()

    summary = consensus_from_raw(
        raw,
        min_confidence=min_confidence,
        min_models=min_models,
    )
    return raw, summary


__all__ = [
    "ALLOWED_RELATION_TYPES",
    "DEFAULT_MODELS",
    "PROMPT_VERSION",
    "QUERY_EXPANSION_SAFE_RELATIONS",
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "align_vocabulary",
    "alignment_report",
    "consensus_details_from_raw",
    "consensus_from_raw",
    "load_cluster_terms",
    "load_env",
]
