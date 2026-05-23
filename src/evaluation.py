"""
Evaluation framework for SQLChange.

Compares 3 baselines against ground truth to measure whether
structured analysis improves LLM accuracy on SQL mutation classification.

Baselines:
  1. zero_shot       — raw SQL pair, no guidance
  2. structured      — step-by-step reasoning instructions
  3. sqlchange       — full pipeline (structural diff + ER graph + execution evidence)
"""

import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(__file__))

from graph_representer import llm_universal_call_utility
from recommend import _structural_diff, _build_record, _execution_evidence, _llm_call
from parser import parse_sql, get_join_keys, get_where_details
from graph_representer import build_graph

DIMENSIONS = ("semantic", "performance", "risk")

LABEL_SETS = {
    "semantic": {"equivalent", "narrower", "broader", "different"},
    "performance": {"improves", "degrades", "neutral", "unknown"},
    "risk": {"low", "medium", "high"},
}


# ---------------------------------------------------------------------------
# Ground truth extraction (from synthetic execution + rule labels)
# ---------------------------------------------------------------------------

def ground_truth_from_record(record: Dict[str, Any]) -> Dict[str, str]:
    """Extract ground truth labels from a labeled dataset record."""
    return {
        "semantic": record.get("semantic_label", "unknown"),
        "performance": record.get("performance_label", "unknown"),
        "risk": record.get("risk_label", "unknown"),
    }


# ---------------------------------------------------------------------------
# Baseline prompts
# ---------------------------------------------------------------------------

def _zero_shot_prompt(record, dimension):
    labels = ", ".join(sorted(LABEL_SETS[dimension]))
    return (
        f"Compare these two SQL queries and classify the {dimension} impact.\n\n"
        f"Original: {record['original_sql']}\n"
        f"Modified: {record['modified_sql']}\n\n"
        f"Allowed labels: {labels}\n\n"
        f'Return ONLY JSON: {{"label": "...", "confidence": 0.0, "rationale": "..."}}'
    )


def _structured_prompt(record, dimension):
    labels = ", ".join(sorted(LABEL_SETS[dimension]))
    steps = {
        "semantic": "1. Identify what SQL clauses changed. 2. Determine how the result set is affected. 3. Classify the relationship.",
        "performance": "1. Identify what SQL clauses changed. 2. Consider index usage, scan type, and row volume. 3. Predict the direction.",
        "risk": "1. Identify what SQL clauses changed. 2. Consider downstream consumers and data correctness. 3. Assess severity.",
    }
    return (
        f"You are a SQL analysis expert. Follow these steps to classify the "
        f"{dimension} impact of this SQL modification.\n\n"
        f"{steps[dimension]}\n\n"
        f"Original: {record['original_sql']}\n"
        f"Modified: {record['modified_sql']}\n\n"
        f"Allowed labels: {labels}\n\n"
        f'Return ONLY JSON: {{"label": "...", "confidence": 0.0, "rationale": "..."}}'
    )


def _sqlchange_prompt(record, dimension, diff, er_graph, execution):
    labels = ", ".join(sorted(LABEL_SETS[dimension]))
    evidence = {
        "original_sql": record["original_sql"],
        "modified_sql": record["modified_sql"],
        "structural_diff": diff,
        "er_graph": er_graph or {},
    }
    if execution and "query_pair" in execution:
        comp = execution["query_pair"]["comparison"]
        evidence["execution"] = {
            "output_relation": comp["output_relation"],
            "row_count_original": comp["row_count_original"],
            "row_count_modified": comp["row_count_modified"],
            "both_succeeded": comp["both_succeeded"],
        }
    if execution and "performance" in execution:
        evidence["performance_timing"] = execution["performance"]
    return (
        f"You are a SQL query analysis expert. Analyze this SQL modification's "
        f"{dimension} impact using the structured evidence below.\n\n"
        f"Allowed labels: {labels}\n\n"
        f"Evidence:\n{json.dumps(evidence, indent=2, default=str)}\n\n"
        f'Return ONLY JSON: {{"label": "...", "confidence": 0.0, "rationale": "..."}}'
    )


# ---------------------------------------------------------------------------
# Run baselines
# ---------------------------------------------------------------------------

def _run_baseline(prompt, provider, model, api_key):
    try:
        return _llm_call(prompt, provider, model, api_key)
    except Exception as e:
        return {"label": "unknown", "confidence": 0.0, "rationale": f"error: {e}"}


def evaluate_record(record, provider="anthropic", model="claude-sonnet-4-20250514",
                    api_key=None):
    """Run all 3 baselines on one record and return predictions."""
    diff = _structural_diff(record["original_sql"], record["modified_sql"])

    er_graph = {}
    if record.get("context"):
        try:
            out = build_graph(record["context"], record.get("join_keys", []),
                              record.get("where_details", []), model, provider, api_key)
            er_graph = out.get("data_graph", {})
        except Exception:
            pass

    execution = _execution_evidence(record) if record.get("context") else {}

    results = {}
    for dim in DIMENSIONS:
        results[dim] = {
            "ground_truth": ground_truth_from_record(record)[dim],
            "zero_shot": _run_baseline(
                _zero_shot_prompt(record, dim), provider, model, api_key),
            "structured": _run_baseline(
                _structured_prompt(record, dim), provider, model, api_key),
            "sqlchange": _run_baseline(
                _sqlchange_prompt(record, dim, diff, er_graph, execution),
                provider, model, api_key),
        }
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_results: List[Dict]) -> Dict[str, Any]:
    """Compute accuracy and per-label breakdown across all evaluated records."""
    baselines = ("zero_shot", "structured", "sqlchange")
    metrics = {}

    for dim in DIMENSIONS:
        dim_metrics = {}
        for baseline in baselines:
            correct = 0
            total = 0
            label_counts = Counter()
            confusion = Counter()
            for result in eval_results:
                gt = result[dim]["ground_truth"]
                pred = result[dim][baseline].get("label", "unknown")
                total += 1
                label_counts[pred] += 1
                if pred == gt:
                    correct += 1
                else:
                    confusion[(gt, pred)] += 1
            dim_metrics[baseline] = {
                "accuracy": correct / total if total else 0,
                "correct": correct,
                "total": total,
                "label_distribution": dict(label_counts),
                "top_confusions": [
                    {"true": t, "predicted": p, "count": c}
                    for (t, p), c in confusion.most_common(5)
                ],
            }
        metrics[dim] = dim_metrics
    return metrics


def print_metrics(metrics: Dict[str, Any]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("SQLChange Evaluation Results")
    print("=" * 70)

    for dim in DIMENSIONS:
        print(f"\n--- {dim.upper()} ---")
        print(f"  {'Baseline':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
        print(f"  {'-'*48}")
        for baseline in ("zero_shot", "structured", "sqlchange"):
            m = metrics[dim][baseline]
            print(f"  {baseline:<20} {m['accuracy']:>10.1%} {m['correct']:>10} {m['total']:>8}")

        print(f"\n  Top confusions (sqlchange):")
        for c in metrics[dim]["sqlchange"]["top_confusions"][:3]:
            print(f"    {c['true']} -> {c['predicted']}: {c['count']}x")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(records, sample_size=None, provider="anthropic",
                     model="claude-sonnet-4-20250514", api_key=None):
    """Evaluate a sample of records and return metrics."""
    selected = records[:sample_size] if sample_size else records
    results = []
    for i, record in enumerate(selected):
        print(f"  Evaluating record {i+1}/{len(selected)} (id={record.get('unique_id')})...")
        results.append(evaluate_record(record, provider, model, api_key))
    return results, compute_metrics(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SQLChange Evaluation Framework")
    parser.add_argument("--input", default="../data/sqlchange_labeled.json")
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", default=None, help="Save raw results to JSON")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    with open(args.input) as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records, evaluating {args.sample_size}...")
    results, metrics = evaluate_dataset(records, args.sample_size,
                                        args.provider, args.model, api_key)
    print_metrics(metrics)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"results": results, "metrics": metrics}, f, indent=2, default=str)
        print(f"\nRaw results saved to {args.output}")
