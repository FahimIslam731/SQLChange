"""
Evaluation framework for SQLChange.

Compares 3 baselines to measure whether structured analysis (mutation
generation + execution evidence + ER graph) improves recommendation
quality over naive LLM approaches.

Baselines:
  1. zero_shot       — give LLM the query, ask for optimization advice
  2. structured      — step-by-step instructions, no evidence
  3. sqlchange       — full pipeline (mutations + execution + graph + rules)

Ground truth: rule-based labels from reasoning_pipeline + execution evidence.
"""

import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(__file__))

from graph_representer import llm_universal_call_utility
from parser import parse_sql
from recommend import recommend

DIMENSIONS = ("semantic", "performance", "risk")

LABEL_SETS = {
    "semantic": {"equivalent", "narrower", "broader", "different"},
    "performance": {"improves", "degrades", "neutral", "unknown"},
    "risk": {"low", "medium", "high"},
}


def _llm_call(prompt, provider, model, api_key):
    try:
        raw = llm_universal_call_utility(prompt=prompt, provider=provider,
                                         api_key=api_key, model=model)
        text = raw.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        return {"label": "unknown", "confidence": 0.0, "rationale": f"error: {e}"}


def _zero_shot_prompt(record):
    return (
        "You are a SQL optimization expert. Look at this query and suggest "
        "whether any optimization is possible.\n\n"
        f"Query: {record['original_sql']}\n\n"
        "For each dimension, classify:\n"
        "- semantic: would your change be equivalent, narrower, broader, or different?\n"
        "- performance: would it improve, degrade, be neutral, or unknown?\n"
        "- risk: low, medium, or high?\n\n"
        "Return ONLY JSON:\n"
        '{"semantic": {"label": "..."}, "performance": {"label": "..."}, "risk": {"label": "..."}}'
    )


def _structured_prompt(record):
    return (
        "You are a SQL optimization expert. Follow these steps:\n"
        "1. Identify which clauses could be modified (WHERE, JOIN, GROUP BY, LIMIT, columns)\n"
        "2. For each possible change, reason about correctness and performance\n"
        "3. Assess the best optimization opportunity\n\n"
        f"Query: {record['original_sql']}\n"
        f"Schema context: {json.dumps(record.get('context', {}), default=str)}\n\n"
        "Classify your best optimization:\n"
        "- semantic: equivalent, narrower, broader, or different?\n"
        "- performance: improves, degrades, neutral, or unknown?\n"
        "- risk: low, medium, or high?\n\n"
        "Return ONLY JSON:\n"
        '{"semantic": {"label": "..."}, "performance": {"label": "..."}, "risk": {"label": "..."}}'
    )


def evaluate_record(record, provider="anthropic",
                    model="claude-sonnet-4-20250514", api_key=None):
    """Run all 3 baselines on one record, compare against ground truth."""
    gt = {
        "semantic": record.get("semantic_label", "unknown"),
        "performance": record.get("performance_label", "unknown"),
        "risk": record.get("risk_label", "unknown"),
    }

    zero = _llm_call(_zero_shot_prompt(record), provider, model, api_key)
    structured = _llm_call(_structured_prompt(record), provider, model, api_key)

    schema_ddl = "\n".join(
        f"CREATE TABLE {t} ({', '.join(f'{c} {info['types'].get(c, 'TEXT')}' for c in info['columns'])})"
        for t, info in record.get("context", {}).items()
    )
    sqlchange_result = recommend(record["original_sql"], schema_ddl,
                                 provider, model, api_key)
    sqlchange_rec = sqlchange_result.get("recommendation", {})

    results = {}
    for dim in DIMENSIONS:
        zero_label = zero.get(dim, {}).get("label", zero.get("label", "unknown"))
        struct_label = structured.get(dim, {}).get("label", structured.get("label", "unknown"))
        sc_label = sqlchange_rec.get(dim, {}).get("label", "unknown")
        results[dim] = {
            "ground_truth": gt[dim],
            "zero_shot": {"label": zero_label},
            "structured": {"label": struct_label},
            "sqlchange": {"label": sc_label},
        }
    return results


def compute_metrics(eval_results: List[Dict]) -> Dict[str, Any]:
    baselines = ("zero_shot", "structured", "sqlchange")
    metrics = {}
    for dim in DIMENSIONS:
        dim_metrics = {}
        for baseline in baselines:
            correct = 0
            total = 0
            confusion = Counter()
            for result in eval_results:
                gt = result[dim]["ground_truth"]
                pred = result[dim][baseline].get("label", "unknown")
                total += 1
                if pred == gt:
                    correct += 1
                else:
                    confusion[(gt, pred)] += 1
            dim_metrics[baseline] = {
                "accuracy": correct / total if total else 0,
                "correct": correct,
                "total": total,
                "top_confusions": [
                    {"true": t, "predicted": p, "count": c}
                    for (t, p), c in confusion.most_common(5)
                ],
            }
        metrics[dim] = dim_metrics
    return metrics


def print_metrics(metrics: Dict[str, Any]):
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


def evaluate_dataset(records, sample_size=None, provider="anthropic",
                     model="claude-sonnet-4-20250514", api_key=None):
    selected = records[:sample_size] if sample_size else records
    results = []
    for i, record in enumerate(selected):
        print(f"  [{i+1}/{len(selected)}] Evaluating record {record.get('unique_id', i)}...")
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
    parser.add_argument("--output", default=None)
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
