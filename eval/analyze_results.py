#!/usr/bin/env python3
"""
Analyze evaluation results and produce a comprehensive report.

Usage:
    python eval/analyze_results.py [--input eval/eval_results.jsonl] [--save eval/eval_report.txt]
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict


SIMPLE_LABELS = {"basic SQL", "single join"}
MODERATE_LABELS = {"aggregation", "subqueries"}
COMPLEX_LABELS = {"multiple_joins", "window functions", "set operations", "CTEs"}


def load_results(path):
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def get_tier(complexity):
    if complexity in SIMPLE_LABELS:
        return "simple"
    elif complexity in MODERATE_LABELS:
        return "moderate"
    elif complexity in COMPLEX_LABELS:
        return "complex"
    return "unknown"


def pct(n, total):
    if total == 0:
        return "0.0%"
    return f"{100 * n / total:.1f}%"


def avg(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def print_section(title, lines, output):
    output.append("")
    output.append(f"{'═' * 60}")
    output.append(f"  {title}")
    output.append(f"{'═' * 60}")
    for line in lines:
        output.append(line)


def print_table(headers, rows, output):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = "  ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "  ".join("-" * w for w in col_widths)
    output.append(f"  {header_line}")
    output.append(f"  {sep_line}")
    for row in rows:
        row_line = "  ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row))
        output.append(f"  {row_line}")


def analyze(results):
    output = []
    output.append("SQLChange Pipeline — Evaluation Report")
    output.append(f"Queries evaluated: {len(results)}")

    for r in results:
        r["_tier"] = get_tier(r.get("sql_complexity", ""))

    # ── 1. Pipeline Health ──
    errors = [r for r in results if r.get("pipeline_error")]
    successes = [r for r in results if not r.get("pipeline_error")]
    times = [r["wall_clock_seconds"] for r in results]

    lines = [
        f"  Success rate:   {pct(len(successes), len(results))} ({len(successes)}/{len(results)})",
        f"  Errors:         {len(errors)}",
        f"  Avg time:       {avg(times):.1f}s",
        f"  Min/Max time:   {min(times):.1f}s / {max(times):.1f}s" if times else "  No timing data",
    ]

    if errors:
        lines.append("")
        lines.append("  Errors by tier:")
        err_by_tier = Counter(r["_tier"] for r in errors)
        for tier, count in err_by_tier.most_common():
            lines.append(f"    {tier}: {count}")
        lines.append("")
        lines.append("  Error samples:")
        for r in errors[:5]:
            lines.append(f"    [{r['query_id']}] {r['pipeline_error'][:80]}")

    print_section("1. Pipeline Health", lines, output)

    # ── 2. Action Distribution ──
    action_counts = Counter(r.get("action", "unknown") for r in results)
    lines = []
    for action, count in action_counts.most_common():
        lines.append(f"  {action:20s} {count:4d}  ({pct(count, len(results))})")

    lines.append("")
    lines.append("  By complexity tier:")
    tiers = ["simple", "moderate", "complex"]
    actions = sorted(action_counts.keys())
    headers = ["Tier"] + actions
    rows = []
    for tier in tiers:
        tier_results = [r for r in results if r["_tier"] == tier]
        row = [tier]
        for action in actions:
            count = sum(1 for r in tier_results if r.get("action") == action)
            row.append(f"{count} ({pct(count, len(tier_results))})")
        rows.append(row)
    print_table(headers, rows, lines)
    print_section("2. Action Distribution", lines, output)

    # ── 3. Label Distributions ──
    lines = []

    lines.append("  Performance labels:")
    perf_labels = Counter()
    for r in successes:
        pl = r.get("performance_label", {})
        perf_labels[pl.get("label", "missing")] += 1
    for label, count in perf_labels.most_common():
        lines.append(f"    {label:15s} {count:4d}  ({pct(count, len(successes))})")

    lines.append("")
    lines.append("  Risk labels:")
    risk_labels = Counter()
    for r in successes:
        rl = r.get("risk_label", {})
        risk_labels[rl.get("label", "missing")] += 1
    for label, count in risk_labels.most_common():
        lines.append(f"    {label:15s} {count:4d}  ({pct(count, len(successes))})")

    lines.append("")
    lines.append("  Semantic labels:")
    sem_labels = Counter()
    for r in successes:
        sl = r.get("semantic_label", {})
        sem_labels[sl.get("label", "missing")] += 1
    for label, count in sem_labels.most_common():
        lines.append(f"    {label:15s} {count:4d}  ({pct(count, len(successes))})")

    print_section("3. Label Distributions", lines, output)

    # ── 4. Semantic Preservation ──
    equiv_count = sum(
        1 for r in successes
        if r.get("semantic_label", {}).get("label") == "equivalent"
    )
    lines = [
        f"  Overall equivalence rate: {pct(equiv_count, len(successes))} ({equiv_count}/{len(successes)})",
        "",
        "  By complexity tier:",
    ]
    for tier in tiers:
        tier_s = [r for r in successes if r["_tier"] == tier]
        tier_eq = sum(1 for r in tier_s if r.get("semantic_label", {}).get("label") == "equivalent")
        lines.append(f"    {tier:12s} {pct(tier_eq, len(tier_s)):>6s}  ({tier_eq}/{len(tier_s)})")
    print_section("4. Semantic Preservation", lines, output)

    # ── 5. Performance Improvement ──
    lines = []

    perf_scores = [
        r.get("performance_label", {}).get("score")
        for r in successes
        if r.get("performance_label", {}).get("score") is not None
    ]
    if perf_scores:
        lines.append(f"  Avg performance score: {avg(perf_scores):.1f} / 10")
        lines.append(f"  Score distribution:")
        score_dist = Counter(int(s) for s in perf_scores)
        for score in range(1, 11):
            count = score_dist.get(score, 0)
            bar = "█" * count
            lines.append(f"    {score:2d}: {bar} ({count})")

    speedups = []
    for r in successes:
        ev = r.get("execution_evidence", {})
        if isinstance(ev, dict):
            large = ev.get("large", {})
            if isinstance(large, dict) and large.get("speedup"):
                try:
                    speedups.append(float(large["speedup"]))
                except (ValueError, TypeError):
                    pass

    if speedups:
        improved = sum(1 for s in speedups if s > 1.0)
        lines.append("")
        lines.append(f"  Queries with speedup > 1.0 (large scale): {pct(improved, len(speedups))} ({improved}/{len(speedups)})")
        lines.append(f"  Avg speedup (large scale): {avg(speedups):.2f}x")
        lines.append(f"  Min/Max speedup: {min(speedups):.2f}x / {max(speedups):.2f}x")

        lines.append("")
        lines.append("  Avg speedup by tier:")
        for tier in tiers:
            tier_speedups = []
            for r in successes:
                if r["_tier"] != tier:
                    continue
                ev = r.get("execution_evidence", {})
                if isinstance(ev, dict):
                    large = ev.get("large", {})
                    if isinstance(large, dict) and large.get("speedup"):
                        try:
                            tier_speedups.append(float(large["speedup"]))
                        except (ValueError, TypeError):
                            pass
            if tier_speedups:
                lines.append(f"    {tier:12s} {avg(tier_speedups):.2f}x  (n={len(tier_speedups)})")
    else:
        lines.append("  No speedup data available in execution evidence")

    print_section("5. Performance Improvement", lines, output)

    # ── 6. Degraded Query Recovery ──
    degraded = [r for r in results if r.get("is_degraded")]
    lines = [f"  Degraded queries: {len(degraded)}"]

    if degraded:
        degraded_optimized = [r for r in degraded if r.get("action") in ("optimize", "rewrite")]
        lines.append(f"  Recommended optimization: {pct(len(degraded_optimized), len(degraded))} ({len(degraded_optimized)}/{len(degraded)})")

        degraded_equiv = [
            r for r in degraded
            if r.get("semantic_label", {}).get("label") == "equivalent"
            and r.get("action") in ("optimize", "rewrite")
        ]
        lines.append(f"  Optimized + equivalent: {pct(len(degraded_equiv), len(degraded))} ({len(degraded_equiv)}/{len(degraded)})")

        lines.append("")
        lines.append("  By degradation type:")
        deg_types = set(r.get("degradation_type", "none") for r in degraded)
        for dt in sorted(deg_types):
            dt_queries = [r for r in degraded if r.get("degradation_type") == dt]
            dt_optimized = [r for r in dt_queries if r.get("action") in ("optimize", "rewrite")]
            lines.append(f"    {dt:25s} optimized: {len(dt_optimized)}/{len(dt_queries)}")

    print_section("6. Degraded Query Recovery", lines, output)

    # ── 7. DDL vs No-DDL ──
    ddl_queries = [r for r in results if r.get("has_ddl")]
    no_ddl_queries = [r for r in results if not r.get("has_ddl")]

    lines = [
        f"  DDL queries:    {len(ddl_queries)}",
        f"  No-DDL queries: {len(no_ddl_queries)}",
    ]

    if ddl_queries and no_ddl_queries:
        ddl_success = [r for r in ddl_queries if not r.get("pipeline_error")]
        no_ddl_success = [r for r in no_ddl_queries if not r.get("pipeline_error")]

        lines.append("")
        lines.append("  Comparison:")
        headers = ["Metric", "DDL", "No-DDL"]
        rows = [
            ["Success rate", pct(len(ddl_success), len(ddl_queries)), pct(len(no_ddl_success), len(no_ddl_queries))],
            ["Avg time (s)", f"{avg([r['wall_clock_seconds'] for r in ddl_queries]):.1f}", f"{avg([r['wall_clock_seconds'] for r in no_ddl_queries]):.1f}"],
        ]

        ddl_equiv = sum(1 for r in ddl_success if r.get("semantic_label", {}).get("label") == "equivalent")
        no_ddl_equiv = sum(1 for r in no_ddl_success if r.get("semantic_label", {}).get("label") == "equivalent")
        rows.append(["Equivalence rate", pct(ddl_equiv, len(ddl_success)), pct(no_ddl_equiv, len(no_ddl_success))])

        for action in ["optimize", "keep_original"]:
            ddl_a = sum(1 for r in ddl_success if r.get("action") == action)
            no_ddl_a = sum(1 for r in no_ddl_success if r.get("action") == action)
            rows.append([f"Action: {action}", pct(ddl_a, len(ddl_success)), pct(no_ddl_a, len(no_ddl_success))])

        print_table(headers, rows, lines)

    print_section("7. DDL vs No-DDL Comparison", lines, output)

    # ── 8. Iteration Behavior ──
    iterations = [r.get("iterations_used", 0) for r in successes]
    lines = []
    if iterations:
        lines.append(f"  Avg iterations used: {avg(iterations):.1f}")
        iter_dist = Counter(iterations)
        for it in sorted(iter_dist.keys()):
            lines.append(f"    {it} iterations: {iter_dist[it]} queries ({pct(iter_dist[it], len(successes))})")

        multi_iter = [r for r in successes if r.get("iterations_used", 0) > 1]
        if multi_iter:
            lines.append("")
            lines.append(f"  Queries using >1 iteration: {len(multi_iter)} ({pct(len(multi_iter), len(successes))})")
            multi_perf_scores = [
                r.get("performance_label", {}).get("score")
                for r in multi_iter
                if r.get("performance_label", {}).get("score") is not None
            ]
            single_perf_scores = [
                r.get("performance_label", {}).get("score")
                for r in successes
                if r.get("iterations_used", 0) == 1
                and r.get("performance_label", {}).get("score") is not None
            ]
            if multi_perf_scores and single_perf_scores:
                lines.append(f"  Avg perf score (1 iter):  {avg(single_perf_scores):.1f}")
                lines.append(f"  Avg perf score (>1 iter): {avg(multi_perf_scores):.1f}")

    print_section("8. Iteration Behavior", lines, output)

    # ── 9. Weakness Patterns ──
    lines = []

    flagged = [r for r in results if r.get("action") == "flag_for_review"]
    if flagged:
        lines.append(f"  Flagged for review: {len(flagged)}")
        flag_tiers = Counter(r["_tier"] for r in flagged)
        for tier, count in flag_tiers.most_common():
            lines.append(f"    {tier}: {count}")
    else:
        lines.append("  No queries flagged for review")

    invalid = [r for r in successes if not r.get("recommendation_is_valid", True)]
    if invalid:
        lines.append("")
        lines.append(f"  Invalid SQL recommendations: {len(invalid)}")
        inv_tiers = Counter(r["_tier"] for r in invalid)
        for tier, count in inv_tiers.most_common():
            lines.append(f"    {tier}: {count}")

    non_equiv = [
        r for r in successes
        if r.get("semantic_label", {}).get("label") not in ("equivalent", "missing", None)
        and r.get("action") in ("optimize", "rewrite")
    ]
    if non_equiv:
        lines.append("")
        lines.append(f"  Non-equivalent optimizations: {len(non_equiv)} (changed query semantics)")
        ne_tiers = Counter(r["_tier"] for r in non_equiv)
        for tier, count in ne_tiers.most_common():
            lines.append(f"    {tier}: {count}")
        ne_sem = Counter(r.get("semantic_label", {}).get("label") for r in non_equiv)
        for label, count in ne_sem.most_common():
            lines.append(f"    semantic={label}: {count}")

    lines.append("")
    lines.append("  Avg time by tier:")
    for tier in tiers:
        tier_times = [r["wall_clock_seconds"] for r in results if r["_tier"] == tier]
        if tier_times:
            lines.append(f"    {tier:12s} {avg(tier_times):.1f}s  (n={len(tier_times)})")

    print_section("9. Weakness Patterns", lines, output)

    return "\n".join(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument("--input", default="eval/eval_results.jsonl")
    parser.add_argument("--save", default=None, help="Save report to file")
    args = parser.parse_args()

    results = load_results(args.input)
    report = analyze(results)
    print(report)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            f.write(report)
        print(f"\nReport saved to {args.save}")
