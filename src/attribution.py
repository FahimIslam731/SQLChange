"""
LLM-as-judge attribution analysis for SQL query modifications.

Runs each query pair through the full SQLChange execution pipeline
(synthetic DB, equivalence check, performance benchmarking) then asks
the LLM to judge which structural components drove its assessment —
grounded in real execution evidence, not just raw SQL.

Designed for Caliper — observe the attention weights in Caliper's
endpoint visualizer while calls run.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(__file__))

from graph_representer import llm_universal_call_utility, build_graph
from parser import parse_sql, get_join_keys, get_where_details
from equivalence import check_equivalence
from performance import compare_performance
from reasoning_pipeline import _base_reasoning, _rule_signals

COMPONENTS = ["WHERE", "JOIN", "GROUP_BY", "SELECT_COLUMNS", "LIMIT", "ORDER_BY"]

DIMENSIONS = ("semantic", "performance", "risk")


def _gather_execution_evidence(record: Dict[str, Any]) -> Dict[str, Any]:
    """Run the query pair through synthetic DB for equivalence and performance."""
    evidence = {}

    print("  Running equivalence check...")
    try:
        equiv = check_equivalence(record, seed=42, rows_per_table=100)
        evidence["equivalence"] = equiv
        print(f"    Result: {equiv.get('output_relation', '?')}"
              f"  rows: {equiv.get('row_count_original', '?')} -> {equiv.get('row_count_modified', '?')}")
    except Exception as e:
        evidence["equivalence"] = {"error": str(e)}
        print(f"    Equivalence failed: {e}")

    print("  Running performance benchmark...")
    try:
        perf = compare_performance(record, scales={"small": 50, "large": 1000},
                                   repeats=5, seed=42)
        evidence["performance"] = perf
        for scale in ("small", "large"):
            s = perf.get(scale, {})
            if s.get("speedup"):
                print(f"    {scale}: {s['speedup']:.2f}x"
                      f"  ({s.get('original_ms', 0):.2f}ms -> {s.get('modified_ms', 0):.2f}ms)")
    except Exception as e:
        evidence["performance"] = {"error": str(e)}
        print(f"    Performance failed: {e}")

    print("  Computing rule-based labels...")
    try:
        rules = _base_reasoning(record)
        evidence["rule_labels"] = {dim: rules[dim]["label"] for dim in DIMENSIONS}
        print(f"    Rules: semantic={rules['semantic']['label']}"
              f"  performance={rules['performance']['label']}"
              f"  risk={rules['risk']['label']}")
    except Exception as e:
        evidence["rule_labels"] = {"error": str(e)}

    return evidence


def _build_attribution_prompt(record: Dict[str, Any], evidence: Dict[str, Any] = None) -> str:
    original = record.get("original_sql", "")
    modified = record.get("modified_sql", "")
    mutation_type = record.get("mutation_type", "unknown")
    context = record.get("context", {})

    schema_summary = ", ".join(
        f"{t}({','.join(info.get('columns', []))})"
        for t, info in context.items()
    )

    evidence_block = ""
    if evidence:
        parts = []
        eq = evidence.get("equivalence", {})
        if not eq.get("error"):
            parts.append(f"Execution: {eq.get('output_relation','?')}, "
                         f"rows {eq.get('row_count_original','?')}->{eq.get('row_count_modified','?')}")
        perf = evidence.get("performance", {})
        if not perf.get("error"):
            for scale in ("small", "large"):
                s = perf.get(scale, {})
                if s.get("speedup"):
                    parts.append(f"{scale}: {s['speedup']:.2f}x speedup "
                                 f"({s.get('original_ms',0):.1f}ms->{s.get('modified_ms',0):.1f}ms)")
        rules = evidence.get("rule_labels", {})
        if not rules.get("error"):
            parts.append(f"Rule labels: sem={rules.get('semantic','?')} "
                         f"perf={rules.get('performance','?')} risk={rules.get('risk','?')}")
        evidence_block = "\n".join(parts)

    return f"""Classify SQL change and rate component importance. JSON only, no explanation.

Mutation: {mutation_type} | Schema: {schema_summary}
Old: {original}
New: {modified}
{evidence_block}
Classify: semantic={{equivalent,narrower,broader,different}} performance={{improves,degrades,neutral,unknown}} risk={{low,medium,high}}
Rate each component importance: high/medium/low/none.
ONLY valid JSON:
{{"classification":{{"semantic":"...","performance":"...","risk":"..."}},"attribution":{{"WHERE":{{"semantic":"...","performance":"...","risk":"..."}},"JOIN":{{"semantic":"...","performance":"...","risk":"..."}},"GROUP_BY":{{"semantic":"...","performance":"...","risk":"..."}},"SELECT_COLUMNS":{{"semantic":"...","performance":"...","risk":"..."}},"LIMIT":{{"semantic":"...","performance":"...","risk":"..."}},"ORDER_BY":{{"semantic":"...","performance":"...","risk":"..."}}}}}}"""


def _parse_response(raw: str) -> Dict[str, Any]:
    text = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


IMPORTANCE_SCORES = {"high": 3, "medium": 2, "low": 1, "none": 0}


def _importance_to_score(importance: str) -> int:
    return IMPORTANCE_SCORES.get(importance.lower().strip(), 0)


def attribute_record(record: Dict[str, Any], provider: str = "caliper",
                     model: str = None, api_key: str = None) -> Dict[str, Any]:
    """Run execution pipeline then LLM-as-judge attribution on a single record."""
    print(f"\n{'='*60}")
    print(f"ATTRIBUTION: record {record.get('unique_id', '?')} | {record.get('mutation_type', '?')}")
    print(f"  Original: {record.get('original_sql', '')[:80]}...")
    print(f"  Modified: {record.get('modified_sql', '')[:80]}...")

    evidence = _gather_execution_evidence(record)

    prompt = _build_attribution_prompt(record, evidence)
    print(f"  Calling {provider} with execution evidence...")

    try:
        raw = llm_universal_call_utility(
            prompt=prompt, provider=provider,
            api_key=api_key, model=model,
            num_predict=350, think=False
        )
    except Exception as e:
        print(f"  LLM call failed: {e}")
        return {
            "unique_id": record.get("unique_id"),
            "mutation_type": record.get("mutation_type"),
            "original_sql": record.get("original_sql"),
            "modified_sql": record.get("modified_sql"),
            "classification": {},
            "component_attribution": {},
            "error": str(e),
        }

    print(f"  Response received ({len(raw)} chars)")

    parsed = _parse_response(raw)

    classification = parsed.get("classification", {})
    attributions = parsed.get("attribution", parsed.get("component_attribution", {}))

    result = {
        "unique_id": record.get("unique_id"),
        "mutation_type": record.get("mutation_type"),
        "original_sql": record.get("original_sql"),
        "modified_sql": record.get("modified_sql"),
        "execution_evidence": evidence,
        "classification": classification,
        "component_attribution": attributions,
        "raw_response": raw,
    }

    _print_attribution(result)
    return result


def _extract_importance(comp_data, dim):
    """Extract importance from either nested or flat response format."""
    if not isinstance(comp_data, dict):
        return "none"
    val = comp_data.get(dim, "none")
    if isinstance(val, dict):
        return val.get("importance", "none")
    if isinstance(val, str):
        return val
    return "none"


def _print_attribution(result: Dict[str, Any]):
    """Print execution evidence + component attribution heatmap."""
    evidence = result.get("execution_evidence", {})
    attributions = result.get("component_attribution", {})
    classification = result.get("classification", {})

    eq = evidence.get("equivalence", {})
    if eq and not eq.get("error"):
        print(f"\n  Execution: {eq.get('output_relation', '?')}"
              f"  rows: {eq.get('row_count_original', '?')} -> {eq.get('row_count_modified', '?')}")
    perf = evidence.get("performance", {})
    if perf and not perf.get("error"):
        for scale in ("small", "large"):
            s = perf.get(scale, {})
            if s.get("speedup"):
                print(f"  {scale:>7}: {s['speedup']:.2f}x"
                      f"  ({s.get('original_ms', 0):.2f}ms -> {s.get('modified_ms', 0):.2f}ms)")
    rules = evidence.get("rule_labels", {})
    if rules and not rules.get("error"):
        print(f"  Rules:   semantic={rules.get('semantic', '?')}"
              f"  performance={rules.get('performance', '?')}"
              f"  risk={rules.get('risk', '?')}")

    print(f"\n  LLM Classification: semantic={classification.get('semantic', '?')}"
          f"  performance={classification.get('performance', '?')}"
          f"  risk={classification.get('risk', '?')}")

    bar_chars = {3: "███", 2: "██░", 1: "█░░", 0: "░░░"}

    print(f"\n  {'Component':<18} {'Semantic':>10} {'Performance':>13} {'Risk':>10}")
    print(f"  {'-'*51}")

    for comp in COMPONENTS:
        comp_data = attributions.get(comp, {})
        scores = []
        for dim in DIMENSIONS:
            importance = _extract_importance(comp_data, dim)
            score = _importance_to_score(importance)
            scores.append(score)

        sem_bar = bar_chars.get(scores[0], "░░░")
        perf_bar = bar_chars.get(scores[1], "░░░")
        risk_bar = bar_chars.get(scores[2], "░░░")

        print(f"  {comp:<18} {sem_bar:>10} {perf_bar:>13} {risk_bar:>10}")

    print()


def attribute_dataset(records: List[Dict[str, Any]], sample_size: int = None,
                      provider: str = "caliper", model: str = None,
                      api_key: str = None) -> List[Dict[str, Any]]:
    """Run attribution analysis on multiple records."""
    selected = records[:sample_size] if sample_size else records
    results = []
    for i, record in enumerate(selected):
        print(f"\n[{i+1}/{len(selected)}] ", end="")
        results.append(attribute_record(record, provider, model, api_key))
    return results


def summarize_attributions(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate attribution scores across records to find overall patterns."""
    totals = {comp: {dim: [] for dim in DIMENSIONS} for comp in COMPONENTS}

    for result in results:
        attributions = result.get("component_attribution", {})
        for comp in COMPONENTS:
            comp_data = attributions.get(comp, {})
            for dim in DIMENSIONS:
                importance = _extract_importance(comp_data, dim)
                totals[comp][dim].append(_importance_to_score(importance))

    summary = {}
    for comp in COMPONENTS:
        summary[comp] = {}
        for dim in DIMENSIONS:
            scores = totals[comp][dim]
            summary[comp][dim] = {
                "mean_score": sum(scores) / len(scores) if scores else 0,
                "high_count": scores.count(3),
                "medium_count": scores.count(2),
                "low_count": scores.count(1),
                "none_count": scores.count(0),
            }

    return summary


def print_summary(summary: Dict[str, Any], total_records: int):
    """Print aggregate attribution summary."""
    print(f"\n{'='*60}")
    print(f"ATTRIBUTION SUMMARY ({total_records} records)")
    print(f"{'='*60}")

    print(f"\n  Mean importance scores (0-3 scale):")
    print(f"  {'Component':<18} {'Semantic':>10} {'Performance':>13} {'Risk':>10}")
    print(f"  {'-'*51}")

    for comp in COMPONENTS:
        comp_data = summary.get(comp, {})
        sem = comp_data.get("semantic", {}).get("mean_score", 0)
        perf = comp_data.get("performance", {}).get("mean_score", 0)
        risk = comp_data.get("risk", {}).get("mean_score", 0)
        print(f"  {comp:<18} {sem:>10.2f} {perf:>13.2f} {risk:>10.2f}")

    print(f"\n{'='*60}")
