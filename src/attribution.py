"""
LLM-as-judge attribution analysis for SQL query modifications.

Sends SQL query pairs to the LLM and asks it to judge which structural
components (WHERE, JOIN, GROUP BY, SELECT columns, LIMIT) were most
important to its semantic/performance/risk assessment.

Designed for Caliper — while this runs, observe the attention weights
in Caliper's endpoint visualizer to see what the model actually attends to
versus what it self-reports.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(__file__))

from graph_representer import llm_universal_call_utility
from parser import parse_sql, get_join_keys, get_where_details

COMPONENTS = ["WHERE", "JOIN", "GROUP_BY", "SELECT_COLUMNS", "LIMIT", "ORDER_BY"]

DIMENSIONS = ("semantic", "performance", "risk")


def _build_attribution_prompt(record: Dict[str, Any]) -> str:
    original = record.get("original_sql", "")
    modified = record.get("modified_sql", "")
    mutation_type = record.get("mutation_type", "unknown")
    context = record.get("context", {})

    schema_summary = ", ".join(
        f"{t}({','.join(info.get('columns', []))})"
        for t, info in context.items()
    )

    return f"""SQL change analysis. Two tasks:

1. Classify this SQL modification:
   - semantic: equivalent, narrower, broader, or different
   - performance: improves, degrades, neutral, or unknown
   - risk: low, medium, or high

2. For each SQL component, rate how important it was to YOUR classification above.
   Use ONLY these importance values: high, medium, low, none.
   "high" = this component was critical to your judgment.
   "none" = this component was irrelevant.

Mutation: {mutation_type}
Schema: {schema_summary}
Original: {original}
Modified: {modified}

Return ONLY valid JSON (no markdown, no explanation):
{{"classification":{{"semantic":"...","performance":"...","risk":"..."}},"attribution":{{"WHERE":{{"semantic":"high|medium|low|none","performance":"high|medium|low|none","risk":"high|medium|low|none"}},"JOIN":{{"semantic":"...","performance":"...","risk":"..."}},"GROUP_BY":{{"semantic":"...","performance":"...","risk":"..."}},"SELECT_COLUMNS":{{"semantic":"...","performance":"...","risk":"..."}},"LIMIT":{{"semantic":"...","performance":"...","risk":"..."}},"ORDER_BY":{{"semantic":"...","performance":"...","risk":"..."}}}}}}"""


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
    """Run LLM-as-judge attribution on a single record."""
    prompt = _build_attribution_prompt(record)

    print(f"\n{'='*60}")
    print(f"ATTRIBUTION: record {record.get('unique_id', '?')} | {record.get('mutation_type', '?')}")
    print(f"  Original: {record.get('original_sql', '')[:80]}...")
    print(f"  Modified: {record.get('modified_sql', '')[:80]}...")
    print(f"  Calling {provider}...")

    try:
        raw = llm_universal_call_utility(
            prompt=prompt, provider=provider,
            api_key=api_key, model=model,
            num_predict=512
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
    """Print a terminal heatmap of component attributions."""
    attributions = result.get("component_attribution", {})
    classification = result.get("classification", {})

    print(f"\n  Classification: semantic={classification.get('semantic', '?')}"
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
