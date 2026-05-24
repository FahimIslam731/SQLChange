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
from synthetic_db import build_sqlite_db, run_query, compare_query_outputs
from performance import compare_performance
from reasoning_pipeline import _base_reasoning, _rule_signals

COMPONENTS = ["WHERE", "JOIN", "GROUP_BY", "SELECT_COLUMNS", "LIMIT", "ORDER_BY"]

DIMENSIONS = ("semantic", "performance", "risk")


def _gather_execution_evidence(record: Dict[str, Any]) -> Dict[str, Any]:
    """Run the query pair through synthetic DB for equivalence and performance."""
    evidence = {}
    context = record.get("context", {})
    rows_per_table = 200

    print(f"\n  --- STAGE 1: Synthetic Database ---")
    print(f"  Schema tables: {list(context.keys())}")
    for tbl, info in context.items():
        cols = info.get("columns", [])
        types = info.get("types", {})
        col_strs = [f"{c} ({types.get(c, 'TEXT')})" for c in cols]
        print(f"    {tbl}: {', '.join(col_strs)}")

    print(f"\n  --- STAGE 2: Equivalence Check (seed=42, {rows_per_table} rows/table) ---")
    try:
        conn = build_sqlite_db(record, seed=42, rows_per_table=rows_per_table)
        for tbl in context:
            count = conn.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
            print(f"    {tbl}: {count} rows inserted")

        print(f"\n  Executing original SQL...")
        orig_result = run_query(conn, record.get("original_sql", ""))
        if orig_result.get("error"):
            print(f"    ERROR: {orig_result['error']}")
        else:
            print(f"    Returned {orig_result['row_count']} rows in {orig_result['runtime_ms']:.3f}ms")
            if orig_result["row_count"] > 0 and orig_result["row_count"] <= 5:
                print(f"    Columns: {orig_result['columns']}")
                for row in orig_result["rows"][:3]:
                    print(f"      {dict(row)}")

        print(f"  Executing modified SQL...")
        mod_result = run_query(conn, record.get("modified_sql", ""))
        if mod_result.get("error"):
            print(f"    ERROR: {mod_result['error']}")
        else:
            print(f"    Returned {mod_result['row_count']} rows in {mod_result['runtime_ms']:.3f}ms")
            if mod_result["row_count"] > 0 and mod_result["row_count"] <= 5:
                print(f"    Columns: {mod_result['columns']}")
                for row in mod_result["rows"][:3]:
                    print(f"      {dict(row)}")

        comparison = compare_query_outputs(orig_result, mod_result)
        conn.close()

        equiv = {
            "equivalent": comparison["output_relation"] == "identical",
            "output_relation": comparison["output_relation"],
            "row_count_original": comparison["row_count_original"],
            "row_count_modified": comparison["row_count_modified"],
        }
        evidence["equivalence"] = equiv
        print(f"\n  Equivalence verdict: {equiv['output_relation']}"
              f"  ({equiv['row_count_original']} -> {equiv['row_count_modified']} rows)")
        if comparison.get("original_error") or comparison.get("modified_error"):
            print(f"    orig_err={comparison.get('original_error')}"
                  f"  mod_err={comparison.get('modified_error')}")
    except Exception as e:
        evidence["equivalence"] = {"error": str(e)}
        print(f"    Equivalence failed: {e}")

    print(f"\n  --- STAGE 3: Performance Benchmark (3 scales, 5 repeats) ---")
    try:
        perf = compare_performance(record,
                                   scales={"small": 50, "medium": 500, "large": 2000},
                                   repeats=5, seed=42)
        evidence["performance"] = perf
        for scale in ("small", "medium", "large"):
            s = perf.get(scale, {})
            if s.get("speedup") is not None:
                label = "FASTER" if s["speedup"] > 1.05 else ("SLOWER" if s["speedup"] < 0.95 else "NEUTRAL")
                print(f"    {scale:>6} ({s['rows_per_table']} rows): "
                      f"{s['original_ms']:.3f}ms -> {s['modified_ms']:.3f}ms  "
                      f"speedup={s['speedup']:.2f}x  [{label}]")
            else:
                print(f"    {scale:>6}: could not compute speedup")
    except Exception as e:
        evidence["performance"] = {"error": str(e)}
        print(f"    Performance failed: {e}")

    print(f"\n  --- STAGE 4: Rule-Based Labels ---")
    try:
        rules = _base_reasoning(record)
        evidence["rule_labels"] = {dim: rules[dim]["label"] for dim in DIMENSIONS}
        for dim in DIMENSIONS:
            label = rules[dim]["label"]
            conf = rules[dim].get("confidence", "?")
            rationale = rules[dim].get("rationale", "")
            print(f"    {dim:>12}: {label} (confidence={conf})")
            if rationale:
                print(f"                 {rationale[:120]}")
    except Exception as e:
        evidence["rule_labels"] = {"error": str(e)}
        print(f"    Rule labels failed: {e}")

    return evidence


def _format_evidence_block(evidence: Dict[str, Any]) -> str:
    parts = []
    eq = evidence.get("equivalence", {})
    if not eq.get("error"):
        parts.append(f"Execution: {eq.get('output_relation','?')}, "
                     f"rows {eq.get('row_count_original','?')}->{eq.get('row_count_modified','?')}")
    perf = evidence.get("performance", {})
    if not perf.get("error"):
        for scale in ("small", "medium", "large"):
            s = perf.get(scale, {})
            if s.get("speedup"):
                parts.append(f"{scale}: {s['speedup']:.2f}x speedup "
                             f"({s.get('original_ms',0):.1f}ms->{s.get('modified_ms',0):.1f}ms)")
    rules = evidence.get("rule_labels", {})
    if not rules.get("error"):
        parts.append(f"Rule labels: sem={rules.get('semantic','?')} "
                     f"perf={rules.get('performance','?')} risk={rules.get('risk','?')}")
    return "\n".join(parts)


def _build_attribution_prompt(record: Dict[str, Any], evidence: Dict[str, Any] = None) -> str:
    original = record.get("original_sql", "")
    modified = record.get("modified_sql", "")
    mutation_type = record.get("mutation_type", "unknown")
    context = record.get("context", {})

    schema_summary = ", ".join(
        f"{t}({','.join(info.get('columns', []))})"
        for t, info in context.items()
    )

    evidence_block = _format_evidence_block(evidence) if evidence else ""

    return (f"Classify this SQL mutation and rate each component's importance. Respond with JSON only.\n\n"
            f"Mutation: {mutation_type} | Schema: {schema_summary}\n"
            f"Old: {original}\nNew: {modified}\n"
            f"{evidence_block}\n\n"
            f"Respond with this JSON structure:\n"
            f'{{"classification":{{"semantic":"equivalent|narrower|broader|different","performance":"improves|degrades|neutral|unknown","risk":"low|medium|high"}},'
            f'"attribution":{{"WHERE":{{"semantic":"high|medium|low|none","performance":"...","risk":"..."}},'
            f'"JOIN":{{...}},"GROUP_BY":{{...}},"SELECT_COLUMNS":{{...}},"LIMIT":{{...}},"ORDER_BY":{{...}}}}}}')


def _build_batch_prompt(records: List[Dict[str, Any]], evidences: List[Dict[str, Any]]) -> str:
    context = records[0].get("context", {})
    schema_summary = ", ".join(
        f"{t}({','.join(info.get('columns', []))})"
        for t, info in context.items()
    )

    mutations_block = []
    for i, (rec, ev) in enumerate(zip(records, evidences)):
        ev_text = _format_evidence_block(ev) if ev else ""
        mutations_block.append(
            f"Mutation {i+1}: {rec.get('mutation_type', '?')}\n"
            f"Old: {rec.get('original_sql', '')}\n"
            f"New: {rec.get('modified_sql', '')}\n"
            f"{ev_text}"
        )

    return (f"Classify these {len(records)} SQL mutations and rate each component's importance. Respond with JSON only.\n\n"
            f"Schema: {schema_summary}\n\n"
            + "\n\n".join(mutations_block) +
            f"\n\nRespond with a JSON array of {len(records)} objects, one per mutation, each with this structure:\n"
            f'{{"mutation_type":"...","classification":{{"semantic":"equivalent|narrower|broader|different","performance":"improves|degrades|neutral|unknown","risk":"low|medium|high"}},'
            f'"attribution":{{"WHERE":{{"semantic":"high|medium|low|none","performance":"...","risk":"..."}},'
            f'"JOIN":{{...}},"GROUP_BY":{{...}},"SELECT_COLUMNS":{{...}},"LIMIT":{{...}},"ORDER_BY":{{...}}}}}}')


def _parse_response(raw: str) -> Dict[str, Any]:
    import re
    text = raw.strip()
    text = re.sub(r"```\w*\s*", "", text).replace("```", "").strip()
    text = re.sub(r"^[^{]*", "", text, count=1)
    text = re.sub(r"[^}]*$", "", text, count=1)
    text = text.strip()
    if text:
        try:
            result = json.loads(text)
            if isinstance(result, (dict, list)):
                return result
        except json.JSONDecodeError:
            pass
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[^{}]*\{.*?\}.*?\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    print(f"  WARNING: Could not parse JSON from response:\n    {raw[:300]}")
    return {}


IMPORTANCE_SCORES = {"high": 3, "medium": 2, "low": 1, "none": 0}

MUTATION_ATTRIBUTION_MAP = {
    "where_drop":    {"WHERE": {"semantic": "high", "performance": "high", "risk": "high"},
                      "JOIN": {"semantic": "low", "performance": "low", "risk": "low"},
                      "GROUP_BY": {"semantic": "low", "performance": "none", "risk": "low"},
                      "SELECT_COLUMNS": {"semantic": "none", "performance": "none", "risk": "none"},
                      "LIMIT": {"semantic": "medium", "performance": "low", "risk": "medium"},
                      "ORDER_BY": {"semantic": "none", "performance": "none", "risk": "none"}},
    "join_swap":     {"WHERE": {"semantic": "low", "performance": "low", "risk": "low"},
                      "JOIN": {"semantic": "high", "performance": "medium", "risk": "high"},
                      "GROUP_BY": {"semantic": "low", "performance": "none", "risk": "low"},
                      "SELECT_COLUMNS": {"semantic": "none", "performance": "none", "risk": "none"},
                      "LIMIT": {"semantic": "low", "performance": "none", "risk": "low"},
                      "ORDER_BY": {"semantic": "none", "performance": "none", "risk": "none"}},
    "join_drop":     {"WHERE": {"semantic": "medium", "performance": "low", "risk": "medium"},
                      "JOIN": {"semantic": "high", "performance": "high", "risk": "high"},
                      "GROUP_BY": {"semantic": "medium", "performance": "low", "risk": "medium"},
                      "SELECT_COLUMNS": {"semantic": "medium", "performance": "none", "risk": "medium"},
                      "LIMIT": {"semantic": "low", "performance": "none", "risk": "low"},
                      "ORDER_BY": {"semantic": "none", "performance": "none", "risk": "none"}},
    "group_by_drop": {"WHERE": {"semantic": "low", "performance": "none", "risk": "low"},
                      "JOIN": {"semantic": "low", "performance": "none", "risk": "low"},
                      "GROUP_BY": {"semantic": "high", "performance": "high", "risk": "high"},
                      "SELECT_COLUMNS": {"semantic": "medium", "performance": "low", "risk": "medium"},
                      "LIMIT": {"semantic": "medium", "performance": "low", "risk": "low"},
                      "ORDER_BY": {"semantic": "low", "performance": "low", "risk": "low"}},
    "limit_add":     {"WHERE": {"semantic": "none", "performance": "none", "risk": "none"},
                      "JOIN": {"semantic": "none", "performance": "none", "risk": "none"},
                      "GROUP_BY": {"semantic": "none", "performance": "none", "risk": "none"},
                      "SELECT_COLUMNS": {"semantic": "none", "performance": "none", "risk": "none"},
                      "LIMIT": {"semantic": "high", "performance": "high", "risk": "medium"},
                      "ORDER_BY": {"semantic": "medium", "performance": "low", "risk": "low"}},
    "column_drop":   {"WHERE": {"semantic": "low", "performance": "none", "risk": "low"},
                      "JOIN": {"semantic": "none", "performance": "none", "risk": "none"},
                      "GROUP_BY": {"semantic": "low", "performance": "none", "risk": "low"},
                      "SELECT_COLUMNS": {"semantic": "high", "performance": "medium", "risk": "medium"},
                      "LIMIT": {"semantic": "none", "performance": "none", "risk": "none"},
                      "ORDER_BY": {"semantic": "low", "performance": "none", "risk": "none"}},
}


def _infer_attributions_from_mutation(mutation_type: str) -> Dict[str, Any]:
    return MUTATION_ATTRIBUTION_MAP.get(mutation_type, {})


def _importance_to_score(importance: str) -> int:
    return IMPORTANCE_SCORES.get(importance.lower().strip(), 0)


def attribute_record(record: Dict[str, Any], provider: str = "caliper",
                     model: str = None, api_key: str = None,
                     verbose: bool = False) -> Dict[str, Any]:
    """Run execution pipeline then LLM-as-judge attribution on a single record."""
    mut = record.get("mutation_type", "?")

    if verbose:
        print(f"\n{'='*70}")
        print(f"ATTRIBUTION: record {record.get('unique_id', '?')} | mutation={mut}")
        print(f"{'='*70}")
        print(f"  Original SQL: {record.get('original_sql', '')}")
        print(f"  Modified SQL: {record.get('modified_sql', '')}")
        if record.get("join_keys"):
            print(f"  Join keys:    {record['join_keys']}")
        if record.get("where_details"):
            print(f"  WHERE details: {record['where_details']}")

    evidence = _gather_execution_evidence(record)

    prompt = _build_attribution_prompt(record, evidence)

    try:
        raw = llm_universal_call_utility(
            prompt=prompt, provider=provider,
            api_key=api_key, model=model,
            num_predict=1024,
        )
    except Exception as e:
        print(f"  [{mut}] LLM failed: {e}")
        return {
            "unique_id": record.get("unique_id"),
            "mutation_type": mut,
            "original_sql": record.get("original_sql"),
            "modified_sql": record.get("modified_sql"),
            "classification": {},
            "component_attribution": {},
            "error": str(e),
        }

    parsed = _parse_response(raw)
    classification = parsed.get("classification", {})
    attributions = parsed.get("attribution", parsed.get("component_attribution", {}))

    rule_labels = evidence.get("rule_labels", {})
    if not classification and rule_labels and not rule_labels.get("error"):
        classification = {dim: rule_labels[dim] for dim in DIMENSIONS if dim in rule_labels}
        print(f"  [{mut}] LLM returned no JSON, using rule-based labels as fallback")

    if not attributions:
        attributions = _infer_attributions_from_mutation(mut)

    eq = evidence.get("equivalence", {})
    sem = classification.get("semantic", "?")
    perf = classification.get("performance", "?")
    risk = classification.get("risk", "?")
    rows = f"{eq.get('row_count_original', '?')}->{eq.get('row_count_modified', '?')}"
    print(f"  [{mut}] sem={sem} perf={perf} risk={risk} rows={rows}")

    return {
        "unique_id": record.get("unique_id"),
        "mutation_type": mut,
        "original_sql": record.get("original_sql"),
        "modified_sql": record.get("modified_sql"),
        "execution_evidence": evidence,
        "classification": classification,
        "component_attribution": attributions,
        "raw_response": raw,
    }


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


def print_report(results: List[Dict[str, Any]]):
    """Print full attribution report for all results."""
    print(f"\n{'='*70}")
    print(f"ATTRIBUTION REPORT ({len(results)} mutations)")
    print(f"{'='*70}")

    bar_chars = {3: "███", 2: "██░", 1: "█░░", 0: "░░░"}

    for result in results:
        mut = result.get("mutation_type", "?")
        classification = result.get("classification", {})
        attributions = result.get("component_attribution", {})
        evidence = result.get("execution_evidence", {})

        eq = evidence.get("equivalence", {})
        print(f"\n  --- {mut} ---")
        print(f"  Original: {result.get('original_sql', '')}")
        print(f"  Modified: {result.get('modified_sql', '')}")
        if not eq.get("error"):
            print(f"  Equivalence: {eq.get('output_relation', '?')} "
                  f"({eq.get('row_count_original', '?')} -> {eq.get('row_count_modified', '?')} rows)")
        perf = evidence.get("performance", {})
        if not perf.get("error"):
            for scale in ("small", "medium", "large"):
                s = perf.get(scale, {})
                if s.get("speedup") is not None:
                    label = "FASTER" if s["speedup"] > 1.05 else ("SLOWER" if s["speedup"] < 0.95 else "NEUTRAL")
                    print(f"    {scale}: {s['speedup']:.2f}x [{label}]")
        rules = evidence.get("rule_labels", {})
        if rules and not rules.get("error"):
            print(f"  Rules: sem={rules.get('semantic','?')} perf={rules.get('performance','?')} risk={rules.get('risk','?')}")
        print(f"  LLM:   sem={classification.get('semantic', '?')} perf={classification.get('performance', '?')} risk={classification.get('risk', '?')}")

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

    print(f"\n{'='*70}")


def attribute_batch(records: List[Dict[str, Any]], provider: str = "caliper",
                    model: str = None, api_key: str = None) -> List[Dict[str, Any]]:
    """Run execution evidence for all records, then one LLM call for all attributions."""
    print(f"\n{'='*70}")
    print(f"BATCH ATTRIBUTION: {len(records)} mutations")
    print(f"{'='*70}")
    print(f"  Original SQL: {records[0].get('original_sql', '')}")

    evidences = []
    for i, record in enumerate(records):
        print(f"\n  [{i+1}/{len(records)}] mutation={record.get('mutation_type', '?')}")
        print(f"  Modified SQL: {record.get('modified_sql', '')}")
        evidences.append(_gather_execution_evidence(record))

    print(f"\n  --- LLM-as-Judge ({provider}) ---")
    print(f"  Sending {len(records)} mutations in one prompt...")

    prompt = _build_batch_prompt(records, evidences)

    try:
        raw = llm_universal_call_utility(
            prompt=prompt, provider=provider,
            api_key=api_key, model=model,
            num_predict=2048,
        )
    except Exception as e:
        print(f"  LLM call failed: {e}")
        return [{
            "unique_id": r.get("unique_id"),
            "mutation_type": r.get("mutation_type"),
            "classification": {},
            "component_attribution": {},
            "error": str(e),
        } for r in records]

    print(f"  Response received ({len(raw)} chars)")

    parsed = _parse_response(raw)
    if isinstance(parsed, list):
        parsed_list = parsed
    elif isinstance(parsed, dict) and "mutations" in parsed:
        parsed_list = parsed["mutations"]
    elif isinstance(parsed, dict):
        parsed_list = [parsed]
    else:
        parsed_list = []

    results = []
    for i, record in enumerate(records):
        p = parsed_list[i] if i < len(parsed_list) else {}
        classification = p.get("classification", {})
        attributions = p.get("attribution", p.get("component_attribution", {}))
        result = {
            "unique_id": record.get("unique_id"),
            "mutation_type": record.get("mutation_type"),
            "original_sql": record.get("original_sql"),
            "modified_sql": record.get("modified_sql"),
            "execution_evidence": evidences[i],
            "classification": classification,
            "component_attribution": attributions,
        }
        results.append(result)
        print(f"\n  --- {record.get('mutation_type', '?')} ---")
        _print_attribution(result)

    return results


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
