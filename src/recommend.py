"""
SQLChange recommendation engine.

Takes ONE SQL query + schema DDL, generates all valid mutations,
tests each against a synthetic database, and recommends the best
optimization with justification.

Uses every contributor's module:
  - mutation_engine.py  (Dev)    — generates candidate mutations
  - parser.py           (Dev)    — structural parsing
  - graph_representer.py (Fahim) — ER graph construction
  - synthetic_db.py     (Fahim)  — execution evidence
  - equivalence.py      (Fahim)  — correctness checking
  - performance.py      (Fahim)  — multi-scale timing
  - reasoning_pipeline.py (Fahim) — rule-based pre-classification
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from mutation_engine import match_sql_to_mutation, mutation_function_mapping
from parser import parse_sql, get_join_keys, get_where_details, validate_sql_columns
from graph_representer import build_graph, llm_universal_call_utility
from equivalence import check_equivalence
from performance import compare_performance
from reasoning_pipeline import _base_reasoning, _rule_signals


def _generate_candidates(sql, context):
    """Generate all valid mutations of the input query."""
    applicable = match_sql_to_mutation(sql)
    candidates = []
    for mutation_type in applicable:
        mutated = mutation_function_mapping[mutation_type](sql)
        if mutated and mutated.strip() != sql.strip():
            if validate_sql_columns(mutated, context, sql) is not False:
                candidates.append({"mutation_type": mutation_type, "modified_sql": mutated})
    return candidates


def _test_candidate(record):
    """Run equivalence and performance checks on one candidate."""
    try:
        equiv = check_equivalence(record, seed=42, rows_per_table=100)
    except Exception as e:
        equiv = {"equivalent": False, "output_relation": "error", "error": str(e)}

    try:
        perf = compare_performance(record, scales={"small": 50, "large": 1000},
                                   repeats=5, seed=42)
    except Exception as e:
        perf = {"error": str(e)}

    rules = _base_reasoning(record)
    signals = _rule_signals(record)

    return {"equivalence": equiv, "performance": perf, "rules": rules, "signals": signals}


def _build_recommendation_prompt(original_sql, candidates_with_evidence):
    """Build a single LLM prompt that evaluates all candidates and picks the best."""
    summary = []
    for i, c in enumerate(candidates_with_evidence):
        entry = {
            "index": i,
            "mutation_type": c["mutation_type"],
            "modified_sql": c["modified_sql"],
            "equivalent": c["test"]["equivalence"].get("equivalent"),
            "output_relation": c["test"]["equivalence"].get("output_relation"),
            "rule_semantic": c["test"]["rules"]["semantic"]["label"],
            "rule_performance": c["test"]["rules"]["performance"]["label"],
            "rule_risk": c["test"]["rules"]["risk"]["label"],
        }
        if "error" not in c["test"]["performance"]:
            large = c["test"]["performance"].get("large", {})
            entry["speedup_large"] = large.get("speedup")
            entry["original_ms"] = large.get("original_ms")
            entry["modified_ms"] = large.get("modified_ms")
        summary.append(entry)

    return (
        "You are a SQL optimization advisor. A developer wants to optimize their query. "
        "Below is the original query and all candidate mutations that were generated, "
        "tested on synthetic data, and pre-classified by rules.\n\n"
        f"Original SQL:\n{original_sql}\n\n"
        f"Candidates:\n{json.dumps(summary, indent=2)}\n\n"
        "Your task:\n"
        "1. Review each candidate's correctness (equivalent/narrower/broader/different)\n"
        "2. Review each candidate's performance impact (speedup ratio)\n"
        "3. Review each candidate's risk level\n"
        "4. Recommend the BEST candidate, or recommend keeping the original if no "
        "mutation is safe and beneficial\n\n"
        "Return ONLY JSON:\n"
        "{\n"
        '  "recommended_index": <int or null if original is best>,\n'
        '  "recommended_sql": "<the SQL to use>",\n'
        '  "semantic": {"label": "...", "confidence": 0.0, "rationale": "..."},\n'
        '  "performance": {"label": "...", "confidence": 0.0, "rationale": "..."},\n'
        '  "risk": {"label": "...", "confidence": 0.0, "rationale": "..."},\n'
        '  "summary": "one-paragraph justification"\n'
        "}"
    )


def recommend(sql, schema_ddl, provider="anthropic",
              model="claude-sonnet-4-20250514", api_key=None):
    """
    Given ONE SQL query + schema, generate mutations, test them, and recommend
    the best optimization.
    """
    context = parse_sql(schema_ddl)
    join_keys = get_join_keys(sql)
    where_details = get_where_details(sql)

    er_graph = {}
    try:
        out = build_graph(context, join_keys, where_details, model, provider, api_key)
        er_graph = out.get("data_graph", {})
    except Exception:
        pass

    candidates = _generate_candidates(sql, context)
    if not candidates:
        return {
            "original_sql": sql,
            "candidates": [],
            "recommendation": {
                "recommended_index": None,
                "recommended_sql": sql,
                "summary": "No valid mutations could be generated for this query.",
            },
        }

    tested = []
    for c in candidates:
        record = {
            "context": context,
            "original_sql": sql,
            "modified_sql": c["modified_sql"],
            "mutation_type": c["mutation_type"],
            "join_keys": join_keys,
            "where_details": where_details,
            "er_graph": er_graph,
        }
        c["test"] = _test_candidate(record)
        tested.append(c)

    prompt = _build_recommendation_prompt(sql, tested)
    try:
        raw = llm_universal_call_utility(prompt=prompt, provider=provider,
                                         api_key=api_key, model=model)
        text = raw.strip().replace("```json", "").replace("```", "").strip()
        recommendation = json.loads(text)
    except Exception as e:
        recommendation = {
            "recommended_index": None,
            "recommended_sql": sql,
            "summary": f"LLM reasoning failed: {e}",
        }

    return {
        "original_sql": sql,
        "er_graph": er_graph,
        "candidates": [
            {
                "mutation_type": c["mutation_type"],
                "modified_sql": c["modified_sql"],
                "equivalence": c["test"]["equivalence"],
                "performance": c["test"]["performance"],
                "rules": c["test"]["rules"],
            }
            for c in tested
        ],
        "recommendation": recommendation,
    }
