"""
Recommendation engine for SQLChange.

Takes two SQL queries + optional schema DDL, gathers structural,
graph, and execution evidence, then makes 3 LLM calls to produce
a unified semantic / performance / risk assessment.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from parser import parse_sql, get_join_keys, get_where_details
from graph_representer import build_graph, llm_universal_call_utility
from synthetic_db import run_query_pair
from performance import compare_performance

DIMENSIONS = {
    "semantic": {
        "labels": "equivalent, narrower, broader, different",
        "instruction": (
            "Classify how the modified query's result set relates to the original. "
            "'equivalent' = same rows, 'narrower' = subset, 'broader' = superset, "
            "'different' = neither subset nor superset."
        ),
    },
    "performance": {
        "labels": "improves, degrades, neutral, unknown",
        "instruction": (
            "Predict whether the modification improves, degrades, or has neutral "
            "effect on query execution time. Use the execution evidence if available."
        ),
    },
    "risk": {
        "labels": "low, medium, high",
        "instruction": (
            "Assess the risk of deploying this modification. Consider: could it "
            "break downstream consumers? Return incorrect data? How severe are "
            "the consequences?"
        ),
    },
}


def _build_record(original_sql, modified_sql, schema_ddl):
    context = parse_sql(schema_ddl) if schema_ddl else {}
    return {
        "context": context,
        "original_sql": original_sql,
        "modified_sql": modified_sql,
        "join_keys": get_join_keys(original_sql),
        "where_details": get_where_details(original_sql),
    }


def _structural_diff(original_sql, modified_sql):
    orig_joins = get_join_keys(original_sql)
    mod_joins = get_join_keys(modified_sql)
    orig_where = get_where_details(original_sql)
    mod_where = get_where_details(modified_sql)
    return {
        "joins_removed": [j for j in orig_joins if j not in mod_joins],
        "joins_added": [j for j in mod_joins if j not in orig_joins],
        "where_removed": [w for w in orig_where if w not in mod_where],
        "where_added": [w for w in mod_where if w not in orig_where],
    }


def _execution_evidence(record):
    try:
        pair = run_query_pair(record, seed=42, rows_per_table=100)
        perf = compare_performance(record, scales={"small": 50, "large": 1000}, repeats=5, seed=42)
        return {"query_pair": pair, "performance": perf}
    except Exception as e:
        return {"error": str(e)}


def _build_prompt(dimension, record, diff, execution, er_graph):
    dim = DIMENSIONS[dimension]
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
        f"{dimension} impact.\n\n{dim['instruction']}\n\n"
        f"Allowed labels: {dim['labels']}\n\n"
        f"Evidence:\n{json.dumps(evidence, indent=2, default=str)}\n\n"
        f'Return ONLY JSON: {{"label": "...", "confidence": 0.0, "rationale": "..."}}'
    )


def _llm_call(prompt, provider, model, api_key):
    raw = llm_universal_call_utility(prompt=prompt, provider=provider, api_key=api_key, model=model)
    text = raw.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def recommend(original_sql, modified_sql, schema_ddl=None,
              provider="anthropic", model="claude-sonnet-4-20250514", api_key=None):
    """Produce a structured recommendation for a SQL modification."""
    record = _build_record(original_sql, modified_sql, schema_ddl)
    diff = _structural_diff(original_sql, modified_sql)

    er_graph = {}
    if record["context"]:
        try:
            out = build_graph(record["context"], record["join_keys"],
                              record["where_details"], model, provider, api_key)
            er_graph = out.get("data_graph", {})
        except Exception:
            pass

    execution = _execution_evidence(record) if record["context"] else {}

    recommendation = {}
    for dim in DIMENSIONS:
        prompt = _build_prompt(dim, record, diff, execution, er_graph)
        try:
            recommendation[dim] = _llm_call(prompt, provider, model, api_key)
        except Exception as e:
            recommendation[dim] = {"label": "unknown", "confidence": 0.0, "rationale": f"LLM error: {e}"}

    return {
        "original_sql": original_sql,
        "modified_sql": modified_sql,
        "structural_diff": diff,
        "er_graph": er_graph,
        "execution_evidence": execution if "error" not in (execution or {}) else None,
        "recommendation": recommendation,
    }


def recommend_from_record(record, provider="anthropic",
                          model="claude-sonnet-4-20250514", api_key=None):
    """Run the recommendation engine on an existing dataset record."""
    return recommend(
        original_sql=record["original_sql"],
        modified_sql=record["modified_sql"],
        schema_ddl=None,
        provider=provider,
        model=model,
        api_key=api_key,
    )
