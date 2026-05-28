"""
End-to-end demo of the SQLChange pipeline on a single query.

Runs Stage A (dataset enrichment), Stage B (synthetic-DB evaluation), and
Stage C (labeling), printing the artifacts produced at each step.

Run from inside src/:
    python demo_workflow.py
"""

import json

import pandas as pd

from mutation_engine import match_sql_to_mutation, mutation_function_mapping
from parser import parse_sql, get_join_keys, get_where_details
from graph_representer import build_graph
from equivalence import check_equivalence
from performance import compare_performance
from reasoning_pipeline import classify_record


CSV_PATH = "../data/queries.csv"
DEMO_ROW = 0


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def pretty(obj):
    print(json.dumps(obj, indent=2, default=str))


def main():
    df = pd.read_csv(CSV_PATH)
    row = df.iloc[DEMO_ROW]
    original_sql = row["sql"]
    ddl_context = row["sql_context"]

    banner("INPUT: one CSV row")
    print("domain:", row.get("domain"))
    print("complexity:", row.get("sql_complexity"))
    print("\noriginal_sql:\n", original_sql)
    print("\nsql_context (DDL):\n", ddl_context[:400], "..." if len(ddl_context) > 400 else "")

    # --- STAGE A ---
    banner("STAGE A.1 — parse DDL into schema")
    schema = parse_sql(ddl_context)
    pretty(schema)

    banner("STAGE A.2 — extract join keys")
    join_keys = get_join_keys(original_sql)
    pretty(join_keys)

    banner("STAGE A.3 — extract WHERE dependencies")
    where_details = get_where_details(original_sql)
    pretty(where_details)

    banner("STAGE A.4 — detect applicable mutations")
    applicable = match_sql_to_mutation(original_sql)
    print(applicable)

    banner("STAGE A.5 — apply first applicable mutation")
    mutation_type = applicable[0]
    modified_sql = mutation_function_mapping[mutation_type](original_sql)
    print("mutation_type:", mutation_type)
    print("modified_sql:\n", modified_sql)

    banner("STAGE A.6 — build ER graph (offline, since join_keys present)")
    er_graph = build_graph(schema, join_keys, where_details, model_name="", provider="none", api_key="")
    pretty(er_graph)

    record = {
        "unique_id": f"demo_{DEMO_ROW}",
        "source_id": str(row.get("id")),
        "domain": row.get("domain"),
        "complexity": row.get("sql_complexity"),
        "context": schema,
        "original_sql": original_sql,
        "mutation_type": mutation_type,
        "modified_sql": modified_sql,
        "join_keys": join_keys,
        "where_details": where_details,
        "er_graph": er_graph,
        "semantic_label": None,
        "performance_label": None,
        "risk_label": None,
    }

    banner("STAGE A — assembled record (Stage A output)")
    pretty({k: v for k, v in record.items() if k != "er_graph"})

    # --- STAGE B ---
    banner("STAGE B.1 — equivalence check on synthetic DB")
    equivalence = check_equivalence(record)
    pretty(equivalence)

    banner("STAGE B.2 — performance check across DB scales")
    perf = compare_performance(record, scales={"small": 50, "medium": 500, "large": 2000}, repeats=10)
    pretty(perf)

    # --- STAGE C ---
    banner("STAGE C — labeling (rules only, provider='none')")
    labeled = classify_record(record, provider="none")
    pretty({
        "semantic_label": labeled["semantic_label"],
        "performance_label": labeled["performance_label"],
        "risk_label": labeled["risk_label"],
        "reasoning": labeled["reasoning"],
    })

    banner("DONE")


if __name__ == "__main__":
    main()
