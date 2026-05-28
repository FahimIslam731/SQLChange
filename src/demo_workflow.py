"""
End-to-end demo of the SQLChange pipeline on a single query.

Stages shown:
    A — schema parse, join keys, WHERE details, ER graph
    A.5 — agentic recommendation (mutations + execution + deterministic pick)
    B — equivalence + multi-scale performance for the chosen mutation
    C — label record with execution evidence (Layer 1 + Layer 2)
    D — per-component attribution grid

Runs fully offline: the LLM utility is patched so provider="none" returns ""
instead of raising. The recommender's final summary node and attribution's
LLM-judge call both degrade gracefully (attribution falls back to its static
MUTATION_ATTRIBUTION_MAP).

Run from inside src/:
    python demo_workflow.py
"""

import json

import pandas as pd

import graph_representer as _gr

_real_llm = _gr.llm_universal_call_utility


def _llm_offline_safe(prompt, provider, api_key=None, model=None, **kwargs):
    if provider == "none":
        return "{}"
    return _real_llm(prompt, provider, api_key=api_key, model=model, **kwargs)


_gr.llm_universal_call_utility = _llm_offline_safe

from parser import parse_sql, get_join_keys, get_where_details
from graph_representer import build_graph
from equivalence import check_equivalence
from performance import compare_performance
from reasoning_pipeline import classify_record
from recommend import recommend
from attribution import attribute_record


CSV_PATH = "../data/queries.csv"
DEMO_ROW = 0
PROVIDER = "none"


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

    # --- STAGE A: structural extraction ---
    banner("STAGE A.1 — parse DDL into schema")
    schema = parse_sql(ddl_context)
    pretty(schema)

    banner("STAGE A.2 — extract join keys")
    join_keys = get_join_keys(original_sql)
    pretty(join_keys)

    banner("STAGE A.3 — extract WHERE dependencies")
    where_details = get_where_details(original_sql)
    pretty(where_details)

    banner("STAGE A.4 — build ER graph (offline path since join_keys exist)")
    er_graph_full = build_graph(schema, join_keys, where_details, model_name="", provider="none", api_key="")
    er_graph = er_graph_full.get("data_graph", {})
    pretty(er_graph)

    # --- STAGE A.5: agentic recommendation ---
    banner("STAGE A.5 — agentic recommendation (recommend.py)")
    print("Generates every applicable mutation, tests each on a synthetic DB,")
    print("then deterministically picks: speedup * correctness * risk_penalty.\n")
    rec_out = recommend(original_sql, ddl_context, provider=PROVIDER, model=None, api_key=None)

    print("\nCandidates tested:")
    for c in rec_out["candidates"]:
        eq = c["equivalence"].get("output_relation", "?")
        perf = c.get("performance", {})
        spd_small = perf.get("small", {}).get("speedup")
        spd_large = perf.get("large", {}).get("speedup")
        risk = c["rules"]["risk"]["label"]
        print(f"  - {c['mutation_type']:<15} eq={eq:<10} "
              f"speedup_small={spd_small if spd_small is None else round(spd_small, 2):<6} "
              f"speedup_large={spd_large if spd_large is None else round(spd_large, 2):<6} "
              f"risk={risk}")

    chosen_sql = rec_out["recommendation"].get("recommended_sql")
    winner = next((c for c in rec_out["candidates"] if c["modified_sql"] == chosen_sql), None)
    if winner is None:
        print("\nNo viable mutation chosen (recommender returned the original SQL).")
        return
    print(f"\nWINNER: {winner['mutation_type']}")
    print(f"modified_sql:\n  {winner['modified_sql']}")

    # Assemble the record using the recommender's pick.
    record = {
        "unique_id": f"demo_{DEMO_ROW}",
        "source_id": str(row.get("id")),
        "domain": row.get("domain"),
        "complexity": row.get("sql_complexity"),
        "context": schema,
        "original_sql": original_sql,
        "mutation_type": winner["mutation_type"],
        "modified_sql": winner["modified_sql"],
        "join_keys": join_keys,
        "where_details": where_details,
        "er_graph": er_graph,
        "semantic_label": None,
        "performance_label": None,
        "risk_label": None,
    }

    # --- STAGE B: synthetic-DB evidence on the chosen mutation ---
    banner("STAGE B.1 — equivalence check on synthetic DB")
    equivalence = check_equivalence(record)
    pretty(equivalence)

    banner("STAGE B.2 — performance check across DB scales")
    perf = compare_performance(record, scales={"small": 50, "medium": 500, "large": 2000}, repeats=10)
    pretty(perf)

    execution_evidence = {
        "equivalence": equivalence,
        "performance": perf,
        "comparison": {
            "output_relation": equivalence["output_relation"],
            "row_count_original": equivalence["row_count_original"],
            "row_count_modified": equivalence["row_count_modified"],
            "both_succeeded": equivalence["output_relation"] != "error",
            "original_error": None,
            "modified_error": None,
        },
    }

    # --- STAGE C: evidence-aware labeling ---
    banner("STAGE C — labeling with execution evidence (Layer 1 + Layer 2)")
    labeled = classify_record(record, provider=PROVIDER, execution_evidence=execution_evidence)
    pretty({
        "semantic_label": labeled["semantic_label"],
        "performance_label": labeled["performance_label"],
        "risk_label": labeled["risk_label"],
        "reasoning": labeled["reasoning"],
        "risk_evidence": labeled.get("risk_evidence"),
    })

    # --- STAGE D: attribution ---
    banner("STAGE D — per-component attribution (attribution.py)")
    print("Asks: which structural component drove each label?")
    print("LLM-as-judge falls back to MUTATION_ATTRIBUTION_MAP when offline.\n")
    attribution = attribute_record(record, provider=PROVIDER, model=None, api_key=None, verbose=False)
    pretty({
        "classification": attribution.get("classification"),
        "component_attribution": attribution.get("component_attribution"),
    })

    banner("DONE")


if __name__ == "__main__":
    main()
