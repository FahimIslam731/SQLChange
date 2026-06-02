#!/usr/bin/env python3
"""
Build a fresh evaluation dataset from gretelai/synthetic_text_to_sql,
excluding queries used during development. Applies stratified sampling,
DDL stripping, and programmatic query degradation.

Usage:
    python eval/build_dataset.py [--seed 42] [--output eval/eval_dataset.csv]
"""

import argparse
import csv
import os
import random
import re
import sys


SIMPLE_LABELS = ["basic SQL", "single join"]
MODERATE_LABELS = ["aggregation", "subqueries"]
COMPLEX_LABELS = ["multiple_joins", "window functions", "set operations", "CTEs"]

SIMPLE_COUNT = 30
MODERATE_COUNT = 30
COMPLEX_COUNT = 20
TOTAL_NORMAL = SIMPLE_COUNT + MODERATE_COUNT + COMPLEX_COUNT  # 80
DEGRADED_COUNT = 20
TOTAL = TOTAL_NORMAL + DEGRADED_COUNT  # 100

NO_DDL_COUNT = 20
NO_DDL_SIMPLE = 8
NO_DDL_MODERATE = 7
NO_DDL_COMPLEX = 5


def load_existing_ids(data_dir="data"):
    ids = set()
    path = os.path.join(data_dir, "queries.csv")
    if os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                ids.add(str(row["id"]))
    return ids


def load_gretelai_dataset():
    from datasets import load_dataset
    print("Loading gretelai/synthetic_text_to_sql from HuggingFace...")
    dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    df = dataset.to_pandas()
    print(f"  Loaded {len(df)} total queries")
    return df


def filter_select_only(df):
    mask = df["sql"].str.strip().str.upper().str.startswith("SELECT")
    filtered = df[mask].copy()
    print(f"  After SELECT-only filter: {len(filtered)} queries")
    return filtered


def stratified_sample(df, exclude_ids, seed):
    rng = random.Random(seed)

    available = df[~df["id"].astype(str).isin(exclude_ids)].copy()
    print(f"  After excluding {len(exclude_ids)} existing IDs: {len(available)} available")

    simple = available[available["sql_complexity"].isin(SIMPLE_LABELS)]
    moderate = available[available["sql_complexity"].isin(MODERATE_LABELS)]
    complex_ = available[available["sql_complexity"].isin(COMPLEX_LABELS)]

    print(f"  Available by tier: simple={len(simple)}, moderate={len(moderate)}, complex={len(complex_)}")

    sampled_simple = simple.sample(n=SIMPLE_COUNT, random_state=seed)
    sampled_moderate = moderate.sample(n=MODERATE_COUNT, random_state=seed)
    sampled_complex = complex_.sample(n=COMPLEX_COUNT, random_state=seed)

    result = []
    for _, row in sampled_simple.iterrows():
        result.append(dict(row))
    for _, row in sampled_moderate.iterrows():
        result.append(dict(row))
    for _, row in sampled_complex.iterrows():
        result.append(dict(row))

    rng.shuffle(result)
    return result


# ── Degradation functions ──

def degrade_select_star(sql):
    pattern = r"(?i)(SELECT\s+)(DISTINCT\s+)?(.*?)(\s+FROM\s+)"
    match = re.search(pattern, sql, re.DOTALL)
    if match and match.group(3).strip() != "*":
        distinct = match.group(2) or ""
        return f"{match.group(1)}{distinct}*{match.group(4)}" + sql[match.end():]
    return None


def degrade_unnecessary_subquery(sql):
    return f"SELECT * FROM ({sql.rstrip(';')}) AS subq;"


def degrade_remove_limit(sql):
    pattern = r"(?i)\s+LIMIT\s+\d+"
    if re.search(pattern, sql):
        return re.sub(pattern, "", sql)
    return None


def degrade_redundant_condition(sql):
    if re.search(r"(?i)\bWHERE\b", sql):
        return re.sub(r"(?i)(\bWHERE\b)", r"\1 1=1 AND", sql)
    return None


DEGRADATIONS = [
    ("select_star", degrade_select_star),
    ("unnecessary_subquery", degrade_unnecessary_subquery),
    ("remove_limit", degrade_remove_limit),
    ("redundant_condition", degrade_redundant_condition),
]


def apply_degradations(queries, count, seed):
    rng = random.Random(seed + 1)
    candidates = list(queries)
    rng.shuffle(candidates)

    degraded = []
    used_indices = set()

    for idx, query in enumerate(candidates):
        if len(degraded) >= count:
            break

        available_degrades = list(DEGRADATIONS)
        rng.shuffle(available_degrades)

        for deg_name, deg_fn in available_degrades:
            result = deg_fn(query["sql"])
            if result is not None:
                degraded_query = dict(query)
                degraded_query["ground_truth_sql"] = query["sql"]
                degraded_query["sql"] = result
                degraded_query["is_degraded"] = True
                degraded_query["degradation_type"] = deg_name
                used_indices.add(idx)
                degraded.append(degraded_query)
                break

    if len(degraded) < count:
        print(f"  Warning: only created {len(degraded)}/{count} degraded queries")

    return degraded


def strip_ddl_stratified(queries, seed):
    rng = random.Random(seed + 2)

    simple = [q for q in queries if q["sql_complexity"] in SIMPLE_LABELS]
    moderate = [q for q in queries if q["sql_complexity"] in MODERATE_LABELS]
    complex_ = [q for q in queries if q["sql_complexity"] in COMPLEX_LABELS]

    rng.shuffle(simple)
    rng.shuffle(moderate)
    rng.shuffle(complex_)

    no_ddl_ids = set()
    for q in simple[:NO_DDL_SIMPLE]:
        no_ddl_ids.add(str(q["id"]))
    for q in moderate[:NO_DDL_MODERATE]:
        no_ddl_ids.add(str(q["id"]))
    for q in complex_[:NO_DDL_COMPLEX]:
        no_ddl_ids.add(str(q["id"]))

    for q in queries:
        if str(q["id"]) in no_ddl_ids:
            q["original_ddl"] = q["sql_context"]
            q["sql_context"] = ""
            q["has_ddl"] = False
        else:
            q["has_ddl"] = True

    return queries


def build_dataset(seed=42, output_path="eval/eval_dataset.csv"):
    exclude_ids = load_existing_ids()
    df = load_gretelai_dataset()
    df = filter_select_only(df)

    print(f"\nSampling {TOTAL_NORMAL} normal queries (stratified)...")
    normal_queries = stratified_sample(df, exclude_ids, seed)

    for q in normal_queries:
        q["is_degraded"] = False
        q["degradation_type"] = "none"
        q["ground_truth_sql"] = q["sql"]

    print(f"\nCreating {DEGRADED_COUNT} degraded queries...")
    degraded_queries = apply_degradations(normal_queries, DEGRADED_COUNT, seed)

    all_queries = normal_queries + degraded_queries
    print(f"\nTotal queries: {len(all_queries)} ({len(normal_queries)} normal + {len(degraded_queries)} degraded)")

    print(f"\nApplying DDL stripping ({NO_DDL_COUNT} queries, stratified)...")
    non_degraded = [q for q in all_queries if not q["is_degraded"]]
    strip_ddl_stratified(non_degraded, seed)
    for q in all_queries:
        if "has_ddl" not in q:
            q["has_ddl"] = True

    rng = random.Random(seed + 3)
    rng.shuffle(all_queries)

    for i, q in enumerate(all_queries):
        suffix = f"_deg_{q['degradation_type']}" if q["is_degraded"] else ""
        q["eval_id"] = f"{q['id']}{suffix}"

    output_columns = [
        "eval_id", "id", "domain", "domain_description", "sql_complexity",
        "sql_complexity_description", "sql_task_type", "sql_task_type_description",
        "sql_prompt", "sql_context", "sql", "sql_explanation",
        "has_ddl", "is_degraded", "degradation_type", "ground_truth_sql",
    ]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_queries)

    print(f"\nDataset written to {output_path}")
    print_summary(all_queries)


def print_summary(queries):
    from collections import Counter

    print("\n── Dataset Summary ──")
    print(f"Total queries: {len(queries)}")

    complexity = Counter(q["sql_complexity"] for q in queries)
    print(f"\nBy complexity:")
    for k, v in complexity.most_common():
        print(f"  {k}: {v}")

    tiers = Counter()
    for q in queries:
        c = q["sql_complexity"]
        if c in SIMPLE_LABELS:
            tiers["simple"] += 1
        elif c in MODERATE_LABELS:
            tiers["moderate"] += 1
        elif c in COMPLEX_LABELS:
            tiers["complex"] += 1
    print(f"\nBy tier: {dict(tiers)}")

    degraded = sum(1 for q in queries if q["is_degraded"])
    print(f"\nDegraded: {degraded}")
    deg_types = Counter(q["degradation_type"] for q in queries if q["is_degraded"])
    for k, v in deg_types.most_common():
        print(f"  {k}: {v}")

    no_ddl = sum(1 for q in queries if not q["has_ddl"])
    print(f"\nNo-DDL queries: {no_ddl}")

    domains = Counter(q["domain"] for q in queries)
    print(f"Unique domains: {len(domains)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build evaluation dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="eval/eval_dataset.csv")
    args = parser.parse_args()

    build_dataset(seed=args.seed, output_path=args.output)
