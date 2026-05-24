"""
CLI for LLM-as-judge attribution analysis.

Usage:
    python attribution_cli.py --record 0 --provider caliper
    python attribution_cli.py --sql "SELECT ..." --schema "CREATE TABLE ..." --provider caliper
    python attribution_cli.py --sample-size 5 --provider caliper
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

sys.path.insert(0, os.path.dirname(__file__))

from attribution import attribute_record, attribute_dataset, summarize_attributions, print_summary
from mutation_engine import match_sql_to_mutation, mutation_function_mapping
from parser import parse_sql, get_join_keys, get_where_details


def _resolve_api_key(args):
    if args.api_key:
        return args.api_key
    if args.provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    if args.provider == "openai":
        return os.environ.get("OPENAI_API_KEY")
    return ""


def _build_records_from_sql(sql, schema_ddl):
    """Generate all applicable mutation records from a raw SQL + schema."""
    context = parse_sql(schema_ddl)
    join_keys = get_join_keys(sql)
    where_details = get_where_details(sql)

    applicable = match_sql_to_mutation(sql)
    records = []
    for mut_type in applicable:
        mutate_fn = mutation_function_mapping.get(mut_type)
        if not mutate_fn:
            continue
        modified = mutate_fn(sql)
        if not modified or modified.strip() == sql.strip():
            continue
        records.append({
            "unique_id": f"custom_{mut_type}",
            "context": context,
            "original_sql": sql,
            "modified_sql": modified,
            "mutation_type": mut_type,
            "join_keys": join_keys,
            "where_details": where_details,
            "er_graph": {},
            "complexity": "custom",
        })
    return records


def main():
    parser = argparse.ArgumentParser(description="SQLChange LLM Attribution Analysis")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--record", type=int, default=None,
                       help="Single record index from dataset")
    group.add_argument("--sample-size", type=int, default=None,
                       help="Number of records to analyze from dataset")
    group.add_argument("--sql", type=str, default=None,
                       help="Custom SQL query to analyze")

    parser.add_argument("--schema", type=str, default=None,
                        help="Schema DDL (required with --sql)")
    parser.add_argument("--input", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "data", "sqlchange_dataset.json"))
    parser.add_argument("--provider", default="caliper",
                        choices=["anthropic", "openai", "local", "caliper"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", default=None,
                        help="Save results JSON to this path")

    args = parser.parse_args()
    api_key = _resolve_api_key(args)

    if args.sql:
        if not args.schema:
            print("Error: --schema is required with --sql")
            sys.exit(1)
        records = _build_records_from_sql(args.sql, args.schema)
        if not records:
            print("Error: No applicable mutations found for this query")
            sys.exit(1)
        print(f"Generated {len(records)} mutations: {[r['mutation_type'] for r in records]}")
        results = []
        for r in records:
            results.append(attribute_record(r, args.provider, args.model, api_key))
        if len(results) > 1:
            summary = summarize_attributions(results)
            print_summary(summary, len(results))
    elif args.record is not None:
        with open(args.input) as f:
            records = json.load(f)
        print(f"Loaded {len(records)} records")
        if args.record >= len(records):
            print(f"Error: Record {args.record} out of range (max {len(records) - 1})")
            sys.exit(1)
        results = [attribute_record(records[args.record], args.provider, args.model, api_key)]
    else:
        with open(args.input) as f:
            records = json.load(f)
        print(f"Loaded {len(records)} records, analyzing {args.sample_size}...")
        results = attribute_dataset(records, args.sample_size, args.provider, args.model, api_key)
        summary = summarize_attributions(results)
        print_summary(summary, len(results))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output_data = {"results": results}
        if len(results) > 1:
            output_data["summary"] = summarize_attributions(results)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
