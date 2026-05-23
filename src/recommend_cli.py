"""
CLI for the SQLChange recommendation engine.

Usage:
    python recommend_cli.py --sql "SELECT ..." --schema "CREATE TABLE ..."
    python recommend_cli.py --record 42 --input ../data/sqlchange_dataset.json
"""

import argparse
import json
import os
import sys

from recommend import recommend


def _resolve_api_key(args):
    if args.api_key:
        return args.api_key
    if args.provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    if args.provider == "openai":
        return os.environ.get("OPENAI_API_KEY")
    return ""


def _print_result(result):
    rec = result["recommendation"]

    print("\n" + "=" * 60)
    print("SQLChange Optimization Report")
    print("=" * 60)
    print(f"\nOriginal:  {result['original_sql'][:100]}")

    if not result["candidates"]:
        print(f"\n{rec['summary']}")
        return

    print(f"\nCandidates tested: {len(result['candidates'])}")
    for i, c in enumerate(result["candidates"]):
        equiv = c["equivalence"]
        print(f"\n  [{i}] {c['mutation_type']}")
        print(f"      SQL: {c['modified_sql'][:80]}...")
        print(f"      Equiv: {equiv.get('output_relation', '?')}"
              f"  Rows: {equiv.get('row_count_original', '?')}"
              f" -> {equiv.get('row_count_modified', '?')}")
        if "error" not in c["performance"]:
            large = c["performance"].get("large", {})
            if large.get("speedup"):
                print(f"      Speedup: {large['speedup']:.2f}x"
                      f" ({large.get('original_ms', 0):.2f}ms"
                      f" -> {large.get('modified_ms', 0):.2f}ms)")
        print(f"      Rules: semantic={c['rules']['semantic']['label']}"
              f"  perf={c['rules']['performance']['label']}"
              f"  risk={c['rules']['risk']['label']}")

    print("\n" + "-" * 60)
    print("LLM RECOMMENDATION")
    print("-" * 60)

    if rec.get("recommended_sql"):
        print(f"\n  Recommended SQL: {rec['recommended_sql'][:100]}")

    for dim in ("semantic", "performance", "risk"):
        if dim in rec:
            d = rec[dim]
            print(f"\n  {dim.upper():>12}: {d.get('label', '?')}"
                  f"  (confidence: {d.get('confidence', 0):.2f})")
            print(f"  {'':>13} {d.get('rationale', '')}")

    if rec.get("summary"):
        print(f"\n  Summary: {rec['summary']}")
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="SQLChange Optimization Engine")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sql", type=str, help="SQL query to optimize")
    group.add_argument("--record", type=int, help="Record index from dataset JSON")

    parser.add_argument("--schema", type=str, help="Schema DDL (required with --sql)")
    parser.add_argument("--input", type=str, default="../data/sqlchange_dataset.json")
    parser.add_argument("--provider", default="anthropic",
                        choices=["anthropic", "openai", "local"])
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()
    api_key = _resolve_api_key(args)

    if not api_key and args.provider != "local":
        print(f"Error: No API key for {args.provider}. Set ANTHROPIC_API_KEY or pass --api-key.")
        sys.exit(1)

    if args.sql:
        if not args.schema:
            print("Error: --schema is required with --sql")
            sys.exit(1)
        result = recommend(args.sql, args.schema, args.provider, args.model, api_key)
    else:
        if not os.path.exists(args.input):
            print(f"Error: Dataset not found: {args.input}")
            sys.exit(1)
        with open(args.input) as f:
            records = json.load(f)
        if args.record >= len(records):
            print(f"Error: Record {args.record} out of range (max {len(records) - 1})")
            sys.exit(1)
        r = records[args.record]
        schema_ddl = "\n".join(
            f"CREATE TABLE {t} ({', '.join(f'{c} {info['types'].get(c, 'TEXT')}' for c in info['columns'])})"
            for t, info in r["context"].items()
        )
        result = recommend(r["original_sql"], schema_ddl, args.provider, args.model, api_key)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_result(result)


if __name__ == "__main__":
    main()
