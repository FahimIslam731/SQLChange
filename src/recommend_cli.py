"""
CLI for the SQLChange recommendation engine.

Usage:
    python recommend_cli.py --original "SELECT ..." --modified "SELECT ..." --schema "CREATE TABLE ..."
    python recommend_cli.py --record 42 --input ../data/sqlchange_dataset.json
"""

import argparse
import json
import os
import sys

from recommend import recommend, recommend_from_record


def _resolve_api_key(args):
    if args.api_key:
        return args.api_key
    if args.provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    if args.provider == "openai":
        return os.environ.get("OPENAI_API_KEY")
    return ""


def _print_recommendation(result):
    rec = result["recommendation"]
    diff = result["structural_diff"]

    print("\n" + "=" * 60)
    print("SQLChange Recommendation")
    print("=" * 60)

    print(f"\nOriginal:  {result['original_sql'][:80]}...")
    print(f"Modified:  {result['modified_sql'][:80]}...")

    if any(diff.values()):
        print("\nStructural changes:")
        for key in ("joins_removed", "joins_added", "where_removed", "where_added"):
            if diff[key]:
                print(f"  {key}: {len(diff[key])}")

    if result.get("execution_evidence") and "query_pair" in result["execution_evidence"]:
        comp = result["execution_evidence"]["query_pair"]["comparison"]
        print(f"\nExecution: {comp['output_relation']} "
              f"(rows {comp['row_count_original']} -> {comp['row_count_modified']})")

    print("\n" + "-" * 60)
    for dim in ("semantic", "performance", "risk"):
        d = rec[dim]
        conf = d.get("confidence", 0)
        print(f"\n{dim.upper():>12}:  {d['label']}  (confidence: {conf:.2f})")
        print(f"{'':>14} {d.get('rationale', '')}")
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="SQLChange Recommendation Engine")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--original", type=str, help="Original SQL query")
    group.add_argument("--record", type=int, help="Record index from dataset JSON")

    parser.add_argument("--modified", type=str, help="Modified SQL query (required with --original)")
    parser.add_argument("--schema", type=str, help="Schema DDL (CREATE TABLE statements)")
    parser.add_argument("--input", type=str, default="../data/sqlchange_dataset.json",
                        help="Dataset JSON for --record mode")
    parser.add_argument("--provider", type=str, default="anthropic",
                        choices=["anthropic", "openai", "local"])
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()
    api_key = _resolve_api_key(args)

    if not api_key and args.provider != "local":
        print(f"Error: No API key for {args.provider}. Set ANTHROPIC_API_KEY or pass --api-key.")
        sys.exit(1)

    if args.original:
        if not args.modified:
            print("Error: --modified is required with --original")
            sys.exit(1)
        result = recommend(args.original, args.modified, args.schema,
                           args.provider, args.model, api_key)
    else:
        if not os.path.exists(args.input):
            print(f"Error: Dataset not found: {args.input}")
            sys.exit(1)
        with open(args.input) as f:
            records = json.load(f)
        if args.record >= len(records):
            print(f"Error: Record {args.record} out of range (max {len(records) - 1})")
            sys.exit(1)
        result = recommend_from_record(records[args.record], args.provider, args.model, api_key)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_recommendation(result)


if __name__ == "__main__":
    main()
