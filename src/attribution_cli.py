"""
CLI for LLM-as-judge attribution analysis.

Usage:
    python attribution_cli.py --record 0 --provider caliper
    python attribution_cli.py --sample-size 5 --provider caliper --model qwen3:8b
    python attribution_cli.py --record 0 --provider caliper --output results/attribution.json
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from attribution import attribute_record, attribute_dataset, summarize_attributions, print_summary


def _resolve_api_key(args):
    if args.api_key:
        return args.api_key
    if args.provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    if args.provider == "openai":
        return os.environ.get("OPENAI_API_KEY")
    return ""


def main():
    parser = argparse.ArgumentParser(description="SQLChange LLM Attribution Analysis")

    parser.add_argument("--record", type=int, default=None,
                        help="Single record index from dataset")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Number of records to analyze")
    parser.add_argument("--input", type=str, default="../data/sqlchange_dataset.json")
    parser.add_argument("--provider", default="caliper",
                        choices=["anthropic", "openai", "local", "caliper"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", default=None,
                        help="Save results JSON to this path")

    args = parser.parse_args()
    api_key = _resolve_api_key(args)

    if not os.path.exists(args.input):
        print(f"Error: Dataset not found: {args.input}")
        sys.exit(1)

    with open(args.input) as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records")

    if args.record is not None:
        if args.record >= len(records):
            print(f"Error: Record {args.record} out of range (max {len(records) - 1})")
            sys.exit(1)
        results = [attribute_record(records[args.record], args.provider, args.model, api_key)]
    elif args.sample_size:
        print(f"Analyzing {args.sample_size} records...")
        results = attribute_dataset(records, args.sample_size, args.provider, args.model, api_key)
        summary = summarize_attributions(results)
        print_summary(summary, len(results))
    else:
        print("Error: Specify --record N or --sample-size N")
        sys.exit(1)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output_data = {
            "results": results,
        }
        if len(results) > 1:
            output_data["summary"] = summarize_attributions(results)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
