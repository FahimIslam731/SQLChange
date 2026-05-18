"""
Command line interface for labeling SQLChange mutation records.
"""

import argparse
import json
import os
import sys

from reasoning_pipeline import classify_dataset


def _resolve_api_key(provider, api_key):
    if api_key is not None:
        return api_key
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY")
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="SQLChange Reasoning Labeler - fills semantic, performance, and risk labels"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/sqlchange_dataset.json",
        help="Path to the enriched SQLChange dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sqlchange_labeled.json",
        help="Path to write the labeled dataset JSON",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="none",
        choices=["none", "anthropic", "openai", "local"],
        help="Optional LLM provider for rationale/label refinement",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for optional LLM refinement",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the provider; can also be set via environment variable",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Only label the first N records for quick demos",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)
    if args.sample_size is not None and args.sample_size < 0:
        print("Error: --sample-size must be zero or greater")
        sys.exit(1)

    api_key = _resolve_api_key(args.provider, args.api_key)
    if args.provider in {"anthropic", "openai"} and not api_key:
        print(f"Error: No API key provided for {args.provider}")
        print("Pass --api-key or set the provider environment variable.")
        sys.exit(1)
    if args.provider != "none" and not args.model:
        print("Error: --model is required when --provider is not none")
        sys.exit(1)

    with open(args.input) as f:
        records = json.load(f)
    if not isinstance(records, list):
        print("Error: input JSON must contain a list of records")
        sys.exit(1)

    labeled_records = classify_dataset(
        records,
        provider=args.provider,
        model=args.model,
        api_key=api_key,
        sample_size=args.sample_size,
    )

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output, "w") as f:
        json.dump(labeled_records, f, indent=2, default=str)

    print(f"Labeled records: {len(labeled_records)}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
