"""
Command line interface for labeling SQLChange mutation records.
"""

import argparse
import json
import os
import sys

from reasoning_pipeline import classify_record


def _resolve_api_key(provider, api_key):
    if api_key is not None:
        return api_key
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY")
    return ""


def _gather_performance_evidence(record, repeats, scales):
    """Run compare_performance() and run_query_pair() for one record.

    Returns an evidence dict with up to two keys:
      "performance"  — timing data from compare_performance() (may be None on error)
      "comparison"   — output-relation data from run_query_pair() at medium scale
                       (may be absent on error)

    A top-level "error" key is set only when BOTH sub-tasks fail; partial failures
    are noted per-key so the caller can still use whatever evidence succeeded.
    """
    evidence = {}
    errors = []

    try:
        from performance import compare_performance
        evidence["performance"] = compare_performance(record, scales=scales, repeats=repeats)
    except Exception as exc:
        evidence["performance"] = None
        errors.append(f"performance: {exc}")

    try:
        from synthetic_db import run_query_pair
        pair_result = run_query_pair(record, seed=0, rows_per_table=500)
        evidence["comparison"] = pair_result["comparison"]
    except Exception as exc:
        errors.append(f"comparison: {exc}")

    if errors and evidence.get("performance") is None and "comparison" not in evidence:
        evidence["error"] = "; ".join(errors)
    elif errors:
        evidence["partial_error"] = "; ".join(errors)

    return evidence


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
    # Execution-evidence flags
    parser.add_argument(
        "--with-execution-evidence",
        dest="with_execution_evidence",
        action="store_true",
        default=True,
        help="Run synthetic-DB timing for each record and use it for performance labels (default: on)",
    )
    parser.add_argument(
        "--no-execution-evidence",
        dest="with_execution_evidence",
        action="store_false",
        help="Skip synthetic-DB timing; use static mutation-type rules only",
    )
    parser.add_argument(
        "--performance-repeats",
        type=int,
        default=5,
        help="Number of timing repeats per scale in compare_performance() (default: 5)",
    )
    parser.add_argument(
        "--limit-records",
        type=int,
        default=None,
        help="Alias for --sample-size; process only the first N records",
    )
    args = parser.parse_args()

    # --limit-records is an alias for --sample-size
    if args.limit_records is not None and args.sample_size is None:
        args.sample_size = args.limit_records

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)
    if args.sample_size is not None and args.sample_size < 0:
        print("Error: --sample-size must be zero or greater")
        sys.exit(1)
    if args.performance_repeats < 1:
        print("Error: --performance-repeats must be at least 1")
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

    selected = records[:args.sample_size] if args.sample_size is not None else records

    perf_scales = {"small": 50, "medium": 500, "large": 5000}
    labeled_records = []
    exec_errors = 0

    for i, record in enumerate(selected):
        execution_evidence = None

        if args.with_execution_evidence:
            evidence = _gather_performance_evidence(
                record, repeats=args.performance_repeats, scales=perf_scales
            )
            if evidence.get("error"):
                exec_errors += 1
                print(
                    f"  [record {i}] execution evidence failed: {evidence['error'][:80]}",
                    file=sys.stderr,
                )
            else:
                execution_evidence = evidence

        labeled = classify_record(
            record,
            provider=args.provider,
            model=args.model,
            api_key=api_key,
            execution_evidence=execution_evidence,
            use_execution_evidence=args.with_execution_evidence,
        )
        labeled_records.append(labeled)

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output, "w") as f:
        json.dump(labeled_records, f, indent=2, default=str)

    print(f"Labeled records: {len(labeled_records)}")
    if args.with_execution_evidence:
        print(f"Execution evidence gathered: {len(labeled_records) - exec_errors}/{len(labeled_records)}")
        if exec_errors:
            print(f"Execution errors (fell back to rules): {exec_errors}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
