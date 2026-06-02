#!/usr/bin/env python3
"""
Run the SQLChange pipeline on every query in the evaluation dataset
and save structured results to a JSONL file.

Usage:
    python eval/run_eval.py [--dataset eval/eval_dataset.csv] [--output eval/eval_results.jsonl]
                            [--resume] [--limit N] [--timeout 300]
"""

import argparse
import csv
import json
import os
import signal
import sys
import time
from contextlib import contextmanager

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from pipeline import run_pipeline


class TimeoutError(Exception):
    pass


@contextmanager
def timeout(seconds):
    def handler(signum, frame):
        raise TimeoutError(f"Pipeline timed out after {seconds}s")
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def load_completed_ids(output_path):
    ids = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        ids.add(str(record["eval_id"]))
                    except (json.JSONDecodeError, KeyError):
                        pass
    return ids


def extract_result(query_row, pipeline_state, wall_clock, error=None):
    result = {
        "eval_id": str(query_row.get("eval_id", query_row["id"])),
        "query_id": str(query_row["id"]),
        "domain": query_row.get("domain", ""),
        "sql_complexity": query_row.get("sql_complexity", ""),
        "has_ddl": query_row.get("has_ddl", "True") in (True, "True", "true", "1"),
        "is_degraded": query_row.get("is_degraded", "False") in (True, "True", "true", "1"),
        "degradation_type": query_row.get("degradation_type", "none"),
        "ground_truth_sql": query_row.get("ground_truth_sql", ""),
        "original_sql": query_row.get("sql", ""),
        "wall_clock_seconds": round(wall_clock, 2),
        "pipeline_error": error,
    }

    if pipeline_state and not error:
        rec = pipeline_state.get("recommendation", {})
        result["action"] = rec.get("action", "unknown")
        result["recommended_sql"] = pipeline_state.get("recommended_sql", "")
        result["recommendation_score"] = rec.get("score")
        result["recommendation_confidence"] = rec.get("confidence")
        result["recommendation_is_valid"] = rec.get("is_valid")
        result["optimizations_applied"] = rec.get("optimizations_applied", [])
        result["rationale"] = rec.get("rationale", "")

        result["performance_label"] = pipeline_state.get("performance_label", {})
        result["risk_label"] = pipeline_state.get("risk_label", {})
        result["semantic_label"] = pipeline_state.get("semantic_label", {})

        result["execution_evidence"] = pipeline_state.get("execution_evidence", {})
        result["inference_method"] = pipeline_state.get("inference_method")
        result["iterations_used"] = pipeline_state.get("iteration", 0)
        result["iteration_history"] = pipeline_state.get("iteration_history", [])
        result["pipeline_internal_error"] = pipeline_state.get("error")
    else:
        result["action"] = "error"
        result["recommended_sql"] = ""
        result["recommendation_score"] = None
        result["recommendation_confidence"] = None
        result["recommendation_is_valid"] = None
        result["optimizations_applied"] = []
        result["rationale"] = ""
        result["performance_label"] = {}
        result["risk_label"] = {}
        result["semantic_label"] = {}
        result["execution_evidence"] = {}
        result["inference_method"] = None
        result["iterations_used"] = 0
        result["iteration_history"] = []
        result["pipeline_internal_error"] = error

    return result


def run_eval(dataset_path, output_path, resume=False, limit=None, timeout_seconds=300):
    with open(dataset_path) as f:
        queries = list(csv.DictReader(f))

    if limit:
        queries = queries[:limit]

    completed_ids = load_completed_ids(output_path) if resume else set()
    if resume and completed_ids:
        print(f"Resuming: {len(completed_ids)} queries already completed")

    pending = [q for q in queries if str(q["eval_id"]) not in completed_ids]
    total = len(queries)
    done = len(completed_ids)

    print(f"Running evaluation: {len(pending)} queries to process ({done} already done, {total} total)")
    print(f"Output: {output_path}")
    print(f"Timeout: {timeout_seconds}s per query")
    print()

    mode = "a" if resume else "w"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, mode) as out_f:
        for i, query in enumerate(pending):
            idx = done + i + 1
            qid = str(query.get("eval_id", query["id"]))
            complexity = query.get("sql_complexity", "?")
            degraded_tag = " [degraded]" if query.get("is_degraded", "False") in (True, "True", "true", "1") else ""
            ddl_tag = " [no-ddl]" if query.get("has_ddl", "True") in (False, "False", "false", "0") else ""

            print(f"[{idx}/{total}] query {qid} ({complexity}{degraded_tag}{ddl_tag})", end=" ", flush=True)

            start = time.time()
            pipeline_state = None
            error = None

            try:
                ddl = query.get("sql_context", "")
                with timeout(timeout_seconds):
                    pipeline_state = run_pipeline(
                        original_sql=query["sql"],
                        ddl_context=ddl if ddl else None,
                        provider="qwen",
                        max_iterations=3,
                    )
            except TimeoutError as e:
                error = str(e)
            except Exception as e:
                error = f"{type(e).__name__}: {str(e)}"

            elapsed = time.time() - start
            result = extract_result(query, pipeline_state, elapsed, error)

            out_f.write(json.dumps(result, default=str) + "\n")
            out_f.flush()

            if error:
                print(f"— ERROR in {elapsed:.1f}s: {error[:80]}")
            else:
                action = result.get("action", "?")
                perf = result.get("performance_label", {})
                risk = result.get("risk_label", {})
                perf_score = perf.get("score", "?")
                risk_score = risk.get("score", "?")
                print(f"— {elapsed:.1f}s ({action}, perf={perf_score}, risk={risk_score})")

    print(f"\nDone. Results written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SQLChange evaluation")
    parser.add_argument("--dataset", default="eval/eval_dataset.csv")
    parser.add_argument("--output", default="eval/eval_results.jsonl")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed queries")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N queries")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per query in seconds")
    args = parser.parse_args()

    run_eval(args.dataset, args.output, args.resume, args.limit, args.timeout)
