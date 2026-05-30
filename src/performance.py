import statistics
from synthetic_db import build_sqlite_db, run_query

DEFAULT_SCALES = {"small": 50, "medium": 500, "large": 5000}

def _median_runtime_ms(conn, query, repeats):
    runtimes = []
    errors = []
    for _ in range(repeats):
        result = run_query(conn, query)
        if result["error"]:
            errors.append(result["error"])
            continue
        runtimes.append(result["runtime_ms"])
    return {
        "median_ms": statistics.median(runtimes) if runtimes else None,
        "errors": errors,
        "successful_runs": len(runtimes),
    }

def compare_performance(record, scales=None, repeats=10, seed=0):
    scales = scales or DEFAULT_SCALES
    results = {}
    for name, rows_per_table in scales.items():
        conn = build_sqlite_db(record, seed=seed, rows_per_table=rows_per_table)
        try:
            original = _median_runtime_ms(conn, record.get("original_sql") or record.get("query"), repeats)
            modified = _median_runtime_ms(
                conn,
                record.get("modified_sql") or record.get("recommended_sql") or record.get("query"),
                repeats,
            )
        finally:
            conn.close()
        original_ms = original["median_ms"]
        modified_ms = modified["median_ms"]
        results[name] = {
            "rows_per_table": rows_per_table,
            "original_ms": original_ms,
            "modified_ms": modified_ms,
            "speedup": original_ms / modified_ms if modified_ms else None,
            "original_errors": original["errors"],
            "modified_errors": modified["errors"],
            "successful_original_runs": original["successful_runs"],
            "successful_modified_runs": modified["successful_runs"],
        }
    return results
