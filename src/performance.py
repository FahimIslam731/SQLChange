import statistics
from synthetic_db import build_sqlite_db, run_query

DEFAULT_SCALES = {"small": 50, "medium": 500, "large": 5000}

def _median_runtime_ms(conn, query, repeats):
    runtimes = [run_query(conn, query)["runtime_ms"] for _ in range(repeats)]
    return statistics.median(runtimes)

def compare_performance(record, scales=None, repeats=10, seed=0):
    scales = scales or DEFAULT_SCALES
    results = {}
    for name, rows_per_table in scales.items():
        conn = build_sqlite_db(record, seed=seed, rows_per_table=rows_per_table)
        try:
            original_ms = _median_runtime_ms(conn, record["original_sql"], repeats)
            modified_ms = _median_runtime_ms(conn, record["modified_sql"], repeats)
        finally:
            conn.close()
        results[name] = {
            "rows_per_table": rows_per_table,
            "original_ms": original_ms,
            "modified_ms": modified_ms,
            "speedup": original_ms / modified_ms if modified_ms else None,
        }
    return results
