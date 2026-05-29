import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from synthetic_db import (  # noqa: E402
    build_sqlite_db,
    compare_query_outputs,
    infer_context_from_query,
    normalize_sql_for_sqlite,
    prepare_record,
    run_query,
    run_query_pair,
)
from performance import compare_performance  # noqa: E402


def single_table_record():
    return {
        "context": {
            "users": {
                "columns": ["id", "name", "age", "active", "created_at", "balance"],
                "types": {
                    "id": "INT",
                    "name": "TEXT",
                    "age": "INT",
                    "active": "BOOLEAN",
                    "created_at": "DATE",
                    "balance": "DECIMAL",
                },
            }
        },
        "original_sql": "SELECT id, name FROM users WHERE age > 5",
        "modified_sql": "SELECT id, name FROM users WHERE age > 5 LIMIT 2",
        "join_keys": [],
        "where_details": [{"condition": "age > 5", "table": "users", "column": "age"}],
        "er_graph": {},
    }


def join_record():
    return {
        "context": {
            "users": {
                "columns": ["id", "name", "country"],
                "types": {"id": "INT", "name": "TEXT", "country": "TEXT"},
            },
            "orders": {
                "columns": ["id", "user_id", "amount"],
                "types": {"id": "INT", "user_id": "INT", "amount": "DECIMAL"},
            },
        },
        "original_sql": "SELECT users.id, orders.amount FROM users LEFT JOIN orders ON users.id = orders.user_id",
        "modified_sql": "SELECT users.id, orders.amount FROM users INNER JOIN orders ON users.id = orders.user_id",
        "join_keys": [
            {
                "left_table": "users",
                "left_column": "id",
                "right_table": "orders",
                "right_column": "user_id",
            }
        ],
        "where_details": [],
        "er_graph": {},
    }


class SyntheticDbTests(unittest.TestCase):
    def test_single_table_generation_is_deterministic(self):
        record = single_table_record()
        conn_a = build_sqlite_db(record, seed=7, rows_per_table=8)
        conn_b = build_sqlite_db(record, seed=7, rows_per_table=8)
        try:
            rows_a = run_query(conn_a, "SELECT * FROM users ORDER BY id")["rows"]
            rows_b = run_query(conn_b, "SELECT * FROM users ORDER BY id")["rows"]
            self.assertEqual(rows_a, rows_b)
            self.assertEqual(len(rows_a), 8)
            self.assertIn("balance", rows_a[0])
        finally:
            conn_a.close()
            conn_b.close()

    def test_join_generation_has_matching_and_non_matching_rows(self):
        record = join_record()
        conn = build_sqlite_db(record, seed=0, rows_per_table=8)
        try:
            matched = run_query(
                conn,
                "SELECT COUNT(*) AS count FROM users INNER JOIN orders ON users.id = orders.user_id",
            )
            unmatched = run_query(
                conn,
                "SELECT COUNT(*) AS count FROM users LEFT JOIN orders ON users.id = orders.user_id WHERE orders.user_id IS NULL",
            )
            self.assertGreater(matched["rows"][0]["count"], 0)
            self.assertGreater(unmatched["rows"][0]["count"], 0)
        finally:
            conn.close()

    def test_where_boundary_values_include_pass_and_fail_cases(self):
        record = single_table_record()
        conn = build_sqlite_db(record, seed=0, rows_per_table=8)
        try:
            passing = run_query(conn, "SELECT COUNT(*) AS count FROM users WHERE age > 5")
            failing = run_query(conn, "SELECT COUNT(*) AS count FROM users WHERE age <= 5")
            self.assertGreater(passing["rows"][0]["count"], 0)
            self.assertGreater(failing["rows"][0]["count"], 0)
        finally:
            conn.close()

    def test_query_pair_returns_timing_and_comparison(self):
        evidence = run_query_pair(single_table_record(), seed=0, rows_per_table=10)
        self.assertIsNone(evidence["original"]["error"])
        self.assertIsNone(evidence["modified"]["error"])
        self.assertIsInstance(evidence["original"]["runtime_ms"], float)
        self.assertEqual(evidence["comparison"]["output_relation"], "narrower")
        self.assertLessEqual(
            evidence["comparison"]["row_count_modified"],
            evidence["comparison"]["row_count_original"],
        )

    def test_year_rewrite_and_boolean_literals(self):
        normalized, error = normalize_sql_for_sqlite(
            "SELECT * FROM users WHERE YEAR(created_at) = 2020 AND active = true"
        )
        self.assertIsNone(error)
        self.assertIn("strftime('%Y', created_at)", normalized)
        self.assertIn("active = 1", normalized)

    def test_unsupported_sql_returns_structured_error(self):
        conn = build_sqlite_db(single_table_record(), seed=0, rows_per_table=5)
        try:
            result = run_query(
                conn,
                "SELECT * FROM users WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 1 YEAR)",
            )
            self.assertEqual(result["rows"], [])
            self.assertIsNotNone(result["error"])
            self.assertIn("unsupported SQLite construct", result["error"])
        finally:
            conn.close()

    def test_compare_identical_and_different_outputs(self):
        left = {"rows": [{"id": 1}], "row_count": 1, "runtime_ms": 1.0, "error": None}
        same = {"rows": [{"id": 1}], "row_count": 1, "runtime_ms": 2.0, "error": None}
        different = {"rows": [{"id": 2}], "row_count": 1, "runtime_ms": 2.0, "error": None}
        self.assertEqual(compare_query_outputs(left, same)["output_relation"], "identical")
        self.assertEqual(compare_query_outputs(left, different)["output_relation"], "different")

    def test_query_only_record_infers_context_and_runs_window_query(self):
        query = (
            "SELECT AVG(retail_price) OVER (ORDER BY sale_date ROWS BETWEEN 59 PRECEDING "
            "AND CURRENT ROW) FROM sales WHERE strain_id = 11 AND state = 'Washington';"
        )
        record = prepare_record({"query": query})
        self.assertIn("sales", record["context"])
        self.assertEqual(record["context"]["sales"]["types"]["retail_price"], "DECIMAL")
        self.assertEqual(record["context"]["sales"]["types"]["sale_date"], "DATE")
        self.assertEqual(record["context"]["sales"]["types"]["strain_id"], "INT")
        self.assertEqual(record["where_details"][0]["table"], "sales")

        conn = build_sqlite_db(record, seed=0, rows_per_table=12)
        try:
            result = run_query(conn, query)
            self.assertIsNone(result["error"])
            self.assertGreater(result["row_count"], 0)
        finally:
            conn.close()

    def test_query_only_record_infers_join_keys(self):
        query = (
            "SELECT users.name, orders.amount FROM users "
            "JOIN orders ON users.id = orders.user_id WHERE orders.amount > 20"
        )
        inferred = infer_context_from_query(query)
        self.assertIn("users", inferred["context"])
        self.assertIn("orders", inferred["context"])
        self.assertEqual(
            inferred["join_keys"],
            [{
                "left_table": "users",
                "left_column": "id",
                "right_table": "orders",
                "right_column": "user_id",
            }],
        )

    def test_performance_wrapper_handles_query_only_records(self):
        record = {
            "query": "SELECT AVG(retail_price) FROM sales WHERE state = 'Washington'",
        }
        result = compare_performance(record, scales={"small": 10}, repeats=2)
        self.assertIn("small", result)
        self.assertEqual(result["small"]["original_errors"], [])
        self.assertEqual(result["small"]["modified_errors"], [])
        self.assertEqual(result["small"]["successful_original_runs"], 2)


if __name__ == "__main__":
    unittest.main()
