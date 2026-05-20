"""
Deterministic SQLite data generation and query execution helpers.

This module accepts SQLChange-style records containing a schema context plus
queries, builds an in-memory SQLite database, and returns timing/output evidence
for downstream reasoning nodes.
"""

from __future__ import annotations

import random
import re
import sqlite3
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple


UNSUPPORTED_SQL_PATTERNS = (
    "DATE_SUB",
    "DATE_FORMAT",
    "NOW()",
    "CURDATE()",
    "INTERVAL ",
)


def build_sqlite_db(
    record: Dict[str, Any],
    seed: int = 0,
    rows_per_table: int = 50,
) -> sqlite3.Connection:
    """Create and populate an in-memory SQLite database for one record."""
    context = _get_context(record)
    if not context:
        raise ValueError("record must include a non-empty context schema")
    if rows_per_table < 1:
        raise ValueError("rows_per_table must be at least 1")

    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    table_rows = _generate_table_rows(context, rows_per_table, rng)
    _apply_join_values(table_rows, context, record.get("join_keys") or [], rows_per_table)
    _apply_where_boundary_values(table_rows, context, record.get("where_details") or [])

    for table_name, table_info in context.items():
        _create_table(conn, table_name, table_info)
        _insert_rows(conn, table_name, table_info["columns"], table_rows[table_name])

    conn.commit()
    return conn


def run_query(conn: sqlite3.Connection, query: str) -> Dict[str, Any]:
    """Run a query and return rows, columns, runtime, normalized SQL, and errors."""
    normalized_query, normalization_error = normalize_sql_for_sqlite(query)
    if normalization_error:
        return {
            "rows": [],
            "columns": [],
            "row_count": 0,
            "runtime_ms": None,
            "error": normalization_error,
            "normalized_query": normalized_query,
        }

    started = time.perf_counter()
    try:
        cursor = conn.execute(normalized_query)
        fetched_rows = cursor.fetchall()
        runtime_ms = (time.perf_counter() - started) * 1000
        columns = [description[0] for description in cursor.description or []]
        rows = [dict(row) for row in fetched_rows]
        return {
            "rows": rows,
            "columns": columns,
            "row_count": len(rows),
            "runtime_ms": runtime_ms,
            "error": None,
            "normalized_query": normalized_query,
        }
    except sqlite3.Error as exc:
        runtime_ms = (time.perf_counter() - started) * 1000
        return {
            "rows": [],
            "columns": [],
            "row_count": 0,
            "runtime_ms": runtime_ms,
            "error": str(exc),
            "normalized_query": normalized_query,
        }


def run_query_pair(
    record: Dict[str, Any],
    seed: int = 0,
    rows_per_table: int = 50,
) -> Dict[str, Any]:
    """Build a DB for a record and run original and modified SQL against it."""
    conn = build_sqlite_db(record, seed=seed, rows_per_table=rows_per_table)
    try:
        original = run_query(conn, record.get("original_sql") or "")
        modified = run_query(conn, record.get("modified_sql") or "")
        comparison = compare_query_outputs(original, modified)
        return {
            "original": original,
            "modified": modified,
            "comparison": comparison,
        }
    finally:
        conn.close()


def normalize_sql_for_sqlite(query: str) -> Tuple[str, Optional[str]]:
    """Normalize small dialect differences, or return a structured error."""
    if not query or not query.strip():
        return query or "", "query is empty"

    normalized = query.strip().rstrip(";")
    upper_query = normalized.upper()
    for pattern in UNSUPPORTED_SQL_PATTERNS:
        if pattern in upper_query:
            return normalized, f"unsupported SQLite construct: {pattern.strip()}"

    normalized = re.sub(
        r"\bYEAR\s*\(\s*([A-Za-z_][\w.]*?)\s*\)",
        r"CAST(strftime('%Y', \1) AS INTEGER)",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\btrue\b", "1", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bfalse\b", "0", normalized, flags=re.IGNORECASE)

    # Some generated mutations accidentally emit "LEFT INNER JOIN"; execute as
    # INNER JOIN so the harness can still measure the intended join swap.
    normalized = re.sub(r"\bLEFT\s+INNER\s+JOIN\b", "INNER JOIN", normalized, flags=re.IGNORECASE)
    return normalized, None


def compare_query_outputs(original: Dict[str, Any], modified: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two query results by rows, row counts, errors, and runtime."""
    original_rows = _canonical_rows(original.get("rows") or [])
    modified_rows = _canonical_rows(modified.get("rows") or [])
    original_set = set(original_rows)
    modified_set = set(modified_rows)

    original_runtime = original.get("runtime_ms")
    modified_runtime = modified.get("runtime_ms")
    runtime_delta_ms = None
    runtime_ratio = None
    if isinstance(original_runtime, (int, float)) and isinstance(modified_runtime, (int, float)):
        runtime_delta_ms = modified_runtime - original_runtime
        if original_runtime > 0:
            runtime_ratio = modified_runtime / original_runtime

    if original.get("error") or modified.get("error"):
        output_relation = "error"
    elif original_rows == modified_rows:
        output_relation = "identical"
    elif modified_set and modified_set.issubset(original_set):
        output_relation = "narrower"
    elif original_set and original_set.issubset(modified_set):
        output_relation = "broader"
    else:
        output_relation = "different"

    return {
        "output_relation": output_relation,
        "row_count_original": original.get("row_count", 0),
        "row_count_modified": modified.get("row_count", 0),
        "row_count_delta": modified.get("row_count", 0) - original.get("row_count", 0),
        "runtime_delta_ms": runtime_delta_ms,
        "runtime_ratio": runtime_ratio,
        "both_succeeded": not original.get("error") and not modified.get("error"),
        "original_error": original.get("error"),
        "modified_error": modified.get("error"),
    }


def _get_context(record: Dict[str, Any]) -> Dict[str, Any]:
    return record.get("context", record)


def _sqlite_type(raw_type: str) -> str:
    upper_type = str(raw_type or "").upper()
    if any(token in upper_type for token in ("INT", "BOOL")):
        return "INTEGER"
    if any(token in upper_type for token in ("REAL", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC")):
        return "REAL"
    return "TEXT"


def _quote_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _create_table(conn: sqlite3.Connection, table_name: str, table_info: Dict[str, Any]) -> None:
    columns = table_info.get("columns") or []
    types = table_info.get("types") or {}
    column_sql = [
        f"{_quote_identifier(column)} {_sqlite_type(types.get(column, 'TEXT'))}"
        for column in columns
    ]
    conn.execute(f"CREATE TABLE {_quote_identifier(table_name)} ({', '.join(column_sql)})")


def _insert_rows(
    conn: sqlite3.Connection,
    table_name: str,
    columns: List[str],
    rows: List[Dict[str, Any]],
) -> None:
    quoted_columns = ", ".join(_quote_identifier(column) for column in columns)
    placeholders = ", ".join("?" for _ in columns)
    values = [[row.get(column) for column in columns] for row in rows]
    conn.executemany(
        f"INSERT INTO {_quote_identifier(table_name)} ({quoted_columns}) VALUES ({placeholders})",
        values,
    )


def _generate_table_rows(
    context: Dict[str, Any],
    rows_per_table: int,
    rng: random.Random,
) -> Dict[str, List[Dict[str, Any]]]:
    table_rows = {}
    for table_name, table_info in context.items():
        columns = table_info.get("columns") or []
        types = table_info.get("types") or {}
        rows = []
        for index in range(rows_per_table):
            row = {}
            for column in columns:
                row[column] = _value_for_column(table_name, column, types.get(column), index, rng)
            rows.append(row)
        table_rows[table_name] = rows
    return table_rows


def _value_for_column(
    table_name: str,
    column: str,
    raw_type: str,
    index: int,
    rng: random.Random,
) -> Any:
    column_lower = column.lower()
    sqlite_type = _sqlite_type(raw_type)

    if index == 0 and not _looks_like_key(column_lower):
        return None
    if column_lower == "id" or column_lower.endswith("_id") or column_lower.endswith("id"):
        return index + 1
    if sqlite_type == "INTEGER":
        if "year" in column_lower:
            return 2020 + (index % 5)
        return (index % 10) + 1
    if sqlite_type == "REAL":
        return round((index + 1) * 10.5 + rng.random(), 2)
    if "date" in column_lower or "time" in column_lower:
        return f"202{index % 5}-{(index % 12) + 1:02d}-{(index % 28) + 1:02d}"
    if "country" in column_lower or "region" in column_lower or "state" in column_lower:
        return ["Europe", "USA", "North America", "Asia", "Washington"][index % 5]
    if "type" in column_lower or "category" in column_lower or "department" in column_lower:
        return ["Agricultural", "Infrastructure", "HR", "Finance", "Local"][index % 5]
    if "content" in column_lower or "description" in column_lower:
        return f"{table_name} content #{index} keyword"
    if "name" in column_lower or "product" in column_lower:
        return f"{column}_{index + 1}"
    return f"{column}_{index + 1}"


def _looks_like_key(column_lower: str) -> bool:
    return column_lower == "id" or column_lower.endswith("_id") or column_lower.endswith("id")


def _apply_join_values(
    table_rows: Dict[str, List[Dict[str, Any]]],
    context: Dict[str, Any],
    join_keys: List[Dict[str, Any]],
    rows_per_table: int,
) -> None:
    match_count = max(1, int(rows_per_table * 0.75))
    for join_key in join_keys:
        left_table = join_key.get("left_table")
        right_table = join_key.get("right_table")
        left_column = join_key.get("left_column")
        right_column = join_key.get("right_column")
        if not _has_column(context, left_table, left_column) or not _has_column(context, right_table, right_column):
            continue

        for index in range(rows_per_table):
            source_value = table_rows[left_table][index][left_column]
            if index < match_count:
                table_rows[right_table][index][right_column] = source_value
            else:
                table_rows[right_table][index][right_column] = _unmatched_value(source_value, index)


def _apply_where_boundary_values(
    table_rows: Dict[str, List[Dict[str, Any]]],
    context: Dict[str, Any],
    where_details: List[Dict[str, Any]],
) -> None:
    for where in where_details:
        table_name = where.get("table")
        column = where.get("column")
        condition = where.get("condition") or ""
        candidate_tables = [table_name] if _has_column(context, table_name, column) else _tables_with_column(context, column)
        for candidate_table in candidate_tables:
            rows = table_rows.get(candidate_table) or []
            if not rows:
                continue
            passing, failing = _boundary_values_for_condition(condition, column)
            if passing is not _NO_VALUE:
                rows[0][column] = passing
            if len(rows) > 1 and failing is not _NO_VALUE:
                rows[1][column] = failing


class _NoValue:
    pass


_NO_VALUE = _NoValue()


def _boundary_values_for_condition(condition: str, column: str) -> Tuple[Any, Any]:
    escaped_column = re.escape(column)

    if re.search(rf"(?:\b\w+\.)?{escaped_column}\s+IS\s+NULL", condition, re.IGNORECASE):
        return None, _fallback_non_null_value(column)
    if re.search(rf"(?:\b\w+\.)?{escaped_column}\s+IS\s+NOT\s+NULL", condition, re.IGNORECASE):
        return _fallback_non_null_value(column), None

    like_match = re.search(rf"(?:\b\w+\.)?{escaped_column}\s+LIKE\s+'([^']*)'", condition, re.IGNORECASE)
    if like_match:
        pattern_text = like_match.group(1).replace("%", "") or "match"
        return f"prefix{pattern_text}suffix", f"not_{column}"

    comparison_match = re.search(
        rf"(?:\b\w+\.)?{escaped_column}\s*(=|>=|<=|>|<)\s*('([^']*)'|[-+]?\d+(?:\.\d+)?)",
        condition,
        re.IGNORECASE,
    )
    if comparison_match:
        operator = comparison_match.group(1)
        literal = comparison_match.group(3) if comparison_match.group(3) is not None else comparison_match.group(2)
        literal_value = _coerce_literal(literal)
        return _passing_value(operator, literal_value), _failing_value(operator, literal_value)

    return _NO_VALUE, _NO_VALUE


def _coerce_literal(literal: str) -> Any:
    try:
        if "." in literal:
            return float(literal)
        return int(literal)
    except ValueError:
        return literal


def _passing_value(operator: str, literal: Any) -> Any:
    if isinstance(literal, (int, float)):
        if operator == ">":
            return literal + 1
        if operator == "<":
            return literal - 1
        return literal
    return literal


def _failing_value(operator: str, literal: Any) -> Any:
    if isinstance(literal, (int, float)):
        if operator in (">", ">="):
            return literal - 1
        if operator in ("<", "<="):
            return literal + 1
        return literal + 1
    return f"not_{literal}"


def _fallback_non_null_value(column: str) -> str:
    return f"{column}_non_null"


def _has_column(context: Dict[str, Any], table_name: Optional[str], column: Optional[str]) -> bool:
    if not table_name or not column or table_name == "UNKNOWN":
        return False
    table_info = context.get(table_name)
    return bool(table_info and column in (table_info.get("columns") or []))


def _tables_with_column(context: Dict[str, Any], column: Optional[str]) -> List[str]:
    if not column:
        return []
    return [
        table_name
        for table_name, table_info in context.items()
        if column in (table_info.get("columns") or [])
    ]


def _unmatched_value(value: Any, index: int) -> Any:
    if isinstance(value, (int, float)):
        return value + 100000 + index
    if value is None:
        return f"unmatched_{index}"
    return f"unmatched_{value}_{index}"


def _canonical_rows(rows: Iterable[Dict[str, Any]]) -> List[Tuple[Tuple[str, Any], ...]]:
    return [tuple(sorted(row.items())) for row in rows]
