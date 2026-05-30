"""
Parsing module — SQL parsing, schema extraction, and dataset loading.

Modules:
    parse: sqlglot-based DDL/query parsing, join key & WHERE extraction
"""

from .parser import (
    parse_sql,
    get_join_keys,
    get_where_details,
    validate_sql_columns,
    extract_aliases,
    extract_columns,
)