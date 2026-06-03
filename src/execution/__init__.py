"""
Execution module — synthetic database generation, equivalence checking, and performance benchmarking.

Modules:
    synthetic_db  – builds in-memory SQLite DBs, runs queries, compares outputs
    equivalence   – checks if original and modified queries produce identical results
    performance   – multi-scale timing comparison (small/medium/large row counts)
"""

from .synthetic_db import (
    build_sqlite_db,
    run_query,
    run_query_pair,
    compare_query_outputs,
    prepare_record,
    infer_context_from_query,
)
from .equivalence import check_equivalence
from .performance import compare_performance