# SQLChange
ECS 189G Final Project

## Synthetic SQLite Harness

`src/synthetic_db.py` builds deterministic in-memory SQLite databases from
SQLChange-style records. It is intended to sit after the schema/query LangGraph
step and before the reasoning, performance, and risk nodes.

Primary APIs:

```python
from synthetic_db import build_sqlite_db, run_query, run_query_pair

conn = build_sqlite_db(record, seed=0, rows_per_table=50)
result = run_query(conn, record["original_sql"])
evidence = run_query_pair(record)
```

`run_query_pair` returns original query output, modified query output, timing
signals, structured errors, row-count deltas, and an observed output relation
such as `identical`, `narrower`, `broader`, `different`, or `error`.
