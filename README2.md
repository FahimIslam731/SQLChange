# SQLChange
ECS 189G Final Project

## Synthetic SQLite Harness

`src/synthetic_db.py` builds deterministic in-memory SQLite databases from
SQLChange-style records or raw query-only inputs. It is intended to sit after
the schema/query LangGraph step and before the reasoning, performance, and risk
nodes.

Primary APIs:

```python
from synthetic_db import build_sqlite_db, run_query, run_query_pair

conn = build_sqlite_db(record, seed=0, rows_per_table=50)
result = run_query(conn, record["original_sql"])
evidence = run_query_pair(record)

# Query-only inputs are also supported.
conn = build_sqlite_db({"query": "SELECT AVG(price) FROM sales WHERE state = 'CA'"})
```

`run_query_pair` returns original query output, modified query output, timing
signals, structured errors, row-count deltas, and an observed output relation
such as `identical`, `narrower`, `broader`, `different`, or `error`.

If no `context` is supplied, the harness infers a minimal runnable schema from
the query text, including referenced tables, likely columns/types, equality join
keys, and WHERE dependencies.