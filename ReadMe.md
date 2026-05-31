# SQL Optimization Pipeline — LangGraph

A full LangGraph pipeline that parses, analyzes, and optimizes SQL queries using Qwen 7B (via Ollama) with iterative improvement.

## Architecture (matches the flowchart)

```
User SQL Query
      │
      ▼
┌─────────────┐
│ Parse &     │  sqlglot AST traversal
│ Extract     │  tables, columns, joins, WHERE
└──────┬──────┘
       │
  DDL provided?
  ┌────┴────┐
  yes       no
  │         │
parse_sql() infer_context()
  │         │
  └────┬────┘
       ▼
┌─────────────┐
│ ER Graph    │  Table parser → join keys? → python/LLM
│ Builder     │  Build importance hierarchy (root/intermediate/leaf)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Recommend   │  LLM generates optimized query
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Execution   │  synthetic_db → in-memory SQLite
│ Harness     │  Test recommended vs original (small / large scales)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Labeling    │  Deterministic rules + LLM refine
│ Pipeline    │  Performance · Risk · Semantic
└──────┬──────┘
       │
  Iteration < N? ──yes──→ loop back to Recommend (with label feedback)
       │
       no
       ▼
  Final Output
  labels + recommendation
```

## Prerequisites

```bash
# Install Ollama and pull Qwen 7B
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5-coder:7b

# Install Python dependencies
pip install langgraph langchain-core sqlglot rich
```

## Usage

### CLI (interactive debugging)

```bash
# Direct SQL input
python cli.py "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id WHERE o.total > 100"

# With DDL schema
python cli.py --ddl "CREATE TABLE orders (id INT, total DECIMAL); CREATE TABLE customers (id INT, name TEXT);" \
  "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id"

# From files
python cli.py --file query.sql --ddl-file schema.sql --iterations 5

# Interactive mode
python cli.py --interactive

# JSON output for scripting
python cli.py --json "SELECT * FROM orders" | jq .improved
```

### API Server

```bash
# Start the API
python api.py --port 5000

# Query it
curl -X POST http://localhost:5000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "sql": "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id WHERE o.total > 100",
    "ddl": "CREATE TABLE orders (id INT, customer_id INT, total DECIMAL); CREATE TABLE customers (id INT, name TEXT);",
    "max_iterations": 3
  }'
```

### Python API

```python
from pipeline import run_pipeline

result = run_pipeline(
    original_sql="SELECT * FROM orders WHERE total > 100",
    ddl_context="CREATE TABLE orders (id INT, total DECIMAL, name TEXT);",
    provider="qwen",
    model="qwen2.5-coder:7b",
    max_iterations=3,
)

print(f"Improved: {result['recommended_sql'] != result['original_sql']}")
print(f"Performance: {result['performance_label']['label']} ({result['performance_label']['score']}/10)")
print(f"Risk: {result['risk_label']['label']} ({result['risk_label']['score']}/10)")
print(f"Semantic: {result['semantic_label']['label']}")
```

## Bugs Fixed from Original Code

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `utils/llm.py` | Unused `StateGraph, END` imports | Removed |
| 2 | `utils/llm.py` | Provider `"qwen"` not handled (raises `ValueError`) | Added `"qwen"` and `"ollama"` as aliases for local Ollama |
| 3 | `utils/llm.py` | `num_predict` kwarg not forwarded to Ollama API | Added to `options` dict in Ollama payload |
| 4 | `execution/equivalence.py` | `from synthetic_db import ...` (broken import) | Changed to `from .synthetic_db import ...` |
| 5 | `reasoning/__init__.py` | Imports from nonexistent `reasoning_pipeline` module | Fixed to import from actual `performance_labeler`, `risk_labeler`, `semantic_labeler` |
| 6 | `reasoning/semantic_labeler.py` | File completely empty (0 bytes) | Implemented full semantic labeler with deterministic + LLM path |
| 7 | `recommendation/__init__.py` | Imports `recommend` but function is `recommend_query` | Fixed to `from .recommend import recommend_query` |
| 8 | `reasoning/performance_labeler.py` | Duplicate `import json` and `import re` | Removed duplicates |
| 9 | `parsing/infer_context.py` | File missing but imported by `graph_representer.py` | Created with regex + LLM fallback implementation |

## Project Structure

```
├── pipeline.py          # Full LangGraph pipeline (main orchestrator)
├── cli.py               # CLI script with step-by-step debug output
├── api.py               # HTTP API server (POST /optimize)
├── parsing/
│   ├── parser.py        # sqlglot-based DDL/query parsing
│   └── infer_context.py # Schema inference when no DDL (NEW)
├── graph/
│   └── graph_representer.py  # ER graph builder (LangGraph sub-pipeline)
├── execution/
│   ├── synthetic_db.py  # In-memory SQLite generation & query runner
│   ├── equivalence.py   # Output equivalence checker (FIXED)
│   └── performance.py   # Multi-scale timing benchmark
├── reasoning/
│   ├── performance_labeler.py  # Performance classification (FIXED)
│   ├── risk_labeler.py         # Risk assessment
│   └── semantic_labeler.py     # Semantic relationship (NEW)
├── recommendation/
│   └── recommend.py     # LLM-driven query optimization
└── utils/
    └── llm.py           # Universal LLM call utility (FIXED)
```
