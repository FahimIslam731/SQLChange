# SQLChange — `data_extraction` Branch

> **Note for downstream teams:** This branch produces structured, enriched SQL mutation records designed to be consumed directly by semantic, performance, and risk classification engines. The null labels (`risk_label`, `performance_label`, `semantic_label`) at the end of each record are intentional placeholders — they are yours to populate. See [Intended Use for Downstream Teams](#intended-use-for-downstream-teams) for details.

---

## Overview

The `data_extraction` branch is responsible for processing raw SQL query pairs and producing richly annotated records that capture the structural, relational, and semantic context of each mutation. Each record describes a single SQL change — an original query and its mutated variant — along with the schema context, join relationships, WHERE clause dependencies, and an entity-relationship graph that models how tables interact.

The output of this pipeline is a JSON dataset where every entry is self-contained and ready for downstream ML or LLM-based labeling workflows.

---

## What This Branch Produces

Each record in the output dataset contains the following:

| Field | Description |
|---|---|
| `unique_id` | Internal identifier for the record |
| `source_id` | Identifier from the upstream data source |
| `domain` | Industry or subject area of the query (e.g., `retail`, `cannabis industry`) |
| `complexity` | SQL complexity category (e.g., `window functions`, `multiple_joins`) |
| `context` | Schema snapshot — table names, column names, and column types |
| `original_sql` | The query before mutation |
| `mutation_type` | The category of change applied (e.g., `where_drop`, `join_swap`) |
| `modified_sql` | The query after mutation |
| `join_keys` | Explicit join predicates extracted from the SQL, where present |
| `where_details` | Individual WHERE clause conditions with table and column attribution |
| `er_graph` | Entity-relationship graph built from join and schema analysis |
| `risk_label` | Placeholder — to be assigned by the risk classification engine |
| `performance_label` | Placeholder — to be assigned by the performance classification engine |
| `semantic_label` | Placeholder — to be assigned by the semantic classification engine |

---

## Entity-Relationship Graph (`er_graph`)

The `er_graph` field is one of the most significant outputs of this branch. It represents how tables relate to one another, how critical each table is to the query, which WHERE conditions cross table boundaries, and whether the mutation introduces cross-table risk.

### Graph Fields

| Field | Description |
|---|---|
| `join_relationships` | Directed edges between tables, including the join column, confidence level, and how the relationship was derived |
| `table_importance` | Role of each table in the query graph (`root`, `intermediate`, or `leaf`) |
| `where_dependencies` | WHERE conditions mapped to specific tables and columns |
| `join_where_tables` | WHERE conditions that filter on tables also involved in a join |
| `cross_table_risk` | Boolean flag indicating whether the mutation affects logic that spans multiple tables |
| `graph_depth` | Depth of the join chain (number of hops from root to deepest leaf) |
| `total_tables` | Total number of tables involved in the query |

### Relationship Origin: Online vs Offline

There are two modes by which join relationships are established in the graph, and it is important to understand the distinction.

**Offline (join keys present)**

When the SQL contains explicit JOIN predicates, the join columns are extracted directly. The relationship origin is set to `"join_key"` and confidence is `"high"`.

```json
"join_relationships": [
  {
    "source": "retailers",
    "target": "retailer_products",
    "join_column": "id",
    "confidence": "high",
    "origin": "join_key"
  },
  {
    "source": "retailer_products",
    "target": "products",
    "join_column": "product",
    "confidence": "high",
    "origin": "join_key"
  }
]
```

In this example from the retail domain, the query joins `retailers` to `retailer_products` on `retailers.id = retailer_products.retailer_id`, and then joins `retailer_products` to `products` on `retailer_products.product = products.name`. Both relationships are directly observable in the SQL.

**Online (no join keys — LLM inferred)**

When no explicit join predicates are present (for example, a query against a single table with a window function, or a subquery where relationships are implicit), the pipeline uses an LLM to infer likely relationships from the schema context. The relationship origin is set to `"inferred"`.

```json
"join_relationships": [
  {
    "source": "strains",
    "target": "sales",
    "join_column": "id",
    "confidence": "high",
    "origin": "inferred"
  }
]
```

In this example from the cannabis industry domain, the query only references `sales` directly, but the schema contains a `strain_id` foreign key column. The LLM infers that `strains.id` links to `sales.strain_id`, even though no JOIN is written in the SQL. This inference allows the graph to remain structurally complete even for queries that do not use explicit joins.

---

## Record Examples

### Example 1 — Online Relationship (LLM Inferred, No Join Keys)

Domain: Cannabis Industry | Complexity: Window Functions | Mutation: `where_drop`

```json
{
  "unique_id": 31,
  "source_id": 73198,
  "domain": "cannabis industry",
  "complexity": "window functions",
  "context": {
    "strains": {
      "columns": ["id", "name", "type"],
      "types": {"id": "INT", "name": "TEXT", "type": "TEXT"}
    },
    "sales": {
      "columns": ["id", "strain_id", "retail_price", "sale_date", "state"],
      "types": {"id": "INT", "strain_id": "INT", "retail_price": "DECIMAL", "sale_date": "DATE", "state": "TEXT"}
    }
  },
  "original_sql": "SELECT AVG(retail_price) OVER (ORDER BY sale_date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) FROM sales WHERE strain_id = 11 AND state = 'Washington';",
  "mutation_type": "where_drop",
  "modified_sql": "SELECT AVG(retail_price) OVER (ORDER BY sale_date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) FROM sales WHERE strain_id = 11",
  "join_keys": [],
  "where_details": [
    {"condition": "strain_id = 11 AND state = 'Washington'", "table": "UNKNOWN", "column": "strain_id"},
    {"condition": "strain_id = 11 AND state = 'Washington'", "table": "UNKNOWN", "column": "state"}
  ],
  "er_graph": {
    "join_relationships": [
      {"source": "strains", "target": "sales", "join_column": "id", "confidence": "high", "origin": "inferred"}
    ],
    "table_importance": [
      {"table": "strains", "importance": "root"},
      {"table": "sales", "importance": "leaf"}
    ],
    "where_dependencies": [
      {"condition": "strain_id = 11 AND state = 'Washington'", "table": "UNKNOWN", "column": "strain_id"},
      {"condition": "strain_id = 11 AND state = 'Washington'", "table": "UNKNOWN", "column": "state"}
    ],
    "join_where_tables": [],
    "cross_table_risk": false,
    "graph_depth": 2,
    "total_tables": 2
  },
  "risk_label": null,
  "performance_label": null,
  "semantic_label": null
}
```

Because no JOIN is written in the query, `join_keys` is empty and `where_details` records the tables as `UNKNOWN`. The LLM still infers the `strains -> sales` relationship from the schema, keeping the graph meaningful. The dropped `state = 'Washington'` condition does not cross tables, so `cross_table_risk` is `false`.

---

### Example 2 — Offline Relationship (Extracted from Join Keys)

Domain: Retail | Complexity: Multiple Joins | Mutation: `join_swap`

```json
{
  "unique_id": 0,
  "source_id": 32344,
  "domain": "retail",
  "complexity": "multiple_joins",
  "context": {
    "retailers": {
      "columns": ["id", "name", "country"],
      "types": {"id": "INT", "name": "TEXT", "country": "TEXT"}
    },
    "products": {
      "columns": ["id", "name", "is_local"],
      "types": {"id": "INT", "name": "TEXT", "is_local": "BOOLEAN"}
    },
    "retailer_products": {
      "columns": ["retailer_id", "product", "quantity"],
      "types": {"retailer_id": "INT", "product": "TEXT", "quantity": "INT"}
    }
  },
  "original_sql": "SELECT retailers.name FROM retailers LEFT JOIN retailer_products ON retailers.id = retailer_products.retailer_id LEFT JOIN products ON retailer_products.product = products.name WHERE products.is_local IS NULL AND retailers.country = 'Europe';",
  "mutation_type": "join_swap",
  "modified_sql": "SELECT retailers.name FROM retailers LEFT INNER JOIN retailer_products ON retailers.id = retailer_products.retailer_id LEFT JOIN products ON retailer_products.product = products.name WHERE products.is_local IS NULL AND retailers.country = 'Europe'",
  "join_keys": [
    {"right_table": "retailer_products", "left_table": "retailers", "right_column": "retailer_id", "left_column": "id"},
    {"right_table": "products", "left_table": "retailer_products", "right_column": "name", "left_column": "product"}
  ],
  "where_details": [
    {"condition": "products.is_local IS NULL AND retailers.country = 'Europe'", "table": "products", "column": "is_local"},
    {"condition": "products.is_local IS NULL AND retailers.country = 'Europe'", "table": "retailers", "column": "country"}
  ],
  "er_graph": {
    "join_relationships": [
      {"source": "retailers", "target": "retailer_products", "join_column": "id", "confidence": "high", "origin": "join_key"},
      {"source": "retailer_products", "target": "products", "join_column": "product", "confidence": "high", "origin": "join_key"}
    ],
    "table_importance": [
      {"table": "retailers", "importance": "root"},
      {"table": "products", "importance": "leaf"},
      {"table": "retailer_products", "importance": "intermediate"}
    ],
    "where_dependencies": [
      {"condition": "products.is_local IS NULL AND retailers.country = 'Europe'", "table": "products", "column": "is_local"},
      {"condition": "products.is_local IS NULL AND retailers.country = 'Europe'", "table": "retailers", "column": "country"}
    ],
    "join_where_tables": [
      {"condition": "products.is_local IS NULL AND retailers.country = 'Europe'", "table": "products", "column": "is_local"}
    ],
    "cross_table_risk": true,
    "graph_depth": 3,
    "total_tables": 3
  },
  "risk_label": null,
  "performance_label": null,
  "semantic_label": null
}
```

Here all three join relationships are extracted directly from the SQL predicates (`origin: "join_key"`). The WHERE condition filters on `products.is_local`, which is a table also involved in a join — this is captured in `join_where_tables` and causes `cross_table_risk` to be `true`. This flag is a direct signal to the risk engine that the mutation touches semantically sensitive cross-table logic.

---

## Intended Use for Downstream Teams

The three null fields at the end of every record are explicit integration points for the teams building classification engines on top of this dataset:

```json
"risk_label": null,
"performance_label": null,
"semantic_label": null
```

These fields are left unpopulated by design. The `data_extraction` branch is responsible for structural enrichment only. Label assignment is the responsibility of the following consumers:

**Semantic Engine Team**
The `er_graph`, `where_details`, `mutation_type`, and the original/modified SQL pair together provide a rich structural representation of what changed and why. This is sufficient context to build a LangChain-based semantic classification chain that determines whether a mutation changes the logical meaning of the query, narrows or broadens its scope, or is semantically neutral.

**Performance Engine Team**
The `complexity` field, `join_relationships`, `graph_depth`, `total_tables`, and the presence or absence of `join_keys` collectively characterize the computational footprint of the query. A performance engine can use these signals to predict whether a mutation is likely to improve, degrade, or have no effect on query execution.

**Risk Engine Team**
The `cross_table_risk` boolean, `join_where_tables`, `mutation_type`, and `where_dependencies` are the primary signals for risk classification. Mutations that drop WHERE conditions on joined tables, swap join types, or affect columns that participate in both join predicates and filter conditions are structurally higher risk. The `cross_table_risk` field is pre-computed specifically for this use case.

All three engines can be built as LangChain chains or agents that consume the `er_graph` directly. The graph is consistently structured across all records regardless of whether relationships were extracted offline from join keys or inferred online by the LLM.

---

## Mutation Types

The pipeline currently handles the following mutation categories. Each type represents a class of SQL change that may or may not affect query semantics or results.

| Mutation Type | What it does |
|---|---|
| `where_drop` | Drops the rightmost AND condition from the WHERE clause; if only one condition exists, removes the entire WHERE |
| `join_swap` | Swaps the join type — LEFT becomes INNER, anything else becomes LEFT |
| `join_drop` | Removes the last JOIN clause from the query entirely |
| `group_by_drop` | Removes the GROUP BY clause and strips any aggregate functions from the SELECT list |
| `limit_add` | Adds `LIMIT 10` to queries that do not already have a LIMIT clause |
| `column_drop` | Removes the last column from the SELECT list, only applied when more than one column is selected |

---

## Complexity Categories

The `complexity` field classifies each query by the highest-order SQL feature it employs.

| Category | Description |
|---|---|
| `window functions` | Query uses OVER, PARTITION BY, or frame specifications |
| `multiple_joins` | Query joins two or more tables |
| `basic SQL` | Simple SELECT with straightforward filtering |

The complexity field is inherited directly from the source dataset and may include additional categories as the input CSV grows.

---

## Project Layout and File Descriptions

```
SQLChange/
├── data/
│   ├── queries.csv
│   └── sqlchange_dataset.json
├── notebook/
│   └── analysis.ipynb
├── src/
│   ├── cli.py
│   ├── dataset_loader.py
│   ├── graph_representer.py
│   ├── mutation_engine.py
│   └── parser.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Getting Started

Clone the branch and install the dependencies:

```bash
git clone https://github.com/FahimIslam731/SQLChange.git
cd SQLChange
git checkout data_extraction
pip install -r requirements.txt
```

Run the pipeline from inside the `src/` directory:

```bash
# Using Anthropic
cd src
python cli.py --csv ../data/queries.csv --provider anthropic --model claude-sonnet-4-20250514

# Using OpenAI
python cli.py --csv ../data/queries.csv --provider openai --model gpt-4o --api-key sk-...

# Using local Ollama (no API key required)
python cli.py --csv ../data/queries.csv --provider local --model llama3

# Specifying a custom output path
python cli.py --csv ../data/queries.csv --provider anthropic --model claude-sonnet-4-20250514 --output ../data/my_dataset.json
```

### CLI Options

| Flag | Required | Description |
|---|---|---|
| `--csv` | Yes | Path to the source queries CSV file |
| `--provider` | Yes | LLM provider to use for schema inference — `anthropic`, `openai`, or `local` |
| `--model` | Yes | Model name to use (e.g. `claude-sonnet-4-20250514`, `gpt-4o`, `llama3`) |
| `--api-key` | No | API key for the provider; can also be set via environment variable |
| `--output` | No | Output path for the generated JSON (default: `data/sqlchange_dataset.json`) |

API keys can be passed via `--api-key` or set as environment variables:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

Local Ollama (`--provider local`) does not require an API key.