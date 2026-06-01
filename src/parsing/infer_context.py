"""
    Context inference module — infers schema from SQL query when no DDL is provided.
    
    Uses regex-based extraction first (~85-90% accurate) with an LLM fallback
    for ambiguous or complex queries. This is the "no DDL" path in the pipeline
    flowchart.

    This module was missing from the original codebase but was referenced
    by graph_representer.py's python_node_infer_context().
"""

import json
import re
from typing import Any, Dict, Optional

from parsing.parser import get_join_keys, get_where_details
from execution.synthetic_db import infer_context_from_query as _regex_infer
from utils.llm import llm_universal_call_utility


def infer_context(
    sql: str,
    provider: str = "qwen",
    model: str = None,
    api_key: str = None,
) -> Dict[str, Any]:
    """
        Infer schema context from SQL query when no DDL is provided.

        Strategy:
          1. Try regex-based extraction (fast, ~85-90% accurate)
          2. If regex returns empty or low-confidence, fall back to LLM
          3. Merge join keys and where details from parser
    """
    # Step 1: Regex-based extraction using the synthetic_db module
    regex_result = _regex_infer(sql)
    context = regex_result.get("context", {})
    join_keys = get_join_keys(sql)
    where_details = get_where_details(sql)

    confidence = regex_result.get("inference", {}).get("confidence", "low")

    # Step 2: If regex gave us tables and columns, use it
    if context and confidence != "low":
        return {
            "context": context,
            "join_keys": join_keys,
            "where_details": where_details,
            "inference": {
                "method": "regex",
                "confidence": confidence,
            },
        }

    # Step 3: LLM fallback for empty or low-confidence regex
    try:
        prompt = f"""You are a database schema analyst. Given ONLY the SQL query below, infer the minimal schema (tables and their columns with types) needed to run it.

SQL Query:
{sql}

Respond with ONLY this JSON structure. No markdown, no explanation.
{{
  "tables": {{
    "table_name": {{
      "columns": ["col1", "col2"],
      "types": {{"col1": "INT", "col2": "TEXT"}}
    }}
  }}
}}

Rules:
- Include ALL tables referenced in FROM and JOIN clauses
- Include ALL columns referenced anywhere in the query
- Infer types from context (e.g., _id → INT, _date → DATE, _name → TEXT, _amount → DECIMAL)
- Use only these types: INT, TEXT, DATE, DECIMAL, BOOLEAN
"""
        response = llm_universal_call_utility(
            prompt=prompt,
            provider=provider,
            model=model,
            api_key=api_key,
            num_predict=512,
        )

        clean = response.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        llm_context = data.get("tables", {})

        if llm_context:
            return {
                "context": llm_context,
                "join_keys": join_keys,
                "where_details": where_details,
                "inference": {
                    "method": "llm_inferred",
                    "confidence": "medium",
                },
            }
    except Exception as e:
        print(f"Warning: infer_context.py: LLM fallback failed: {e}")

    # Final fallback: return whatever regex gave us, even if low confidence
    return {
        "context": context,
        "join_keys": join_keys,
        "where_details": where_details,
        "inference": {
            "method": "regex_fallback",
            "confidence": "low",
        },
    }
