"""
    This python file makes an LLM call to analyze the risk level of a SQL query
    change based on the ER graph context produced by the graph_representer module.
    The LLM receives the table importance hierarchy (root, intermediate, leaf),
    join relationships, cross-table dependencies and WHERE conditions to assess
    how risky the query change is.

    The score out of 10 is extracted first and used for downstream analysis of
    different query types and for setting thresholds in the recommendation loop
    to decide if a query change should be accepted or rejected.

    Additionally using factiors such as output relation, original row count, modified
    row count, original error, modified error, both succeded, row cound delta
    
    Labels: low, medium, high
    File Name: risk_labeler.py
"""

import json
import re
from typing import Any, Dict, Optional
 
from utils.llm import llm_universal_call_utility
 
 
# Defining the allowed risk labels
RISK_LABELS = {"low", "medium", "high"}
 
 
def classify_risk(
    original_sql: str,
    er_graph: Dict[str, Any],
    join_keys: list,
    where_details: list,
    execution_evidence: Dict[str, Any] = None,
    provider: str = "qwen",
    model: str = None,
    api_key: str = None,
) -> Dict[str, Any]:
    """
        This function sends the ER graph context and execution evidence to the LLM
        and asks it to analyze the risk level of modifying this query. The LLM
        receives the table importance hierarchy showing which tables are root
        (chain starters), intermediate (both source and target) and leaf (end of
        chain) along with cross-table risk flags, join/where dependencies, and
        concrete execution factors like row counts, errors and success states.
    """
    # Extracting the table importance hierarchy from the er graph
    table_importance = er_graph.get("table_importance", [])
    join_relationships = er_graph.get("join_relationships", [])
    where_dependencies = er_graph.get("where_dependencies", [])
    cross_table_risk = er_graph.get("cross_table_risk", False)
    graph_depth = er_graph.get("graph_depth", 0)
    total_tables = er_graph.get("total_tables", 0)
    join_where_tables = er_graph.get("join_where_tables", [])
 
    # Extracting execution factors from the evidence
    exec_evidence = execution_evidence or {}
    output_relation = exec_evidence.get("output_relation", "unknown")
    row_count_original = exec_evidence.get("row_count_original", "unknown")
    row_count_modified = exec_evidence.get("row_count_modified", "unknown")
    row_count_delta = exec_evidence.get("row_count_delta", "unknown")
    both_succeeded = exec_evidence.get("both_succeeded", "unknown")
    original_error = exec_evidence.get("original_error", None)
    modified_error = exec_evidence.get("modified_error", None)
 
    # Building the prompt for the LLM to analyze the risk
    prompt = f"""You are a database risk analyst. Analyze the entity relationship graph context and execution evidence for this SQL query and determine how risky it would be to modify or optimize this query.
 
Original SQL:
{original_sql}
 
Table Importance Hierarchy:
{json.dumps(table_importance, indent=2)}
 
Table importance levels:
- root: the chain starts from this table, it is never a target in any join. Modifying queries touching root tables affects all downstream data.
- intermediate: the table has both source and target relationships. Changes here can cascade in both directions.
- leaf: the table is only a target, end of the chain. Safest to modify as changes are contained.
 
Join Relationships:
{json.dumps(join_relationships, indent=2)}
 
WHERE Dependencies:
{json.dumps(where_dependencies, indent=2)}
 
Cross-Table Risk: {cross_table_risk}
Cross-Table WHERE Conditions: {json.dumps(join_where_tables, indent=2)}
Graph Depth: {graph_depth}
Total Tables: {total_tables}
 
Execution Factors:
- Output relation: {output_relation}
- Original row count: {row_count_original}
- Modified row count: {row_count_modified}
- Row count delta (modified - original): {row_count_delta}
- Both succeeded: {both_succeeded}
- Original error: {original_error}
- Modified error: {modified_error}
 
Instructions:
1. Assign a risk score from 1 to 10 where:
   - 1-2: very low risk (single leaf table, no joins, simple WHERE, rows match exactly)
   - 3-4: low risk (few tables, leaf-only changes, no cross-table dependencies, rows match)
   - 5-6: medium risk (multiple tables, intermediate table involvement, some cross-table dependencies or minor row count delta)
   - 7-8: high risk (root table involvement, deep graph, cross-table WHERE conditions, row count mismatch or errors present)
   - 9-10: very high risk (multiple root tables, deep joins, heavy cross-table filtering, modified query errors out or produces significantly different row counts)
2. Consider graph depth: deeper graphs mean wider blast radius for changes.
3. Cross-table WHERE conditions are especially risky as they create implicit dependencies.
4. Root tables are the most dangerous to modify as all downstream tables depend on them.
5. If the modified query errors out (modified_error is not None), score 8-10 as the change breaks the query.
6. If row counts diverge significantly (row_count_delta is large), increase the risk score as the modification changes the output.
7. If both queries succeeded and row counts match, reduce risk since the change preserves correctness.
 
Respond with ONLY this exact JSON structure. No markdown, no explanation, no text outside the JSON. Keep rationale under 100 words.
{{"score": <integer 1-10>, "label": "low|medium|high", "confidence": "high|medium|low", "rationale": "<100 words max explaining the risk assessment>"}}"""
 
    try:
        # Getting the response from the LLM with limited tokens
        response = llm_universal_call_utility(
            prompt=prompt,
            provider=provider,
            model=model,
            api_key=api_key,
            num_predict=256,
        )
 
        # Parsing the LLM response into a dictionary
        result = _parse_llm_response(response)
 
        # Storing the raw LLM output and er graph context alongside the parsed result
        result["llm_raw"] = response
        result["er_graph_summary"] = {
            "table_importance": table_importance,
            "cross_table_risk": cross_table_risk,
            "graph_depth": graph_depth,
            "total_tables": total_tables,
            "join_where_count": len(join_where_tables),
        }
 
        return result
 
    except Exception as e:
        print(f"Error: risk_labeler.py: LLM call failed: {e}")
        return {
            "score": 5,
            "label": "medium",
            "confidence": "low",
            "rationale": f"LLM call failed: {e}",
            "llm_raw": None,
            "er_graph_summary": {
                "table_importance": table_importance,
                "cross_table_risk": cross_table_risk,
                "graph_depth": graph_depth,
                "total_tables": total_tables,
            },
        }
 
 
def _parse_llm_response(response: str) -> Dict[str, Any]:
    """
        This function parses the LLM response text and extracts the score, label,
        confidence and rationale. Falls back to defaults if the response cannot
        be parsed or contains invalid values. Enforces 100 word rationale limit.
    """
    # Cleaning up the response and parsing the json
    clean = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            return {"score": 5, "label": "medium", "confidence": "low", "rationale": "Could not parse LLM response"}
        data = json.loads(match.group(0))
 
    # Extracting and validating the score
    score = data.get("score", 5)
    try:
        score = int(score)
        score = max(1, min(10, score))
    except (TypeError, ValueError):
        score = 5
 
    # Extracting and validating the label
    label = data.get("label", "medium")
    if label not in RISK_LABELS:
        label = "medium"
 
    # Extracting the confidence and rationale
    confidence = data.get("confidence", "low")
    if confidence not in ("high", "medium", "low"):
        confidence = "low"
 
    # Enforcing the 100 word rationale limit
    rationale = str(data.get("rationale", "")).strip()
    words = rationale.split()
    if len(words) > 100:
        rationale = " ".join(words[:100])
 
    return {
        "score": score,
        "label": label,
        "confidence": confidence,
        "rationale": rationale,
    }
 