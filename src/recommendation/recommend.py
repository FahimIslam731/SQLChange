"""
    This python file makes an LLM call to generate an optimized SQL query based on
    all upstream pipeline signals: execution evidence (performance timing, equivalence),
    reasoning labels (performance, risk, semantic), ER graph context (table importance,
    join relationships, cross-table risk), and schema information (column types, tables).
 
    The LLM receives the full context from every prior stage and produces a recommended
    SQL query that improves performance (if possible) while reducing or maintaining risk.
    The recommended query is then validated using sqlglot for syntactic correctness and
    checked for SQLite dialect compatibility.
 
    The score out of 10 represents overall recommendation confidence factoring in the
    quality of the optimization, risk trade-offs, and validation results.
 
    Returns: recommended_sql, confidence, is_valid (sqlglot parse check),
             is_sqlite_valid (SQLite dialect check), rationale, factors
 
    File Name: recommend.py
"""
 
import json
import re
from typing import Any, Dict, List, Optional
 
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
 
from utils.llm import llm_universal_call_utility
 
 
# Defining the allowed recommendation actions
RECOMMENDATION_ACTIONS = {"optimize", "keep_original", "rewrite", "flag_for_review"}
 
 
def recommend_query(
    original_sql: str,
    sql_column_details: Dict[str, Any],
    er_graph: Dict[str, Any],
    join_keys: list,
    where_details: list,
    execution_evidence: Dict[str, Any] = None,
    performance_label: Dict[str, Any] = None,
    risk_label: Dict[str, Any] = None,
    semantic_label: Dict[str, Any] = None,
    provider: str = "qwen",
    model: str = None,
    api_key: str = None,
) -> Dict[str, Any]:
    """
        This function aggregates all upstream pipeline signals and sends them to the
        LLM to generate an optimized SQL query. The LLM considers performance improvement
        opportunities, risk mitigation strategies, semantic preservation, and schema
        constraints to produce a better query. The recommended query is then validated
        using sqlglot for syntactic correctness and checked for SQLite compatibility.
 
        Incorporates execution evidence (timing, row counts, errors), reasoning labels
        (performance score, risk score, semantic relation), ER graph context (table
        importance, join depth, cross-table risk), and full schema information (column
        names, types, join keys, WHERE dependencies) to give the LLM maximum context
        for producing a high-quality recommendation.
    """
    # Building the schema summary from sql_column_details for the LLM
    schema_summary = _build_schema_summary(sql_column_details)
 
    # Extracting the ER graph context for the prompt
    er_context = _build_er_context(er_graph)
 
    # Extracting the execution evidence summary for the prompt
    exec_summary = _build_execution_summary(execution_evidence)
 
    # Extracting the reasoning labels summary for the prompt
    reasoning_summary = _build_reasoning_summary(performance_label, risk_label, semantic_label)
 
    # Building the prompt for the LLM to generate an optimized query
    prompt = f"""You are a senior database optimization engineer. You are given an original SQL query along with full pipeline analysis from execution benchmarks, risk assessment, performance evaluation, semantic analysis, and schema context. Your job is to produce an optimized version of the query that improves performance and/or reduces risk while preserving semantic correctness.
 
Original SQL:
{original_sql}
 
Schema Information:
{schema_summary}
 
Join Key Relationships:
{json.dumps(join_keys, indent=2)}
 
WHERE Dependencies:
{json.dumps(where_details, indent=2)}
 
ER Graph Context:
{er_context}
 
Execution Evidence:
{exec_summary}
 
Upstream Reasoning Labels:
{reasoning_summary}
 
Instructions:
1. Analyze the original query against all provided signals.
2. If performance can be improved (performance score < 7 or label is "degrades"/"neutral"), apply optimizations such as:
   - Adding or restructuring WHERE clauses for better selectivity
   - Reordering joins to filter early (smaller tables or leaf tables first)
   - Replacing SELECT * with explicit column lists
   - Adding LIMIT if the query returns unbounded results
   - Converting correlated subqueries to JOINs where beneficial
   - Using EXISTS instead of IN for subqueries when appropriate
   - Removing redundant DISTINCT or unnecessary ORDER BY
3. If risk is high (risk score >= 7), be conservative with changes. Prefer minimal safe rewrites.
4. If the semantic label is "narrower" or "different", ensure the recommended query preserves the original intent and row coverage.
5. If performance is already good (score >= 8) and risk is low (score <= 3), the original query may not need changes. Return the original query with action "keep_original".
6. The recommended query MUST be valid SQLite syntax. Avoid MySQL/PostgreSQL-specific functions like DATE_SUB, NOW(), CURDATE(), DATE_FORMAT, INTERVAL. Use SQLite equivalents (strftime, date, julianday).
7. The recommended query MUST reference only tables and columns present in the schema information provided above. Do not invent new tables or columns.
8. Assign a confidence score from 1 to 10 where:
   - 1-3: low confidence (uncertain optimization, limited evidence, high risk)
   - 4-6: medium confidence (reasonable optimization with some trade-offs)
   - 7-10: high confidence (clear improvement path, low risk, strong evidence)
 
Respond with ONLY this exact JSON structure. No markdown, no explanation, no text outside the JSON. Keep rationale under 100 words. The recommended_sql must be a single valid SQL SELECT statement.
{{"recommended_sql": "<the optimized SQL query>", "action": "optimize|keep_original|rewrite|flag_for_review", "score": <integer 1-10>, "confidence": "high|medium|low", "optimizations_applied": ["<list of specific optimizations applied>"], "rationale": "<100 words max explaining what was changed and why>"}}"""
 
    try:
        # Getting the response from the LLM with limited tokens
        response = llm_universal_call_utility(
            prompt=prompt,
            provider=provider,
            model=model,
            api_key=api_key,
            num_predict=512,
        )
 
        # Parsing the LLM response into a dictionary
        result = _parse_llm_response(response)
 
        # Validating the recommended SQL using sqlglot
        validation = validate_sql(result["recommended_sql"])
        result["is_valid"] = validation["is_valid"]
        result["parse_errors"] = validation["errors"]
 
        # Checking SQLite dialect compatibility
        sqlite_check = validate_sqlite_compatibility(result["recommended_sql"])
        result["is_sqlite_valid"] = sqlite_check["is_sqlite_valid"]
        result["sqlite_warnings"] = sqlite_check["warnings"]
 
        # Checking if recommended query references only known columns and tables
        schema_check = validate_schema_references(
            result["recommended_sql"], sql_column_details
        )
        result["schema_valid"] = schema_check["schema_valid"]
        result["unknown_references"] = schema_check["unknown_references"]
 
        # Storing the raw LLM output and upstream context alongside the parsed result
        result["llm_raw"] = response
        result["upstream_context"] = {
            "performance_score": (performance_label or {}).get("score"),
            "performance_label": (performance_label or {}).get("label"),
            "risk_score": (risk_label or {}).get("score"),
            "risk_label": (risk_label or {}).get("label"),
            "semantic_label": (semantic_label or {}).get("label"),
            "total_tables": (er_graph or {}).get("total_tables"),
            "cross_table_risk": (er_graph or {}).get("cross_table_risk"),
        }
 
        # If validation failed, downgrade confidence and add warning to rationale
        if not result["is_valid"]:
            result["confidence"] = "low"
            result["rationale"] += f" [WARNING: sqlglot parse failed: {result['parse_errors']}]"
 
        if not result["is_sqlite_valid"]:
            result["confidence"] = "low" if result["confidence"] == "low" else "medium"
            result["rationale"] += f" [WARNING: SQLite compatibility issues: {result['sqlite_warnings']}]"
 
        return result
 
    except Exception as e:
        print(f"Error: recommend.py: LLM call failed: {e}")
        return {
            "recommended_sql": original_sql,
            "action": "keep_original",
            "score": 5,
            "confidence": "low",
            "optimizations_applied": [],
            "rationale": f"LLM call failed: {e}",
            "is_valid": True,
            "parse_errors": [],
            "is_sqlite_valid": True,
            "sqlite_warnings": [],
            "schema_valid": True,
            "unknown_references": [],
            "llm_raw": None,
            "upstream_context": {
                "performance_score": (performance_label or {}).get("score"),
                "performance_label": (performance_label or {}).get("label"),
                "risk_score": (risk_label or {}).get("score"),
                "risk_label": (risk_label or {}).get("label"),
                "semantic_label": (semantic_label or {}).get("label"),
                "total_tables": (er_graph or {}).get("total_tables"),
                "cross_table_risk": (er_graph or {}).get("cross_table_risk"),
            },
        }
 
 
def _parse_llm_response(response: str) -> Dict[str, Any]:
    """
        This function parses the LLM response text and extracts the recommended SQL,
        action, score, confidence, optimizations applied and rationale. Falls back to
        defaults if the response cannot be parsed or contains invalid values. Enforces
        100 word rationale limit.
    """
    # Cleaning up the response and parsing the json
    clean = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            return {
                "recommended_sql": "",
                "action": "flag_for_review",
                "score": 1,
                "confidence": "low",
                "optimizations_applied": [],
                "rationale": "Could not parse LLM response",
            }
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                "recommended_sql": "",
                "action": "flag_for_review",
                "score": 1,
                "confidence": "low",
                "optimizations_applied": [],
                "rationale": "Could not parse LLM response after regex extraction",
            }
 
    # Extracting and cleaning the recommended SQL
    recommended_sql = str(data.get("recommended_sql", "")).strip()
    # Removing any trailing semicolons for consistency with the execution harness
    recommended_sql = recommended_sql.rstrip(";").strip()
 
    # Extracting and validating the action
    action = data.get("action", "flag_for_review")
    if action not in RECOMMENDATION_ACTIONS:
        action = "flag_for_review"
 
    # Extracting and validating the score
    score = data.get("score", 5)
    try:
        score = int(score)
        score = max(1, min(10, score))
    except (TypeError, ValueError):
        score = 5
 
    # Extracting the confidence and validating
    confidence = data.get("confidence", "low")
    if confidence not in ("high", "medium", "low"):
        confidence = "low"
 
    # Extracting the optimizations applied as a list
    optimizations_applied = data.get("optimizations_applied", [])
    if not isinstance(optimizations_applied, list):
        optimizations_applied = [str(optimizations_applied)]
 
    # Enforcing the 100 word rationale limit
    rationale = str(data.get("rationale", "")).strip()
    words = rationale.split()
    if len(words) > 100:
        rationale = " ".join(words[:100])
 
    return {
        "recommended_sql": recommended_sql,
        "action": action,
        "score": score,
        "confidence": confidence,
        "optimizations_applied": optimizations_applied,
        "rationale": rationale,
    }
 
 
def validate_sql(sql: str) -> Dict[str, Any]:
    """
        This function validates the recommended SQL query using the sqlglot library.
        It attempts to parse the query and returns whether the parse was successful
        along with any error messages. This checks general SQL syntactic correctness
        regardless of the target dialect.
    """
    # If the sql is empty then it is not valid
    if not sql or not sql.strip():
        return {"is_valid": False, "errors": ["Empty SQL query"]}
 
    errors = []
    try:
        # Attempting to parse the SQL using sqlglot without specifying a dialect
        parsed = sqlglot.parse_one(sql)
 
        # Checking if the parsed result is a valid SELECT statement
        if not isinstance(parsed, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
            errors.append(f"Expected SELECT statement but got {type(parsed).__name__}")
 
    except ParseError as e:
        errors.append(f"sqlglot parse error: {str(e)}")
    except Exception as e:
        errors.append(f"Unexpected validation error: {str(e)}")
 
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
    }
 
 
def validate_sqlite_compatibility(sql: str) -> Dict[str, Any]:
    """
        This function checks if the recommended SQL query is compatible with SQLite
        dialect. It looks for known MySQL/PostgreSQL constructs that do not work in
        SQLite and flags them as warnings. Also attempts to transpile the query to
        SQLite dialect using sqlglot to catch additional incompatibilities.
    """
    # If the sql is empty then skip the check
    if not sql or not sql.strip():
        return {"is_sqlite_valid": False, "warnings": ["Empty SQL query"]}
 
    warnings = []
    upper_sql = sql.upper()
 
    # Checking for known unsupported constructs in SQLite
    unsupported_patterns = {
        "DATE_SUB": "DATE_SUB is not supported in SQLite, use date() with modifiers",
        "DATE_FORMAT": "DATE_FORMAT is not supported in SQLite, use strftime()",
        "NOW()": "NOW() is not supported in SQLite, use datetime('now')",
        "CURDATE()": "CURDATE() is not supported in SQLite, use date('now')",
        "INTERVAL ": "INTERVAL syntax is not supported in SQLite, use date() with modifiers",
        "DATE_ADD": "DATE_ADD is not supported in SQLite, use date() with modifiers",
        "TIMESTAMPDIFF": "TIMESTAMPDIFF is not supported in SQLite, use julianday()",
        "IF(": "IF() function is not supported in SQLite, use CASE WHEN",
        "IFNULL": "IFNULL is supported in SQLite but consider COALESCE for portability",
        "AUTO_INCREMENT": "AUTO_INCREMENT is not supported in SQLite, use AUTOINCREMENT",
        "ILIKE": "ILIKE is not supported in SQLite, use LIKE with COLLATE NOCASE",
        "FULL OUTER JOIN": "FULL OUTER JOIN is not supported in SQLite",
        "RIGHT JOIN": "RIGHT JOIN is not natively supported in older SQLite versions",
    }
 
    for pattern, warning_message in unsupported_patterns.items():
        if pattern in upper_sql:
            warnings.append(warning_message)
 
    # Attempting to transpile the query to SQLite dialect using sqlglot
    try:
        transpiled = sqlglot.transpile(sql, read="sqlite", write="sqlite")
        if not transpiled:
            warnings.append("sqlglot could not transpile query to SQLite dialect")
    except Exception as e:
        warnings.append(f"sqlglot SQLite transpile error: {str(e)}")
 
    return {
        "is_sqlite_valid": len(warnings) == 0,
        "warnings": warnings,
    }
 
 
def validate_schema_references(
    sql: str, sql_column_details: Dict[str, Any]
) -> Dict[str, Any]:
    """
        This function checks if the recommended SQL query only references tables
        and columns that exist in the provided schema. It extracts table and column
        references from the query and compares them against the sql_column_details
        dictionary from the parsing stage.
    """
    # If the sql or schema is empty then skip the check
    if not sql or not sql.strip() or not sql_column_details:
        return {"schema_valid": True, "unknown_references": []}
 
    unknown_references = []
 
    try:
        # Parsing the recommended SQL to extract table and column references
        parsed = sqlglot.parse_one(sql)
 
        # Extracting all table references from the query
        known_tables = set(sql_column_details.keys())
 
        # Building a set of all known columns across all tables
        known_columns = set()
        for table_info in sql_column_details.values():
            for column in table_info.get("columns", []):
                known_columns.add(column)
 
        # Extracting aliases from the query to avoid false positives
        query_aliases = set()
        for alias_node in parsed.find_all(exp.Alias):
            if alias_node.alias:
                query_aliases.add(alias_node.alias)
        for cte_node in parsed.find_all(exp.CTE):
            if cte_node.alias:
                query_aliases.add(cte_node.alias)
        # Subquery aliases also count
        for subquery_node in parsed.find_all(exp.Subquery):
            if subquery_node.alias:
                query_aliases.add(subquery_node.alias)
 
        # Checking table references against known tables and aliases
        for table_node in parsed.find_all(exp.Table):
            table_name = table_node.name
            if table_name and table_name not in known_tables and table_name not in query_aliases:
                unknown_references.append(f"Unknown table: {table_name}")
 
        # Checking column references against known columns and aliases
        # Time unit keywords that sqlglot misparses as columns
        time_units = {
            "MINUTE", "HOUR", "DAY", "WEEK", "MONTH", "QUARTER", "YEAR",
            "SECOND", "MICROSECOND", "minute", "hour", "day", "week",
            "month", "quarter", "year", "second",
        }
        for column_node in parsed.find_all(exp.Column):
            column_name = column_node.name
            if not column_name or column_name == "*":
                continue
            if column_name in known_columns:
                continue
            if column_name in query_aliases:
                continue
            if column_name in time_units:
                continue
            if column_name.upper() in time_units:
                continue
            unknown_references.append(f"Unknown column: {column_name}")
 
    except Exception:
        # If parsing fails the schema check is inconclusive so we pass
        pass
 
    return {
        "schema_valid": len(unknown_references) == 0,
        "unknown_references": unknown_references,
    }
 
 
def _build_schema_summary(sql_column_details: Dict[str, Any]) -> str:
    """
        This function builds a compact human-readable schema summary from the
        sql_column_details dictionary for inclusion in the LLM prompt. Shows
        each table with its columns and their types.
    """
    if not sql_column_details:
        return "No schema information available."
 
    lines = []
    for table_name, table_info in sql_column_details.items():
        columns = table_info.get("columns", [])
        types = table_info.get("types", {})
        col_entries = []
        for col in columns:
            col_type = types.get(col, "UNKNOWN")
            col_entries.append(f"{col} ({col_type})")
        lines.append(f"  {table_name}: {', '.join(col_entries)}")
 
    return "Tables and columns:\n" + "\n".join(lines)
 
 
def _build_er_context(er_graph: Dict[str, Any]) -> str:
    """
        This function builds a compact ER graph context summary from the er_graph
        dictionary for inclusion in the LLM prompt. Shows table importance hierarchy,
        cross-table risk, graph depth.
    """
    if not er_graph:
        return "No ER graph context available."
 
    table_importance = er_graph.get("table_importance", [])
    cross_table_risk = er_graph.get("cross_table_risk", False)
    graph_depth = er_graph.get("graph_depth", 0)
    total_tables = er_graph.get("total_tables", 0)
    join_where_tables = er_graph.get("join_where_tables", [])
 
    lines = [
        f"- Total tables: {total_tables}",
        f"- Graph depth: {graph_depth}",
        f"- Cross-table risk: {cross_table_risk}",
        f"- Cross-table WHERE conditions: {len(join_where_tables)}",
        f"- Table importance hierarchy:",
    ]
    for entry in table_importance:
        lines.append(f"    {entry.get('table', '?')}: {entry.get('importance', '?')}")
 
    return "\n".join(lines)
 
 
def _build_execution_summary(execution_evidence: Dict[str, Any]) -> str:
    """
        This function builds a compact execution evidence summary from the
        execution_evidence dictionary for inclusion in the LLM prompt. Shows
        timing data, row counts, equivalence results, and error states.
    """
    if not execution_evidence:
        return "No execution evidence available."
 
    lines = []
 
    # Extracting equivalence data
    equivalence = execution_evidence.get("equivalence", {})
    if equivalence:
        lines.append(f"- Output relation: {equivalence.get('output_relation', 'unknown')}")
        lines.append(f"- Original row count: {equivalence.get('row_count_original', 'unknown')}")
        lines.append(f"- Modified row count: {equivalence.get('row_count_modified', 'unknown')}")
        lines.append(f"- Equivalent: {equivalence.get('equivalent', 'unknown')}")
 
    # Extracting performance timing data
    perf = execution_evidence.get("performance", {})
    if perf:
        for scale in ("small", "medium", "large"):
            scale_data = perf.get(scale)
            if isinstance(scale_data, dict):
                original_ms = scale_data.get("original_ms")
                modified_ms = scale_data.get("modified_ms") or scale_data.get("recommended_ms")
                speedup = scale_data.get("speedup")
                lines.append(f"- {scale}: original={original_ms}ms, modified={modified_ms}ms, speedup={speedup}x")
 
    # Extracting error states
    both_succeeded = execution_evidence.get("both_succeeded")
    if both_succeeded is not None:
        lines.append(f"- Both succeeded: {both_succeeded}")
    original_error = execution_evidence.get("original_error")
    if original_error:
        lines.append(f"- Original error: {original_error}")
    modified_error = execution_evidence.get("modified_error")
    if modified_error:
        lines.append(f"- Modified error: {modified_error}")
 
    return "\n".join(lines) if lines else "No execution evidence available."
 
 
def _build_reasoning_summary(
    performance_label: Dict[str, Any],
    risk_label: Dict[str, Any],
    semantic_label: Dict[str, Any],
) -> str:
    """
        This function builds a compact reasoning labels summary from the three
        labeler outputs for inclusion in the LLM prompt. Shows the score, label,
        confidence and rationale from each upstream labeler.
    """
    lines = []
 
    # Extracting performance label summary
    perf = performance_label or {}
    lines.append("Performance Assessment:")
    lines.append(f"  - Score: {perf.get('score', 'N/A')}/10")
    lines.append(f"  - Label: {perf.get('label', 'N/A')}")
    lines.append(f"  - Confidence: {perf.get('confidence', 'N/A')}")
    lines.append(f"  - Rationale: {perf.get('rationale', 'N/A')}")
 
    # Extracting risk label summary
    risk = risk_label or {}
    lines.append("Risk Assessment:")
    lines.append(f"  - Score: {risk.get('score', 'N/A')}/10")
    lines.append(f"  - Label: {risk.get('label', 'N/A')}")
    lines.append(f"  - Confidence: {risk.get('confidence', 'N/A')}")
    lines.append(f"  - Rationale: {risk.get('rationale', 'N/A')}")
 
    # Extracting semantic label summary
    sem = semantic_label or {}
    lines.append("Semantic Assessment:")
    lines.append(f"  - Label: {sem.get('label', 'N/A')}")
    lines.append(f"  - Confidence: {sem.get('confidence', 'N/A')}")
    lines.append(f"  - Rationale: {sem.get('rationale', 'N/A')}")
 
    return "\n".join(lines)