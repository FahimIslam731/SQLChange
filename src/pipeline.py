"""
    Full SQL Optimization Pipeline — LangGraph implementation.

    Implements the complete flowchart:
      1. Parse & extract (sqlglot AST traversal)
      2. DDL provided? → parse_sql() or infer_context_from_query()
      3. ER graph builder (table parser → join keys? → python/LLM → build graph)
      4. Execution harness (build synthetic DB, time comparison)
      5. Labeling pipeline (deterministic + LLM: performance, risk, semantic)
      6. Recommend (LLM call 3 — suggest optimized query)
      7. Iteration < N? (re-test loop)
      8. Final output (labels + recommendation)

    Uses Qwen 7B via Ollama by default.
"""

import json
import time
import copy
from typing import TypedDict, Optional, Any, Dict, List
from langgraph.graph import StateGraph, END


# ──────────────────────────────────────────────
#  Pipeline State
# ──────────────────────────────────────────────

class PipelineState(TypedDict):
    # ── Inputs ──
    original_sql: str
    ddl_context: Optional[str]
    provider: str
    model: Optional[str]
    api_key: Optional[str]
    max_iterations: int

    # ── Parse & extract outputs ──
    sql_column_details: dict
    join_keys: list
    where_details: list
    inference_method: Optional[str]

    # ── ER graph builder outputs ──
    table_names: list
    join_keys_relationships: list
    er_graph: dict

    # ── Execution harness outputs ──
    execution_evidence: dict

    # ── Labeling pipeline outputs ──
    performance_label: dict
    risk_label: dict
    semantic_label: dict

    # ── Recommendation outputs ──
    recommended_sql: str
    recommendation: dict

    # ── Iteration control ──
    iteration: int
    iteration_history: list
    best_result: dict

    # ── Debug / error tracking ──
    step_logs: list
    error: Optional[str]


# ──────────────────────────────────────────────
#  Helper: append a step log
# ──────────────────────────────────────────────

def _log(state: dict, step: str, data: Any) -> None:
    """Append a debug log entry to the state."""
    logs = state.get("step_logs", [])
    logs.append({
        "step": step,
        "iteration": state.get("iteration", 0),
        "timestamp": time.time(),
        "data": data,
    })
    state["step_logs"] = logs


# ──────────────────────────────────────────────
#  Node 1: Parse & Extract
# ──────────────────────────────────────────────

def node_parse_extract(state: dict) -> dict:
    """
    Parse the SQL query using sqlglot AST traversal.
    Extract tables, columns, joins, WHERE clauses.
    Routes to DDL parse or inference based on ddl_context.
    """
    from parsing.parser import parse_sql, get_join_keys, get_where_details
    from parsing.infer_context import infer_context

    ddl = state.get("ddl_context", "")
    sql = state["original_sql"]

    if ddl and ddl.strip():
        # DDL provided → 100% accurate schema extraction
        context = parse_sql(ddl)
        join_keys = get_join_keys(sql)
        where_details = get_where_details(sql)
        method = "ddl_parse"
        _log(state, "parse_extract", {
            "path": "DDL provided → parse_sql()",
            "tables_found": list(context.keys()),
            "join_keys_count": len(join_keys),
            "where_count": len(where_details),
        })
    else:
        # No DDL → infer from query (~85-90% accurate)
        inferred = infer_context(
            sql=sql,
            provider=state.get("provider", "qwen"),
            model=state.get("model"),
            api_key=state.get("api_key"),
        )
        context = inferred.get("context", {})
        join_keys = inferred.get("join_keys", [])
        where_details = inferred.get("where_details", [])
        method = inferred.get("inference", {}).get("method", "inferred")
        _log(state, "parse_extract", {
            "path": "No DDL → infer_context_from_query()",
            "method": method,
            "tables_found": list(context.keys()),
            "join_keys_count": len(join_keys),
            "where_count": len(where_details),
        })

    return {
        "sql_column_details": context,
        "join_keys": join_keys,
        "where_details": where_details,
        "inference_method": method,
        "step_logs": state.get("step_logs", []),
    }


# ──────────────────────────────────────────────
#  Node 2: ER Graph Builder
# ──────────────────────────────────────────────

def node_build_er_graph(state: dict) -> dict:
    """
    Build the entity-relationship graph:
      - Extract table names from context
      - If join keys present → Python extraction (high confidence)
      - If no join keys → LLM inference (LLM call 1)
      - Build importance hierarchy (root/intermediate/leaf)
      - Detect cross-table risk
    """
    context = state.get("sql_column_details", {})
    join_keys = state.get("join_keys", [])
    where_details = state.get("where_details", [])

    # Extract table names
    tables = list(context.keys())

    # Build join key relationships
    if join_keys or len(context) <= 1:
        # Python node — direct extraction from ON clauses
        relationships = []
        for jk in join_keys:
            relationships.append({
                "source": jk["left_table"],
                "target": jk["right_table"],
                "join_column": jk["left_column"],
                "confidence": "high",
                "origin": "join_key",
            })
        build_method = "python_extraction"
    else:
        # LLM inference — infer from naming conventions (LLM call 1)
        from utils.llm import llm_universal_call_utility
        try:
            prompt = f"""You are a database schema analyst. Analyze the table schemas below and infer entity relationships based on shared column names and naming conventions.
Schema: {json.dumps(context, indent=2)}

Respond ONLY with JSON, no explanation:
{{"table_relationships": [{{"source": "parent_table", "target": "child_table", "join_column": "shared_column", "confidence": "high|medium|low", "origin": "inferred"}}]}}
If no relationships found: {{"table_relationships":[]}}"""

            response = llm_universal_call_utility(
                prompt=prompt,
                provider=state.get("provider", "qwen"),
                model=state.get("model"),
                api_key=state.get("api_key"),
                num_predict=300,
            )
            clean = response.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)
            relationships = data.get("table_relationships", [])
            build_method = "llm_inference"
        except Exception as e:
            relationships = []
            build_method = f"llm_inference_failed: {e}"

    # Build importance hierarchy
    target_set = {r["target"] for r in relationships if r.get("target") and r["target"] != "UNKNOWN"}
    source_set = {r["source"] for r in relationships if r.get("source") and r["source"] != "UNKNOWN"}

    table_importance = []
    for t in tables:
        if t not in target_set:
            importance = "root"
        elif t not in source_set:
            importance = "leaf"
        else:
            importance = "intermediate"
        table_importance.append({"table": t, "importance": importance})

    # Cross-table risk detection
    joined_tables = target_set | source_set
    join_where = [w for w in where_details if w.get("table") in joined_tables]

    er_graph = {
        "join_relationships": relationships,
        "table_importance": table_importance,
        "where_dependencies": where_details,
        "join_where_tables": join_where,
        "cross_table_risk": len(join_where) > 0,
        "graph_depth": len(joined_tables),
        "total_tables": len(tables),
    }

    _log(state, "build_er_graph", {
        "method": build_method,
        "tables": tables,
        "relationships_count": len(relationships),
        "table_importance": table_importance,
        "cross_table_risk": er_graph["cross_table_risk"],
    })

    return {
        "table_names": tables,
        "join_keys_relationships": relationships,
        "er_graph": er_graph,
        "step_logs": state.get("step_logs", []),
    }


# ──────────────────────────────────────────────
#  Node 3: Execution Harness
# ──────────────────────────────────────────────

def node_execution_harness(state: dict) -> dict:
    """
    Build synthetic dataset → in-memory SQLite.
    Run original vs modified (or recommended) SQL.
    Time comparison across configurable scales.
    """
    from execution.synthetic_db import run_query_pair
    from execution.equivalence import check_equivalence
    from execution.performance import compare_performance

    original_sql = state["original_sql"]
    recommended_sql = state.get("recommended_sql", "")

    # Build the record for the execution harness
    record = {
        "original_sql": original_sql,
        "query": original_sql,
        "context": state.get("sql_column_details", {}),
        "join_keys": state.get("join_keys", []),
        "where_details": state.get("where_details", []),
    }

    # If we have a recommended SQL (from iteration > 0), test it
    if recommended_sql and recommended_sql.strip() and recommended_sql != original_sql:
        record["modified_sql"] = recommended_sql
    else:
        record["modified_sql"] = original_sql

    # Equivalence check
    try:
        equiv = check_equivalence(record, seed=42, rows_per_table=50)
    except Exception as e:
        equiv = {
            "equivalent": False,
            "output_relation": "error",
            "row_count_original": 0,
            "row_count_modified": 0,
            "comparison": {"error": str(e)},
        }

    # Performance comparison across scales
    try:
        perf = compare_performance(
            record,
            scales={"small": 50, "large": 2000},
            repeats=5,
            seed=42,
        )
    except Exception as e:
        perf = {"error": str(e)}

    evidence = {
        "equivalence": equiv,
        "performance": perf,
        "output_relation": equiv.get("output_relation", "unknown"),
        "row_count_original": equiv.get("row_count_original", 0),
        "row_count_modified": equiv.get("row_count_modified", 0),
        "row_count_delta": equiv.get("row_count_modified", 0) - equiv.get("row_count_original", 0),
        "both_succeeded": equiv.get("comparison", {}).get("both_succeeded", False),
        "original_error": equiv.get("comparison", {}).get("original_error"),
        "modified_error": equiv.get("comparison", {}).get("modified_error"),
        "runtime_delta_ms": equiv.get("comparison", {}).get("runtime_delta_ms"),
        "runtime_ratio": equiv.get("comparison", {}).get("runtime_ratio"),
    }

    _log(state, "execution_harness", {
        "output_relation": evidence["output_relation"],
        "row_counts": f"{evidence['row_count_original']} → {evidence['row_count_modified']}",
        "both_succeeded": evidence["both_succeeded"],
        "performance_scales": list(perf.keys()) if isinstance(perf, dict) else "error",
    })

    return {
        "execution_evidence": evidence,
        "step_logs": state.get("step_logs", []),
    }


# ──────────────────────────────────────────────
#  Node 4: Labeling Pipeline
# ──────────────────────────────────────────────

def node_labeling_pipeline(state: dict) -> dict:
    """
    Labeling pipeline — deterministic rules first, LLM call 2 to refine.
    Produces three labels:
      - Performance: improves / degrades / neutral / unknown
      - Risk: low / medium / high
      - Semantic: equivalent / narrower / broader / different
    """
    from reasoning.performance_labeler import classify_performance
    from reasoning.risk_labeler import classify_risk
    from reasoning.semantic_labeler import classify_semantic

    original_sql = state["original_sql"]
    recommended_sql = state.get("recommended_sql", original_sql)
    er_graph = state.get("er_graph", {})
    execution_evidence = state.get("execution_evidence", {})
    provider = state.get("provider", "qwen")
    model = state.get("model")
    api_key = state.get("api_key")

    # Performance label
    perf_label = classify_performance(
        original_sql=original_sql,
        execution_evidence=execution_evidence,
        provider=provider,
        model=model,
        api_key=api_key,
    )

    # Risk label
    risk_label_result = classify_risk(
        original_sql=original_sql,
        er_graph=er_graph,
        join_keys=state.get("join_keys", []),
        where_details=state.get("where_details", []),
        execution_evidence=execution_evidence,
        provider=provider,
        model=model,
        api_key=api_key,
    )

    # Semantic label
    sem_label = classify_semantic(
        original_sql=original_sql,
        modified_sql=recommended_sql,
        execution_evidence=execution_evidence,
        provider=provider,
        model=model,
        api_key=api_key,
    )

    _log(state, "labeling_pipeline", {
        "performance": {"score": perf_label.get("score"), "label": perf_label.get("label")},
        "risk": {"score": risk_label_result.get("score"), "label": risk_label_result.get("label")},
        "semantic": {"label": sem_label.get("label"), "confidence": sem_label.get("confidence")},
    })

    return {
        "performance_label": perf_label,
        "risk_label": risk_label_result,
        "semantic_label": sem_label,
        "step_logs": state.get("step_logs", []),
    }


# ──────────────────────────────────────────────
#  Node 5: Recommend
# ──────────────────────────────────────────────

def node_recommend(state: dict) -> dict:
    """
    LLM call 3 — suggest an optimized query based on all upstream signals.
    """
    from recommendation.recommend import recommend_query

    original_sql = state["original_sql"]
    recommendation = recommend_query(
        original_sql=original_sql,
        sql_column_details=state.get("sql_column_details", {}),
        er_graph=state.get("er_graph", {}),
        join_keys=state.get("join_keys", []),
        where_details=state.get("where_details", []),
        execution_evidence=state.get("execution_evidence"),
        performance_label=state.get("performance_label"),
        risk_label=state.get("risk_label"),
        semantic_label=state.get("semantic_label"),
        provider=state.get("provider", "qwen"),
        model=state.get("model"),
        api_key=state.get("api_key"),
    )

    rec_sql = recommendation.get("recommended_sql", original_sql)
    if not rec_sql or not rec_sql.strip():
        rec_sql = original_sql

    # Track iteration history
    history = state.get("iteration_history", [])
    iteration = state.get("iteration", 0)
    history.append({
        "iteration": iteration,
        "recommended_sql": rec_sql,
        "action": recommendation.get("action"),
        "score": recommendation.get("score"),
        "is_valid": recommendation.get("is_valid"),
        "performance_score": state.get("performance_label", {}).get("score"),
        "risk_score": state.get("risk_label", {}).get("score"),
        "semantic_label": state.get("semantic_label", {}).get("label"),
    })

    # Track best result (highest recommendation score with valid SQL)
    best = state.get("best_result", {})
    if recommendation.get("is_valid", False) and recommendation.get("score", 0) > best.get("score", 0):
        best = {
            "score": recommendation["score"],
            "recommended_sql": rec_sql,
            "iteration": iteration,
            "action": recommendation.get("action"),
            "performance_label": state.get("performance_label", {}).get("label"),
            "risk_label": state.get("risk_label", {}).get("label"),
        }

    _log(state, "recommend", {
        "action": recommendation.get("action"),
        "score": recommendation.get("score"),
        "is_valid": recommendation.get("is_valid"),
        "recommended_sql_preview": rec_sql[:120] + "..." if len(rec_sql) > 120 else rec_sql,
    })

    return {
        "recommended_sql": rec_sql,
        "recommendation": recommendation,
        "iteration": iteration + 1,
        "iteration_history": history,
        "best_result": best,
        "step_logs": state.get("step_logs", []),
    }


# ──────────────────────────────────────────────
#  Conditional: Should we iterate?
# ──────────────────────────────────────────────

def should_iterate(state: dict) -> str:
    """
    Decide whether to loop back for another iteration or finish.

    Continue if:
      - iteration < max_iterations
      - AND (performance can improve OR risk can decrease)
      - AND the recommendation is valid
    """
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)

    if iteration >= max_iter:
        _log(state, "iteration_check", {"decision": "STOP", "reason": f"Reached max iterations ({max_iter})"})
        return "done"

    recommendation = state.get("recommendation", {})
    perf_label = state.get("performance_label", {})
    risk_label = state.get("risk_label", {})

    # Stop if recommendation says keep_original
    if recommendation.get("action") == "keep_original":
        _log(state, "iteration_check", {"decision": "STOP", "reason": "Action is keep_original"})
        return "done"

    # Stop if not valid SQL
    if not recommendation.get("is_valid", False):
        _log(state, "iteration_check", {"decision": "STOP", "reason": "Recommended SQL is invalid"})
        return "done"

    # Stop if performance is already great and risk is low
    perf_score = perf_label.get("score", 5)
    risk_score = risk_label.get("score", 5)
    if perf_score >= 8 and risk_score <= 3:
        _log(state, "iteration_check", {"decision": "STOP", "reason": f"Already optimal (perf={perf_score}, risk={risk_score})"})
        return "done"

    # Continue iterating — there's room for improvement
    _log(state, "iteration_check", {
        "decision": "CONTINUE",
        "reason": f"Room to improve (perf={perf_score}, risk={risk_score}, iter={iteration}/{max_iter})",
    })
    return "re_test"


# ──────────────────────────────────────────────
#  Node 6: Final Output
# ──────────────────────────────────────────────

def node_final_output(state: dict) -> dict:
    """
    Assemble the final output: labels + recommendation + best result.
    """
    best = state.get("best_result", {})
    recommendation = state.get("recommendation", {})

    # Use best result from any iteration if available
    final_sql = best.get("recommended_sql") or state.get("recommended_sql") or state["original_sql"]
    final_action = best.get("action") or recommendation.get("action", "keep_original")

    improved = (
        final_sql != state["original_sql"]
        and recommendation.get("is_valid", False)
        and final_action != "keep_original"
    )

    _log(state, "final_output", {
        "improved": improved,
        "total_iterations": state.get("iteration", 0),
        "best_iteration": best.get("iteration", 0),
        "final_action": final_action,
    })

    return {
        "step_logs": state.get("step_logs", []),
    }


# ──────────────────────────────────────────────
#  Build the LangGraph
# ──────────────────────────────────────────────

_cached_pipeline = None


def build_pipeline():
    """Construct and compile the full pipeline LangGraph."""
    global _cached_pipeline
    if _cached_pipeline is not None:
        return _cached_pipeline

    graph = StateGraph(PipelineState)

    # Add all nodes
    graph.add_node("parse_extract", node_parse_extract)
    graph.add_node("build_er_graph", node_build_er_graph)
    graph.add_node("execution_harness", node_execution_harness)
    graph.add_node("labeling_pipeline", node_labeling_pipeline)
    graph.add_node("recommend", node_recommend)
    graph.add_node("final_output", node_final_output)

    # Linear flow: parse → ER graph → execution → labeling → recommend
    graph.set_entry_point("parse_extract")
    graph.add_edge("parse_extract", "build_er_graph")
    graph.add_edge("build_er_graph", "execution_harness")
    graph.add_edge("execution_harness", "labeling_pipeline")
    graph.add_edge("labeling_pipeline", "recommend")

    # Conditional: iterate or finish
    graph.add_conditional_edges(
        "recommend",
        should_iterate,
        {
            "re_test": "execution_harness",   # Loop back: re-test with new recommended SQL
            "done": "final_output",
        },
    )

    graph.add_edge("final_output", END)

    _cached_pipeline = graph.compile()
    return _cached_pipeline


def run_pipeline(
    original_sql: str,
    ddl_context: str = None,
    provider: str = "qwen",
    model: str = None,
    api_key: str = None,
    max_iterations: int = 3,
) -> dict:
    """
    Main entry point: run the full SQL optimization pipeline.

    Args:
        original_sql: The SQL query to optimize
        ddl_context: Optional DDL statements for schema
        provider: LLM provider ("qwen", "local", "ollama", "anthropic", "openai")
        model: Model name (defaults to "qwen2.5-coder:7b" for Ollama)
        api_key: API key (for Anthropic/OpenAI)
        max_iterations: Maximum optimization iterations (default 3)

    Returns:
        Full pipeline state with all labels, recommendations, and debug logs.
    """
    pipeline = build_pipeline()

    initial_state = {
        "original_sql": original_sql,
        "ddl_context": ddl_context or "",
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "max_iterations": max_iterations,
        "sql_column_details": {},
        "join_keys": [],
        "where_details": [],
        "inference_method": None,
        "table_names": [],
        "join_keys_relationships": [],
        "er_graph": {},
        "execution_evidence": {},
        "performance_label": {},
        "risk_label": {},
        "semantic_label": {},
        "recommended_sql": "",
        "recommendation": {},
        "iteration": 0,
        "iteration_history": [],
        "best_result": {},
        "step_logs": [],
        "error": None,
    }

    result = pipeline.invoke(initial_state)
    return result
