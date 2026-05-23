"""
SQLChange agentic recommendation engine.

LangGraph state machine where the LLM is the reasoning brain at every
decision point. Python nodes handle mechanical work (parsing, execution,
timing). The LLM analyzes, evaluates, and decides whether to iterate.

LLM nodes:
  1. analyze_query     — understand the query, propose which mutations to try
  2. evaluate_results  — reason about execution evidence for each candidate
  3. decide_next       — iterate with new ideas, or finalize recommendation

Python nodes:
  - generate_and_test  — mutation_engine + synthetic_db + equivalence + performance

Uses every contributor's module:
  - mutation_engine.py  (Dev)    — generates candidate mutations
  - parser.py           (Dev)    — structural parsing
  - graph_representer.py (Fahim) — ER graph + LLM utility
  - synthetic_db.py     (Fahim)  — execution evidence
  - equivalence.py      (Fahim)  — correctness checking
  - performance.py      (Fahim)  — multi-scale timing
  - reasoning_pipeline.py (Fahim) — rule-based pre-classification
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END

from mutation_engine import match_sql_to_mutation, mutation_function_mapping
from parser import parse_sql, get_join_keys, get_where_details, validate_sql_columns
from graph_representer import build_graph, llm_universal_call_utility
from equivalence import check_equivalence
from performance import compare_performance
from reasoning_pipeline import _base_reasoning, _rule_signals

MAX_ITERATIONS = 3

# -- State ------------------------------------------------------------------

class RecommendState(TypedDict):
    sql: str
    schema_ddl: str
    context: dict
    join_keys: list
    where_details: list
    er_graph: dict
    provider: str
    model: str
    api_key: Optional[str]
    iteration: int
    mutations_to_try: list
    candidates: list
    llm_analysis: str
    llm_evaluation: str
    recommendation: dict
    done: bool


# -- Helpers ----------------------------------------------------------------

def _llm(prompt, state):
    raw = llm_universal_call_utility(
        prompt=prompt, provider=state["provider"],
        api_key=state["api_key"], model=state["model"])
    return raw.strip().replace("```json", "").replace("```", "").strip()


def _parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {}


# -- LLM Node 1: Analyze Query ---------------------------------------------

def llm_analyze_query(state: RecommendState) -> RecommendState:
    """LLM examines the query and proposes which mutations are worth exploring."""
    applicable = match_sql_to_mutation(state["sql"])
    prev_candidates = state["candidates"]

    prev_summary = ""
    if prev_candidates:
        prev_summary = (
            "\n\nPrevious iteration results (avoid repeating these):\n"
            + json.dumps([{
                "mutation_type": c["mutation_type"],
                "equivalent": c["equivalence"].get("output_relation"),
                "speedup": c.get("performance", {}).get("large", {}).get("speedup"),
            } for c in prev_candidates], indent=2)
        )

    prompt = (
        "You are a SQL optimization expert. Analyze this query and decide which "
        "mutations are worth trying to optimize it.\n\n"
        f"Query: {state['sql']}\n"
        f"Schema: {json.dumps(state['context'], default=str)}\n"
        f"ER Graph: {json.dumps(state['er_graph'], default=str)}\n"
        f"Available mutation types: {applicable}\n"
        f"Iteration: {state['iteration'] + 1} of {MAX_ITERATIONS}"
        f"{prev_summary}\n\n"
        "For each mutation type, explain WHY it might help or hurt this specific query. "
        "Then select which ones to test.\n\n"
        "Return JSON:\n"
        '{"analysis": "your reasoning about this query", '
        '"mutations_to_try": ["mutation_type1", ...], '
        '"reasoning_per_mutation": {"mutation_type": "why this might help"}}'
    )
    response = _llm(prompt, state)
    parsed = _parse_json(response)

    mutations = parsed.get("mutations_to_try", applicable)
    valid = [m for m in mutations if m in applicable]

    state["mutations_to_try"] = valid if valid else applicable
    state["llm_analysis"] = parsed.get("analysis", response)
    state["iteration"] = state["iteration"] + 1
    return state


# -- Python Node: Generate & Test -------------------------------------------

def python_generate_and_test(state: RecommendState) -> RecommendState:
    """Generate mutations the LLM chose and test each on synthetic data."""
    sql = state["sql"]
    context = state["context"]
    new_candidates = []

    already_tested = {c["mutation_type"] for c in state["candidates"]}

    for mutation_type in state["mutations_to_try"]:
        if mutation_type in already_tested:
            continue

        mutated = mutation_function_mapping.get(mutation_type, lambda x: None)(sql)
        if not mutated or mutated.strip() == sql.strip():
            continue
        if validate_sql_columns(mutated, context, sql) is False:
            continue

        record = {
            "context": context,
            "original_sql": sql,
            "modified_sql": mutated,
            "mutation_type": mutation_type,
            "join_keys": state["join_keys"],
            "where_details": state["where_details"],
            "er_graph": state["er_graph"],
        }

        try:
            equiv = check_equivalence(record, seed=42, rows_per_table=100)
        except Exception as e:
            equiv = {"equivalent": False, "output_relation": "error", "error": str(e)}

        try:
            perf = compare_performance(record, scales={"small": 50, "large": 1000},
                                       repeats=5, seed=42)
        except Exception as e:
            perf = {"error": str(e)}

        rules = _base_reasoning(record)

        new_candidates.append({
            "mutation_type": mutation_type,
            "modified_sql": mutated,
            "equivalence": equiv,
            "performance": perf,
            "rules": rules,
        })

    state["candidates"] = state["candidates"] + new_candidates
    return state


# -- LLM Node 2: Evaluate Results ------------------------------------------

def llm_evaluate_results(state: RecommendState) -> RecommendState:
    """LLM reviews all tested candidates and reasons about each one."""
    summary = []
    for i, c in enumerate(state["candidates"]):
        entry = {
            "index": i,
            "mutation_type": c["mutation_type"],
            "modified_sql": c["modified_sql"],
            "output_relation": c["equivalence"].get("output_relation"),
            "row_count_original": c["equivalence"].get("row_count_original"),
            "row_count_modified": c["equivalence"].get("row_count_modified"),
            "rule_semantic": c["rules"]["semantic"]["label"],
            "rule_risk": c["rules"]["risk"]["label"],
        }
        if "error" not in c.get("performance", {}):
            large = c["performance"].get("large", {})
            entry["speedup"] = large.get("speedup")
            entry["original_ms"] = large.get("original_ms")
            entry["modified_ms"] = large.get("modified_ms")
        summary.append(entry)

    prompt = (
        "You are evaluating SQL optimization candidates. For each candidate, "
        "reason about:\n"
        "1. Does it preserve correctness? (output_relation: identical = safe)\n"
        "2. Does it actually improve performance? (speedup > 1.0 = faster)\n"
        "3. What is the real-world risk?\n\n"
        f"Original query: {state['sql']}\n"
        f"Your earlier analysis: {state['llm_analysis']}\n\n"
        f"Candidates with evidence:\n{json.dumps(summary, indent=2)}\n\n"
        f"Iteration {state['iteration']} of {MAX_ITERATIONS}. "
        "Should we explore further or is a candidate good enough?\n\n"
        "Return JSON:\n"
        '{"evaluation": "your detailed reasoning about each candidate", '
        '"best_candidate_index": <int or null>, '
        '"should_iterate": true/false, '
        '"iterate_reason": "what would you try differently next round"}'
    )
    response = _llm(prompt, state)
    parsed = _parse_json(response)

    state["llm_evaluation"] = parsed.get("evaluation", response)

    if not parsed.get("should_iterate") or state["iteration"] >= MAX_ITERATIONS:
        state["done"] = True
        best_idx = parsed.get("best_candidate_index")
        if best_idx is not None and 0 <= best_idx < len(state["candidates"]):
            state["recommendation"] = {"best_index": best_idx}
        else:
            state["recommendation"] = {"best_index": None}
    else:
        state["done"] = False

    return state


# -- LLM Node 3: Final Recommendation --------------------------------------

def llm_recommend(state: RecommendState) -> RecommendState:
    """LLM produces the final structured recommendation."""
    best = state["recommendation"].get("best_index")
    candidates = state["candidates"]

    if best is not None and 0 <= best < len(candidates):
        chosen = candidates[best]
        chosen_sql = chosen["modified_sql"]
    else:
        chosen = None
        chosen_sql = state["sql"]

    prompt = (
        "You are a SQL optimization advisor giving your final recommendation.\n\n"
        f"Original SQL: {state['sql']}\n"
        f"Recommended SQL: {chosen_sql}\n"
        f"Your analysis: {state['llm_analysis']}\n"
        f"Your evaluation: {state['llm_evaluation']}\n"
        f"Iterations used: {state['iteration']}\n\n"
        "Provide your final assessment.\n\n"
        "Return ONLY JSON:\n"
        "{\n"
        '  "recommended_sql": "the SQL to use",\n'
        '  "semantic": {"label": "equivalent|narrower|broader|different", '
        '"confidence": 0.0, "rationale": "..."},\n'
        '  "performance": {"label": "improves|degrades|neutral|unknown", '
        '"confidence": 0.0, "rationale": "..."},\n'
        '  "risk": {"label": "low|medium|high", "confidence": 0.0, "rationale": "..."},\n'
        '  "summary": "one-paragraph justification of the recommendation"\n'
        "}"
    )
    response = _llm(prompt, state)
    state["recommendation"] = _parse_json(response)
    if "recommended_sql" not in state["recommendation"]:
        state["recommendation"]["recommended_sql"] = chosen_sql
    return state


# -- Router -----------------------------------------------------------------

def should_iterate(state: RecommendState) -> str:
    if state["done"]:
        return "llm_recommend"
    return "llm_analyze_query"


# -- Build Graph ------------------------------------------------------------

def _build_pipeline():
    graph = StateGraph(RecommendState)

    graph.add_node("llm_analyze_query", llm_analyze_query)
    graph.add_node("python_generate_and_test", python_generate_and_test)
    graph.add_node("llm_evaluate_results", llm_evaluate_results)
    graph.add_node("llm_recommend", llm_recommend)

    graph.set_entry_point("llm_analyze_query")
    graph.add_edge("llm_analyze_query", "python_generate_and_test")
    graph.add_edge("python_generate_and_test", "llm_evaluate_results")
    graph.add_conditional_edges("llm_evaluate_results", should_iterate, {
        "llm_analyze_query": "llm_analyze_query",
        "llm_recommend": "llm_recommend",
    })
    graph.add_edge("llm_recommend", END)

    return graph.compile()

_pipeline = None


# -- Public API -------------------------------------------------------------

def recommend(sql, schema_ddl, provider="anthropic",
              model="claude-sonnet-4-20250514", api_key=None):
    """
    Agentic SQL optimization: LLM analyzes, proposes mutations, reviews
    execution evidence, and iterates until it finds the best optimization.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = _build_pipeline()

    context = parse_sql(schema_ddl)
    join_keys = get_join_keys(sql)
    where_details = get_where_details(sql)

    er_graph = {}
    try:
        out = build_graph(context, join_keys, where_details, model, provider, api_key)
        er_graph = out.get("data_graph", {})
    except Exception:
        pass

    initial_state = {
        "sql": sql,
        "schema_ddl": schema_ddl,
        "context": context,
        "join_keys": join_keys,
        "where_details": where_details,
        "er_graph": er_graph,
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "iteration": 0,
        "mutations_to_try": [],
        "candidates": [],
        "llm_analysis": "",
        "llm_evaluation": "",
        "recommendation": {},
        "done": False,
    }

    result = _pipeline.invoke(initial_state)

    return {
        "original_sql": sql,
        "er_graph": er_graph,
        "iterations": result["iteration"],
        "llm_analysis": result["llm_analysis"],
        "llm_evaluation": result["llm_evaluation"],
        "candidates": result["candidates"],
        "recommendation": result["recommendation"],
    }
