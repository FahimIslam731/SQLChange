"""
Reasoning pipeline for assigning SQL mutation labels.

The data extraction pipeline produces enriched mutation records with null label
fields. This module fills those labels using deterministic rules first, with an
optional LLM pass that may refine rationales and confidence while staying within
the allowed label vocabulary.
"""

import copy
import json
import re
from typing import Any, Dict, List, Optional

try:
    from graph_representer import llm_universal_call_utility
except ImportError:
    llm_universal_call_utility = None


SEMANTIC_LABELS = {"equivalent", "narrower", "broader", "different"}
PERFORMANCE_LABELS = {"improves", "degrades", "neutral", "unknown"}
RISK_LABELS = {"low", "medium", "high"}


def _safe_er_graph(record: Dict[str, Any]) -> Dict[str, Any]:
    er_graph = record.get("er_graph")
    return er_graph if isinstance(er_graph, dict) else {}


def _is_cross_table_risk(record: Dict[str, Any]) -> bool:
    er_graph = _safe_er_graph(record)
    return bool(er_graph.get("cross_table_risk") or er_graph.get("join_where_tables"))


def _has_join_context(record: Dict[str, Any]) -> bool:
    er_graph = _safe_er_graph(record)
    return bool(record.get("join_keys") or er_graph.get("join_relationships"))


def _complexity(record: Dict[str, Any]) -> str:
    return str(record.get("complexity") or "").lower()


def _modified_sql(record: Dict[str, Any]) -> str:
    return str(record.get("modified_sql") or "").upper()


def _confidence(value: str) -> float:
    confidence_map = {
        "very_high": 0.95,
        "high": 0.88,
        "medium": 0.72,
        "low": 0.55,
    }
    return confidence_map[value]


def _rule_signals(record: Dict[str, Any]) -> Dict[str, Any]:
    er_graph = _safe_er_graph(record)
    return {
        "mutation_type": record.get("mutation_type"),
        "complexity": record.get("complexity"),
        "cross_table_risk": bool(er_graph.get("cross_table_risk")),
        "graph_depth": er_graph.get("graph_depth", 0),
        "total_tables": er_graph.get("total_tables", 0),
        "where_dependency_count": len(record.get("where_details") or []),
        "join_relationship_count": len(er_graph.get("join_relationships") or []),
        "join_where_table_count": len(er_graph.get("join_where_tables") or []),
    }


def _performance_label_from_evidence(execution_evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derive a performance label from timing evidence produced by compare_performance().

    Uses the best available scale in priority order: large → medium → small.
    speedup = original_ms / modified_ms  (> 1 means modified is faster).

    Thresholds:
        speedup >= 1.15  →  improves
        speedup <= 0.85  →  degrades
        otherwise        →  neutral
        missing / error  →  unknown

    Confidence is capped at "low" when both query times are under 0.05 ms
    (SQLite in-memory noise floor dominates at that scale).
    """
    IMPROVE_THRESHOLD = 1.15
    DEGRADE_THRESHOLD = 0.85
    NOISE_FLOOR_MS = 0.05

    perf = (execution_evidence or {}).get("performance") or {}

    for scale in ("large", "medium", "small"):
        scale_data = perf.get(scale)
        if not isinstance(scale_data, dict):
            continue
        speedup = scale_data.get("speedup")
        original_ms = scale_data.get("original_ms")
        modified_ms = scale_data.get("modified_ms")

        if speedup is None or not isinstance(speedup, (int, float)):
            continue
        if not isinstance(original_ms, (int, float)) or not isinstance(modified_ms, (int, float)):
            continue

        # Assign base confidence by scale
        if scale == "large":
            confidence = "high"
        elif scale == "medium":
            confidence = "medium"
        else:
            confidence = "low"

        # Noise-floor guard: both queries too fast to measure reliably
        if original_ms < NOISE_FLOOR_MS and modified_ms < NOISE_FLOOR_MS:
            confidence = "low"

        if speedup >= IMPROVE_THRESHOLD:
            label = "improves"
            reason = (
                f"Modified query is {speedup:.2f}x faster than original at {scale} scale "
                f"({original_ms:.3f}ms → {modified_ms:.3f}ms)."
            )
        elif speedup <= DEGRADE_THRESHOLD:
            label = "degrades"
            reason = (
                f"Modified query is {speedup:.2f}x the speed of original at {scale} scale "
                f"({original_ms:.3f}ms → {modified_ms:.3f}ms); performance worsened."
            )
        else:
            label = "neutral"
            reason = (
                f"Speedup ratio {speedup:.2f} at {scale} scale is within the neutral band "
                f"({original_ms:.3f}ms → {modified_ms:.3f}ms)."
            )

        return {
            "label": label,
            "confidence": confidence,
            "reason": reason,
            "signals": {
                "scale_used": scale,
                "speedup_used": round(speedup, 4),
                "original_ms": round(original_ms, 4),
                "modified_ms": round(modified_ms, 4),
            },
        }

    # No usable timing data found
    return {
        "label": "unknown",
        "confidence": "low",
        "reason": "No valid timing evidence available; falling back to rule-based label.",
        "signals": {
            "scale_used": None,
            "speedup_used": None,
            "original_ms": None,
            "modified_ms": None,
        },
    }


def _risk_rank(label: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(label, 1)


def _risk_from_rank(rank: int) -> str:
    return {0: "low", 1: "medium", 2: "high"}.get(max(0, min(2, rank)), "medium")


def _escalate_risk(base_risk: str, steps: int = 1) -> str:
    return _risk_from_rank(_risk_rank(base_risk) + steps)


def _risk_label_from_evidence(
    record: Dict[str, Any],
    base_risk: str,
    execution_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Derive an evidence-adjusted risk label, escalating the static base_risk when
    execution or ER-graph signals indicate elevated danger.

    Escalation rules (each fires independently; total escalation is capped so the
    result never exceeds "high"):

    Rule A — output_relation is "different":
        Escalate +1.  Query change produces structurally different rows; cross-table
        side effects are more likely than the static rule assumes.

    Rule B — row_count_delta is highly positive (modified returns >= 2x original rows):
        Escalate +1.  Large row-count growth signals the modification broadens results
        far more than expected; downstream consumers are at risk.

    Rule C — execution error on at least one query (original or modified):
        Escalate +1.  A query that fails at medium scale implies brittle SQL that may
        break in production with real data.

    Rule D — graph_depth >= 3:
        Escalate +1.  Deep ER graphs indicate complex multi-hop relationships that
        magnify the blast radius of any mutation.

    Rule E — join_where_table_count >= 2:
        Escalate +1.  Multiple joined tables involved in WHERE conditions create
        implicit cross-table dependencies even when cross_table_risk is False.

    Confidence:
        "high"   if comparison data is present and both queries succeeded
        "medium" if comparison data is present but a query errored, or no comparison
                 but ER graph signals were available
        "low"    otherwise (no comparison, no meaningful signals)
    """
    er_graph = _safe_er_graph(record)
    graph_depth = er_graph.get("graph_depth", 0) or 0
    join_where_table_count = len(er_graph.get("join_where_tables") or [])

    comparison = (execution_evidence or {}).get("comparison") or {}
    output_relation = comparison.get("output_relation")
    row_count_original = comparison.get("row_count_original", 0) or 0
    row_count_modified = comparison.get("row_count_modified", 0) or 0
    original_error = comparison.get("original_error")
    modified_error = comparison.get("modified_error")
    both_succeeded = comparison.get("both_succeeded", False)
    has_comparison = bool(output_relation)

    escalation = 0
    triggers = []

    # Rule A
    if output_relation == "different":
        escalation += 1
        triggers.append("output_relation=different")

    # Rule B
    if (
        has_comparison
        and both_succeeded
        and row_count_original > 0
        and row_count_modified >= row_count_original * 2
    ):
        escalation += 1
        triggers.append(f"row_count_delta={row_count_modified - row_count_original} (2x growth)")

    # Rule C
    if has_comparison and (original_error or modified_error):
        escalation += 1
        triggers.append("execution_error")

    # Rule D
    if graph_depth >= 3:
        escalation += 1
        triggers.append(f"graph_depth={graph_depth}")

    # Rule E
    if join_where_table_count >= 2:
        escalation += 1
        triggers.append(f"join_where_table_count={join_where_table_count}")

    final_label = _escalate_risk(base_risk, escalation)

    # Confidence
    if has_comparison and both_succeeded:
        confidence = "high"
    elif has_comparison or (graph_depth > 0 or join_where_table_count > 0):
        confidence = "medium"
    else:
        confidence = "low"

    if escalation > 0:
        reason = (
            f"Base risk '{base_risk}' escalated to '{final_label}' "
            f"(+{escalation} step(s)) due to: {', '.join(triggers)}."
        )
    else:
        reason = (
            f"Base risk '{base_risk}' confirmed; no escalation triggers fired."
        )

    return {
        "label": final_label,
        "confidence": confidence,
        "reason": reason,
        "signals": {
            "base_risk": base_risk,
            "escalation_steps": escalation,
            "triggers": triggers,
            "output_relation": output_relation,
            "row_count_original": row_count_original,
            "row_count_modified": row_count_modified,
            "graph_depth": graph_depth,
            "join_where_table_count": join_where_table_count,
        },
    }


def _base_reasoning(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    mutation_type = record.get("mutation_type")
    complexity = _complexity(record)
    cross_table_risk = _is_cross_table_risk(record)
    join_context = _has_join_context(record)

    if mutation_type == "where_drop":
        risk_label = "high" if cross_table_risk else "medium"
        risk_reason = (
            "Dropped filter condition touches joined-table logic."
            if cross_table_risk
            else "Dropped filter condition broadens results without joined-table evidence."
        )
        return {
            "semantic": {
                "label": "broader",
                "confidence": _confidence("high"),
                "rationale": "Removing a WHERE condition allows additional rows to qualify.",
            },
            "performance": {
                "label": "improves",
                "confidence": _confidence("medium"),
                "rationale": "Less filtering work may reduce predicate evaluation, though more rows can flow downstream.",
            },
            "risk": {
                "label": risk_label,
                "confidence": _confidence("high"),
                "rationale": risk_reason,
            },
        }

    if mutation_type == "join_swap":
        performance_label = "improves" if " INNER JOIN " in _modified_sql(record) else "unknown"
        performance_reason = (
            "Changing to an INNER JOIN can reduce row preservation from the outer side."
            if performance_label == "improves"
            else "Join type swap changes optimizer choices in a data-dependent way."
        )
        return {
            "semantic": {
                "label": "different",
                "confidence": _confidence("very_high"),
                "rationale": "Changing join type can add or remove rows when matches are missing.",
            },
            "performance": {
                "label": performance_label,
                "confidence": _confidence("medium"),
                "rationale": performance_reason,
            },
            "risk": {
                "label": "high",
                "confidence": _confidence("very_high"),
                "rationale": "Join type controls table matching semantics and can materially alter results.",
            },
        }

    if mutation_type == "join_drop":
        return {
            "semantic": {
                "label": "different",
                "confidence": _confidence("very_high"),
                "rationale": "Removing a JOIN removes table relationships and can change result rows or available columns.",
            },
            "performance": {
                "label": "improves",
                "confidence": _confidence("high"),
                "rationale": "Eliminating a JOIN usually reduces join processing cost.",
            },
            "risk": {
                "label": "high",
                "confidence": _confidence("very_high"),
                "rationale": "Dropping a joined table is a high-impact structural query change.",
            },
        }

    if mutation_type == "group_by_drop":
        risk_label = "high" if "aggregation" in complexity else "medium"
        return {
            "semantic": {
                "label": "different",
                "confidence": _confidence("very_high"),
                "rationale": "Removing GROUP BY changes aggregation granularity and result shape.",
            },
            "performance": {
                "label": "improves",
                "confidence": _confidence("high"),
                "rationale": "Removing grouping usually reduces aggregation and sort/hash work.",
            },
            "risk": {
                "label": risk_label,
                "confidence": _confidence("high"),
                "rationale": "Aggregation-level changes are high risk for aggregation queries.",
            },
        }

    if mutation_type == "limit_add":
        risk_label = "low" if "basic" in complexity and not join_context else "medium"
        return {
            "semantic": {
                "label": "narrower",
                "confidence": _confidence("very_high"),
                "rationale": "Adding LIMIT restricts the number of returned rows.",
            },
            "performance": {
                "label": "improves",
                "confidence": _confidence("high"),
                "rationale": "A LIMIT can reduce result materialization and downstream work.",
            },
            "risk": {
                "label": risk_label,
                "confidence": _confidence("medium"),
                "rationale": "LIMIT affects completeness of results but is lower risk for simple single-table queries.",
            },
        }

    if mutation_type == "column_drop":
        risk_label = "medium" if join_context or any(x in complexity for x in ["aggregation", "window", "join"]) else "low"
        return {
            "semantic": {
                "label": "narrower",
                "confidence": _confidence("high"),
                "rationale": "Dropping a selected column narrows the output projection.",
            },
            "performance": {
                "label": "neutral",
                "confidence": _confidence("medium"),
                "rationale": "Projection changes usually have limited performance impact unless the dropped expression is expensive.",
            },
            "risk": {
                "label": risk_label,
                "confidence": _confidence("medium"),
                "rationale": "Output-only changes are lower risk, but complexity or join context raises integration risk.",
            },
        }

    return {
        "semantic": {
            "label": "different",
            "confidence": _confidence("low"),
            "rationale": "Unknown mutation type; defaulting to changed semantics.",
        },
        "performance": {
            "label": "unknown",
            "confidence": _confidence("low"),
            "rationale": "Unknown mutation type prevents a deterministic performance estimate.",
        },
        "risk": {
            "label": "medium",
            "confidence": _confidence("low"),
            "rationale": "Unknown mutation type carries moderate review risk.",
        },
    }


def _build_llm_prompt(record: Dict[str, Any], rule_result: Dict[str, Any]) -> str:
    compact_record = {
        "unique_id": record.get("unique_id"),
        "domain": record.get("domain"),
        "complexity": record.get("complexity"),
        "mutation_type": record.get("mutation_type"),
        "original_sql": record.get("original_sql"),
        "modified_sql": record.get("modified_sql"),
        "join_keys": record.get("join_keys"),
        "where_details": record.get("where_details"),
        "er_graph": record.get("er_graph"),
        "rule_result": rule_result,
    }
    return f"""
You are reviewing a SQL mutation classification. Start from the deterministic
rule_result and refine only if the record evidence clearly supports it.

Allowed semantic labels: equivalent, narrower, broader, different.
Allowed performance labels: improves, degrades, neutral, unknown.
Allowed risk labels: low, medium, high.

Return only JSON with this shape:
{{
  "semantic": {{"label": "...", "confidence": 0.0, "rationale": "..."}},
  "performance": {{"label": "...", "confidence": 0.0, "rationale": "..."}},
  "risk": {{"label": "...", "confidence": 0.0, "rationale": "..."}}
}}

Record:
{json.dumps(compact_record, indent=2)}
"""


def _extract_json_object(text: str) -> Dict[str, Any]:
    clean = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _sanitize_llm_dimension(
    llm_data: Dict[str, Any],
    base_dimension: Dict[str, Any],
    allowed_labels: set,
) -> Dict[str, Any]:
    label = llm_data.get("label")
    if label not in allowed_labels:
        return base_dimension

    confidence = llm_data.get("confidence", base_dimension["confidence"])
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = base_dimension["confidence"]
    confidence = max(0.0, min(1.0, confidence))

    rationale = str(llm_data.get("rationale") or base_dimension["rationale"]).strip()
    if not rationale:
        rationale = base_dimension["rationale"]

    return {
        "label": label,
        "confidence": confidence,
        "rationale": rationale,
    }


def _apply_llm_refinement(
    record: Dict[str, Any],
    reasoning: Dict[str, Any],
    provider: str,
    model: Optional[str],
    api_key: Optional[str],
) -> Dict[str, Any]:
    if provider == "none":
        return reasoning
    if llm_universal_call_utility is None:
        refined = copy.deepcopy(reasoning)
        refined["llm_error"] = "LLM utility is unavailable."
        return refined

    try:
        response = llm_universal_call_utility(
            prompt=_build_llm_prompt(record, reasoning["labels"]),
            provider=provider,
            api_key=api_key,
            model=model,
        )
        data = _extract_json_object(response)
        refined_labels = {
            "semantic": _sanitize_llm_dimension(data.get("semantic", {}), reasoning["labels"]["semantic"], SEMANTIC_LABELS),
            "performance": _sanitize_llm_dimension(data.get("performance", {}), reasoning["labels"]["performance"], PERFORMANCE_LABELS),
            "risk": _sanitize_llm_dimension(data.get("risk", {}), reasoning["labels"]["risk"], RISK_LABELS),
        }
        refined = copy.deepcopy(reasoning)
        refined["labels"] = refined_labels
        refined["method"] = "rules+llm"
        return refined
    except Exception as exc:
        refined = copy.deepcopy(reasoning)
        refined["llm_error"] = str(exc)
        return refined


def classify_record(
    record: Dict[str, Any],
    provider: str = "none",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    execution_evidence: Optional[Dict[str, Any]] = None,
    use_execution_evidence: bool = True,
) -> Dict[str, Any]:
    """Return a labeled copy of one SQLChange mutation record.

    When execution_evidence is supplied and use_execution_evidence is True:
    - The performance label is derived from measured timing data via
      _performance_label_from_evidence() instead of the static mutation-type rule.
    - The risk label is adjusted by _risk_label_from_evidence(), which can
      escalate the static base risk based on output_relation, row-count growth,
      execution errors, graph_depth, and join_where_table_count.
    Both overrides fall back to the static rule when evidence is absent or ambiguous.
    """
    output = copy.deepcopy(record)
    base_labels = _base_reasoning(record)

    perf_evidence_result = None
    risk_evidence_result = None

    if use_execution_evidence and execution_evidence is not None:
        # Performance override
        perf_evidence_result = _performance_label_from_evidence(execution_evidence)
        if perf_evidence_result["label"] != "unknown":
            base_labels["performance"] = {
                "label": perf_evidence_result["label"],
                "confidence": {"high": 0.88, "medium": 0.72, "low": 0.55}.get(
                    perf_evidence_result["confidence"], 0.55
                ),
                "rationale": perf_evidence_result["reason"],
            }

        # Risk override
        static_risk = base_labels["risk"]["label"]
        risk_evidence_result = _risk_label_from_evidence(record, static_risk, execution_evidence)
        base_labels["risk"] = {
            "label": risk_evidence_result["label"],
            "confidence": {"high": 0.88, "medium": 0.72, "low": 0.55}.get(
                risk_evidence_result["confidence"], 0.55
            ),
            "rationale": risk_evidence_result["reason"],
        }

    reasoning = {
        "method": "rules",
        "signals": _rule_signals(record),
        "labels": base_labels,
    }
    reasoning = _apply_llm_refinement(output, reasoning, provider, model, api_key)

    output["semantic_label"] = reasoning["labels"]["semantic"]["label"]
    output["performance_label"] = reasoning["labels"]["performance"]["label"]
    output["risk_label"] = reasoning["labels"]["risk"]["label"]
    output["reasoning"] = reasoning

    if perf_evidence_result is not None:
        output["performance_evidence"] = perf_evidence_result
    if risk_evidence_result is not None:
        output["risk_evidence"] = risk_evidence_result

    return output


def classify_dataset(
    records: List[Dict[str, Any]],
    provider: str = "none",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    sample_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Classify a list of SQLChange records, optionally limiting to a sample."""
    selected_records = records[:sample_size] if sample_size is not None else records
    return [
        classify_record(record, provider=provider, model=model, api_key=api_key)
        for record in selected_records
    ]
