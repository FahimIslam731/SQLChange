"""
    Semantic labeler — classifies the semantic relationship between original
    and modified SQL queries.

    Uses deterministic rules first (based on execution evidence like output
    relation, row counts, column overlap) then optionally refines with an
    LLM call for nuanced cases.

    Labels: equivalent, narrower, broader, different
    File Name: semantic_labeler.py
"""

import json
import re
from typing import Any, Dict, Optional

from utils.llm import llm_universal_call_utility


# Defining the allowed semantic labels
SEMANTIC_LABELS = {"equivalent", "narrower", "broader", "different"}


def classify_semantic(
    original_sql: str,
    modified_sql: str,
    execution_evidence: Dict[str, Any] = None,
    provider: str = "qwen",
    model: str = None,
    api_key: str = None,
) -> Dict[str, Any]:
    """
        Classifies the semantic relationship between two SQL queries.

        Step 1 — Deterministic rules (fast, no LLM cost):
          - If output_relation from execution harness is "identical" → equivalent
          - If output_relation is "narrower" → narrower
          - If output_relation is "broader"  → broader
          - If output_relation is "error" and modified errors → different
          - If output_relation is "different" → different

        Step 2 — LLM refinement (only when deterministic is low-confidence):
          - Ambiguous row-count deltas
          - Error in one query but not the other
          - No execution evidence available
    """
    exec_ev = execution_evidence or {}
    comparison = exec_ev.get("comparison", exec_ev)

    output_relation = comparison.get("output_relation", "unknown")
    row_original = comparison.get("row_count_original", 0)
    row_modified = comparison.get("row_count_modified", 0)
    both_ok = comparison.get("both_succeeded", True)
    orig_error = comparison.get("original_error")
    mod_error = comparison.get("modified_error")

    # ── Step 1: Deterministic classification ──
    det_label, det_confidence, det_rationale = _deterministic_classify(
        output_relation, row_original, row_modified, both_ok, orig_error, mod_error
    )

    # If deterministic is high confidence, skip LLM
    if det_confidence == "high":
        return {
            "label": det_label,
            "confidence": det_confidence,
            "rationale": det_rationale,
            "method": "deterministic",
            "llm_raw": None,
        }

    # ── Step 2: LLM refinement ──
    try:
        prompt = f"""You are a SQL semantics analyst. Compare the original and modified SQL queries below and classify their semantic relationship.

Original SQL:
{original_sql}

Modified SQL:
{modified_sql}

Execution Evidence:
- Output relation (from test harness): {output_relation}
- Original row count: {row_original}
- Modified row count: {row_modified}
- Both succeeded: {both_ok}
- Original error: {orig_error}
- Modified error: {mod_error}

Deterministic pre-label: {det_label} (confidence: {det_confidence})
Deterministic rationale: {det_rationale}

Instructions:
1. Classify the semantic relationship as one of: equivalent, narrower, broader, different.
   - equivalent: queries return the same rows for any dataset
   - narrower: modified returns a strict subset of original rows
   - broader: modified returns a strict superset of original rows
   - different: modified returns rows that are neither a subset nor superset

2. Consider structural differences (WHERE clauses added/removed, JOIN type changes, column list changes).
3. Use the execution evidence as supporting signal but reason about semantics structurally.

Respond with ONLY this JSON. No markdown. Keep rationale under 80 words.
{{"label": "equivalent|narrower|broader|different", "confidence": "high|medium|low", "rationale": "<80 words max>"}}"""

        response = llm_universal_call_utility(
            prompt=prompt,
            provider=provider,
            model=model,
            api_key=api_key,
            num_predict=200,
        )

        result = _parse_llm_response(response)
        result["method"] = "llm_refined"
        result["llm_raw"] = response
        result["deterministic_pre_label"] = det_label
        return result

    except Exception as e:
        # Fall back to deterministic result
        return {
            "label": det_label,
            "confidence": det_confidence,
            "rationale": f"{det_rationale} (LLM refinement failed: {e})",
            "method": "deterministic_fallback",
            "llm_raw": None,
        }


def _deterministic_classify(output_relation, row_orig, row_mod, both_ok, orig_err, mod_err):
    """Apply deterministic rules to classify semantic relationship."""

    # If modified query errors out, it's semantically different
    if mod_err and not orig_err:
        return "different", "high", "Modified query errors out while original succeeds."

    # If both error, can't determine — but they're both broken
    if orig_err and mod_err:
        return "different", "medium", "Both queries error; cannot determine semantic equivalence."

    # Map execution output_relation directly
    if output_relation == "identical":
        return "equivalent", "high", "Test harness confirms identical row sets."

    if output_relation == "narrower":
        return "narrower", "high", f"Modified returns fewer rows ({row_mod} vs {row_orig})."

    if output_relation == "broader":
        return "broader", "high", f"Modified returns more rows ({row_mod} vs {row_orig})."

    if output_relation == "different":
        return "different", "high", "Row sets differ — neither is a subset of the other."

    # Unknown / no evidence → low confidence, needs LLM
    return "equivalent", "low", "No execution evidence; defaulting to equivalent pending LLM review."


def _parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse the LLM JSON response for semantic classification."""
    clean = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            return {"label": "different", "confidence": "low", "rationale": "Could not parse LLM response"}
        data = json.loads(match.group(0))

    label = data.get("label", "different")
    if label not in SEMANTIC_LABELS:
        label = "different"

    confidence = data.get("confidence", "low")
    if confidence not in ("high", "medium", "low"):
        confidence = "low"

    rationale = str(data.get("rationale", "")).strip()
    words = rationale.split()
    if len(words) > 80:
        rationale = " ".join(words[:80])

    return {"label": label, "confidence": confidence, "rationale": rationale}
