"""
    This python file makes an LLM call to analyze the performance of a SQL query
    based on execution timing evidence from the synthetic database module. The LLM
    receives the speedup ratios and timing data across different scales and returns
    a performance score out of 10 along with a label classification.

    The score out of 10 is extracted first and used for downstream analysis of
    different query types and for tracking improvements across iterations in the
    recommendation loop.

    Additionally using factiors such as output relation, original row count, modified
    row count, runtime delta ratio (original/modified), runtime delta ms
    row count delta (modified - row time delta)
        row count, total query runtime, 

    Labels: improves, degrades, neutral, unknown
    File Name: performance_labeler.py
"""

import json
import re
from typing import Any, Dict, Optional
 
from utils.llm import llm_universal_call_utility
 
 
# Defining the allowed performance labels
PERFORMANCE_LABELS = {"improves", "degrades", "neutral", "unknown"}
 
 
def classify_performance(
    original_sql: str,
    execution_evidence: Dict[str, Any],
    provider: str = "qwen",
    model: str = None,
    api_key: str = None,
) -> Dict[str, Any]:
    """
        This function sends the execution timing evidence to the LLM and asks it
        to analyze the performance characteristics of the query. The LLM first
        assigns a score out of 10 where 10 means massive improvement and 1 means
        severe degradation, then classifies into a label.
 
        Incorporates output relation, row counts, runtime deltas and error states
        to give the LLM concrete execution signals beyond just speedup ratios.
    """
    # Extracting the performance timing data from the execution evidence
    perf = (execution_evidence or {}).get("performance") or {}
 
    # Building a compact summary of the timing data across all available scales
    timing_summary = {}
    for scale in ("small", "medium", "large"):
        scale_data = perf.get(scale)
        if isinstance(scale_data, dict):
            timing_summary[scale] = {
                "original_ms": scale_data.get("original_ms"),
                "recommended_ms": scale_data.get("modified_ms") or scale_data.get("recommended_ms"),
                "speedup": scale_data.get("speedup"),
            }
 
    # Extracting the equivalence data for additional context
    equivalence = (execution_evidence or {}).get("equivalence") or {}
 
    # Extracting execution factors from the evidence
    output_relation = execution_evidence.get("output_relation", "unknown")
    row_count_original = execution_evidence.get("row_count_original", equivalence.get("original_row_count", "unknown"))
    row_count_modified = execution_evidence.get("row_count_modified", equivalence.get("recommended_row_count", "unknown"))
    row_count_delta = execution_evidence.get("row_count_delta", "unknown")
    runtime_delta_ms = execution_evidence.get("runtime_delta_ms", "unknown")
    runtime_ratio = execution_evidence.get("runtime_ratio", "unknown")
    both_succeeded = execution_evidence.get("both_succeeded", "unknown")
    original_error = execution_evidence.get("original_error", None)
    modified_error = execution_evidence.get("modified_error", None)
 
    # Building the prompt for the LLM to analyze the performance
    prompt = f"""You are a database performance analyst. Analyze the execution evidence for this SQL query and determine if the recommended optimization improves, degrades or maintains performance.
 
Original SQL:
{original_sql}
 
Timing Evidence (across different database scales):
{json.dumps(timing_summary, indent=2)}
 
Execution Factors:
- Output relation: {output_relation}
- Original row count: {row_count_original}
- Modified row count: {row_count_modified}
- Row count delta (modified - original): {row_count_delta}
- Runtime delta ms (original - modified, positive means faster): {runtime_delta_ms}
- Runtime ratio (original / modified, >1 means faster): {runtime_ratio}
- Both succeeded: {both_succeeded}
- Original error: {original_error}
- Modified error: {modified_error}
 
Instructions:
1. Assign a performance score from 1 to 10 where:
   - 1-2: severe degradation (query is much slower or modified query errors out)
   - 3-4: moderate degradation (query is somewhat slower or returns fewer rows unexpectedly)
   - 5: neutral (no meaningful change in runtime or output)
   - 6-7: moderate improvement (query is somewhat faster with correct output)
   - 8-10: significant improvement (query is much faster with correct output)
2. Prioritize total runtime (runtime_delta_ms and runtime_ratio) and output correctness (row_count_delta, both_succeeded) as the primary signals.
3. If both queries run under 0.05ms treat the result as neutral since the noise floor dominates.
4. If the modified query errors out, score 1-2 regardless of other factors.
5. If row counts diverge significantly, flag it even if runtime improves.
6. Larger scale timing results are more reliable than small scale.
 
Respond with ONLY this exact JSON structure. No markdown, no explanation, no text outside the JSON. Keep rationale under 100 words.
{{"score": <integer 1-10>, "label": "improves|degrades|neutral|unknown", "confidence": "high|medium|low", "rationale": "<100 words max explaining the assessment>"}}"""
 
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
 
        # Storing the raw LLM output alongside the parsed result
        result["llm_raw"] = response
        result["timing_summary"] = timing_summary
 
        return result
 
    except Exception as e:
        print(f"Error: performance_labeler.py: LLM call failed: {e}")
        return {
            "score": 5,
            "label": "unknown",
            "confidence": "low",
            "rationale": f"LLM call failed: {e}",
            "llm_raw": None,
            "timing_summary": timing_summary,
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
            return {"score": 5, "label": "unknown", "confidence": "low", "rationale": "Could not parse LLM response"}
        data = json.loads(match.group(0))
 
    # Extracting and validating the score
    score = data.get("score", 5)
    try:
        score = int(score)
        score = max(1, min(10, score))
    except (TypeError, ValueError):
        score = 5
 
    # Extracting and validating the label
    label = data.get("label", "unknown")
    if label not in PERFORMANCE_LABELS:
        label = "unknown"
 
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