"""
Reasoning module — evidence-based labeling for SQL query changes.

Assigns semantic, performance, and risk labels based on execution evidence
(equivalence check results, timing benchmarks) and ER graph context.
Optional LLM refinement pass for rationale generation.

Modules:
    reasoning_pipeline – classify query pairs using execution evidence + optional LLM
"""

from .reasoning_pipeline import classify_record, classify_dataset