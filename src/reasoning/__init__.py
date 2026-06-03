"""
Reasoning module — evidence-based labeling for SQL query changes.

Assigns semantic, performance, and risk labels based on execution evidence
(equivalence check results, timing benchmarks) and ER graph context.
Optional LLM refinement pass for rationale generation.

BUG FIX: Was importing from nonexistent 'reasoning_pipeline' module.
         Now correctly imports from the actual labeler modules.
"""

from .performance_labeler import classify_performance
from .risk_labeler import classify_risk
from .semantic_labeler import classify_semantic
