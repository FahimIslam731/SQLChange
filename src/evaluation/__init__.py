"""
Evaluation module — baseline comparison framework.

Compares 3 approaches to measure whether structured analysis improves
recommendation quality over naive LLM approaches:
    1. zero_shot   – give LLM the query, ask for optimization advice
    2. structured  – step-by-step instructions, no execution evidence
    3. sqlchange   – full pipeline (execution evidence + ER graph + rules)

Modules:
    evaluation – baseline runners, metrics computation, reporting
"""

from .evaluation import (
    evaluate_record,
    evaluate_dataset,
    compute_metrics,
    print_metrics,
)
