"""
Recommendation module — LLM-driven SQL query optimization via LangGraph.

Agentic pipeline where the LLM analyzes a query, generates improved versions,
tests them against a synthetic database for equivalence and performance,
and iterates up to N times to find the best optimization.

Modules:
    recommend      – LangGraph state machine for query optimization
    recommend_cli  – CLI entry point for the recommendation engine
"""

from .recommend import recommend
