"""
Graph module — ER graph construction via LangGraph.

Builds table importance hierarchy, join relationships, WHERE dependencies,
and cross-table risk detection. Routes through Python (when join keys exist)
or LLM inference (when relationships must be inferred).

Modules:
    graph_representer – LangGraph pipeline for ER graph construction
"""

from .graph_representer import build_graph
