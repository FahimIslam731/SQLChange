"""
Recommendation module — LLM-driven SQL query optimization.

BUG FIX: Was importing nonexistent function 'recommend'.
         The actual function name is 'recommend_query'.
"""

from .recommend import recommend_query
