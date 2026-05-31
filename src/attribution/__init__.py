"""
Attribution module — LLM-as-judge component importance analysis.

Runs query pairs through the full execution pipeline, then asks the LLM
to judge which structural components (WHERE, JOIN, GROUP BY, etc.)
drove each label assessment.

Modules:
    attribution      – core attribution analysis and reporting
    attribution_cli  – CLI entry point for attribution analysis
"""

from .attribution import (
    attribute_record,
    attribute_batch,
    attribute_dataset,
    summarize_attributions,
    print_report,
    print_summary,
)