"""
Tests for evidence-aware risk labeling.

Covers _risk_rank, _risk_from_rank, _escalate_risk,
_risk_label_from_evidence(), and the execution_evidence risk-override
integration path in classify_record().
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from reasoning_pipeline import (
    _risk_rank,
    _risk_from_rank,
    _escalate_risk,
    _risk_label_from_evidence,
    classify_record,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_record(mutation_type="where_drop", er_graph=None):
    return {
        "unique_id": 0,
        "mutation_type": mutation_type,
        "complexity": "basic SQL",
        "original_sql": "SELECT id FROM users WHERE age > 5",
        "modified_sql": "SELECT id FROM users",
        "context": {
            "users": {
                "columns": ["id", "age"],
                "types": {"id": "INT", "age": "INT"},
            }
        },
        "join_keys": [],
        "where_details": [{"condition": "age > 5", "table": "users", "column": "age"}],
        "er_graph": er_graph or {},
    }


def _comparison(
    output_relation="identical",
    row_count_original=10,
    row_count_modified=10,
    both_succeeded=True,
    original_error=None,
    modified_error=None,
):
    return {
        "output_relation": output_relation,
        "row_count_original": row_count_original,
        "row_count_modified": row_count_modified,
        "row_count_delta": row_count_modified - row_count_original,
        "runtime_delta_ms": 0.0,
        "runtime_ratio": 1.0,
        "both_succeeded": both_succeeded,
        "original_error": original_error,
        "modified_error": modified_error,
    }


def _evidence_with_comparison(comp):
    return {"comparison": comp}


def _perf_evidence(scale="large", original_ms=1.0, modified_ms=1.0):
    speedup = original_ms / modified_ms if modified_ms else None
    return {
        "performance": {
            scale: {
                "rows_per_table": {"small": 50, "medium": 500, "large": 5000}[scale],
                "original_ms": original_ms,
                "modified_ms": modified_ms,
                "speedup": speedup,
            }
        }
    }


# ---------------------------------------------------------------------------
# _risk_rank / _risk_from_rank / _escalate_risk unit tests
# ---------------------------------------------------------------------------

class TestRiskHelpers:

    def test_risk_rank_values(self):
        assert _risk_rank("low") == 0
        assert _risk_rank("medium") == 1
        assert _risk_rank("high") == 2

    def test_risk_rank_unknown_defaults_to_medium(self):
        assert _risk_rank("bogus") == 1

    def test_risk_from_rank_values(self):
        assert _risk_from_rank(0) == "low"
        assert _risk_from_rank(1) == "medium"
        assert _risk_from_rank(2) == "high"

    def test_risk_from_rank_clamps_above(self):
        assert _risk_from_rank(5) == "high"

    def test_risk_from_rank_clamps_below(self):
        assert _risk_from_rank(-3) == "low"

    def test_escalate_low_by_1(self):
        assert _escalate_risk("low", 1) == "medium"

    def test_escalate_low_by_2(self):
        assert _escalate_risk("low", 2) == "high"

    def test_escalate_medium_by_1(self):
        assert _escalate_risk("medium", 1) == "high"

    def test_escalate_high_stays_high(self):
        assert _escalate_risk("high", 1) == "high"

    def test_escalate_by_0_unchanged(self):
        assert _escalate_risk("low", 0) == "low"


# ---------------------------------------------------------------------------
# _risk_label_from_evidence unit tests
# ---------------------------------------------------------------------------

class TestRiskLabelFromEvidence:

    # Rule A — output_relation = "different"
    def test_rule_a_different_escalates(self):
        record = _minimal_record()
        comp = _comparison(output_relation="different")
        result = _risk_label_from_evidence(record, "low", _evidence_with_comparison(comp))
        assert result["label"] == "medium"
        assert "output_relation=different" in result["signals"]["triggers"]

    def test_rule_a_identical_no_escalation(self):
        record = _minimal_record()
        comp = _comparison(output_relation="identical")
        result = _risk_label_from_evidence(record, "low", _evidence_with_comparison(comp))
        assert result["label"] == "low"
        assert result["signals"]["escalation_steps"] == 0

    def test_rule_a_broader_no_escalation(self):
        comp = _comparison(output_relation="broader")
        result = _risk_label_from_evidence(_minimal_record(), "medium", _evidence_with_comparison(comp))
        # broader alone doesn't trigger Rule A
        assert result["signals"]["escalation_steps"] == 0

    # Rule B — row_count_modified >= 2x row_count_original
    def test_rule_b_double_rows_escalates(self):
        record = _minimal_record()
        comp = _comparison(output_relation="broader", row_count_original=10, row_count_modified=20, both_succeeded=True)
        result = _risk_label_from_evidence(record, "low", _evidence_with_comparison(comp))
        assert "row_count_delta=10 (2x growth)" in result["signals"]["triggers"]
        assert result["label"] == "medium"

    def test_rule_b_less_than_double_no_escalation(self):
        comp = _comparison(output_relation="broader", row_count_original=10, row_count_modified=15, both_succeeded=True)
        result = _risk_label_from_evidence(_minimal_record(), "low", _evidence_with_comparison(comp))
        assert result["signals"]["escalation_steps"] == 0

    def test_rule_b_zero_original_rows_no_escalation(self):
        # row_count_original=0 → division guard prevents trigger
        comp = _comparison(output_relation="broader", row_count_original=0, row_count_modified=10, both_succeeded=True)
        result = _risk_label_from_evidence(_minimal_record(), "low", _evidence_with_comparison(comp))
        # Rule B should not fire when original is 0
        assert "row_count_delta" not in " ".join(result["signals"]["triggers"])

    # Rule C — execution error
    def test_rule_c_modified_error_escalates(self):
        comp = _comparison(both_succeeded=False, modified_error="syntax error", original_error=None)
        result = _risk_label_from_evidence(_minimal_record(), "low", _evidence_with_comparison(comp))
        assert "execution_error" in result["signals"]["triggers"]

    def test_rule_c_both_succeeded_no_error_trigger(self):
        comp = _comparison(both_succeeded=True, original_error=None, modified_error=None)
        result = _risk_label_from_evidence(_minimal_record(), "low", _evidence_with_comparison(comp))
        assert "execution_error" not in result["signals"]["triggers"]

    # Rule D — graph_depth >= 3
    def test_rule_d_graph_depth_3_escalates(self):
        er = {"graph_depth": 3, "join_where_tables": []}
        record = _minimal_record(er_graph=er)
        result = _risk_label_from_evidence(record, "low", {})
        assert "graph_depth=3" in result["signals"]["triggers"]
        assert result["label"] == "medium"

    def test_rule_d_graph_depth_2_no_escalation(self):
        er = {"graph_depth": 2}
        record = _minimal_record(er_graph=er)
        result = _risk_label_from_evidence(record, "low", {})
        assert result["signals"]["escalation_steps"] == 0

    # Rule E — join_where_table_count >= 2
    def test_rule_e_two_join_where_tables_escalates(self):
        er = {"join_where_tables": ["users", "orders"]}
        record = _minimal_record(er_graph=er)
        result = _risk_label_from_evidence(record, "low", {})
        assert "join_where_table_count=2" in result["signals"]["triggers"]

    def test_rule_e_one_join_where_table_no_escalation(self):
        er = {"join_where_tables": ["users"]}
        record = _minimal_record(er_graph=er)
        result = _risk_label_from_evidence(record, "low", {})
        assert result["signals"]["escalation_steps"] == 0

    # Multiple rules firing together
    def test_multiple_rules_capped_at_high(self):
        er = {"graph_depth": 4, "join_where_tables": ["t1", "t2"]}
        record = _minimal_record(er_graph=er)
        comp = _comparison(output_relation="different", row_count_original=5, row_count_modified=20,
                           both_succeeded=True)
        result = _risk_label_from_evidence(record, "low", _evidence_with_comparison(comp))
        # Multiple rules fired; result must be capped at high
        assert result["label"] == "high"
        assert result["signals"]["escalation_steps"] >= 3

    # No evidence → base_risk unchanged
    def test_no_evidence_returns_base_risk(self):
        result = _risk_label_from_evidence(_minimal_record(), "medium", None)
        assert result["label"] == "medium"
        assert result["signals"]["escalation_steps"] == 0

    def test_empty_evidence_returns_base_risk(self):
        result = _risk_label_from_evidence(_minimal_record(), "medium", {})
        assert result["label"] == "medium"

    # Confidence levels
    def test_high_confidence_when_both_succeeded(self):
        comp = _comparison(both_succeeded=True)
        result = _risk_label_from_evidence(_minimal_record(), "low", _evidence_with_comparison(comp))
        assert result["confidence"] == "high"

    def test_medium_confidence_with_execution_error(self):
        comp = _comparison(both_succeeded=False, modified_error="err")
        result = _risk_label_from_evidence(_minimal_record(), "low", _evidence_with_comparison(comp))
        assert result["confidence"] == "medium"

    def test_medium_confidence_with_graph_depth_no_comparison(self):
        er = {"graph_depth": 2}
        record = _minimal_record(er_graph=er)
        result = _risk_label_from_evidence(record, "low", {})
        assert result["confidence"] == "medium"

    def test_low_confidence_no_signals(self):
        result = _risk_label_from_evidence(_minimal_record(), "low", {})
        assert result["confidence"] == "low"


# ---------------------------------------------------------------------------
# classify_record integration tests
# ---------------------------------------------------------------------------

class TestClassifyRecordRiskEvidence:

    def test_risk_evidence_key_present_in_output(self):
        record = _minimal_record()
        ev = _evidence_with_comparison(_comparison())
        result = classify_record(record, execution_evidence=ev)
        assert "risk_evidence" in result
        assert "signals" in result["risk_evidence"]

    def test_identical_output_no_escalation(self):
        record = _minimal_record(mutation_type="where_drop")
        ev = _evidence_with_comparison(_comparison(output_relation="identical"))
        result = classify_record(record, execution_evidence=ev)
        # where_drop static risk = medium; no escalation triggers → stays medium
        assert result["risk_label"] == "medium"

    def test_different_output_relation_escalates_risk(self):
        record = _minimal_record(mutation_type="where_drop")
        ev = _evidence_with_comparison(_comparison(output_relation="different"))
        result = classify_record(record, execution_evidence=ev)
        # medium → high via Rule A
        assert result["risk_label"] == "high"

    def test_execution_error_escalates_risk(self):
        record = _minimal_record(mutation_type="limit_add")
        ev = _evidence_with_comparison(
            _comparison(both_succeeded=False, modified_error="no such table")
        )
        result = classify_record(record, execution_evidence=ev)
        # limit_add static = low; Rule C → medium
        assert result["risk_label"] == "medium"

    def test_graph_depth_escalates_even_without_comparison(self):
        er = {"graph_depth": 3}
        record = _minimal_record(mutation_type="column_drop", er_graph=er)
        ev = {}  # no comparison key
        result = classify_record(record, execution_evidence=ev)
        # column_drop static = low; Rule D (graph_depth=3) → medium
        assert result["risk_label"] == "medium"

    def test_no_evidence_falls_back_to_static_rule(self):
        record = _minimal_record(mutation_type="where_drop")
        result = classify_record(record)
        assert result["risk_label"] == "medium"
        assert "risk_evidence" not in result

    def test_use_execution_evidence_false_ignores_evidence(self):
        # Use er_graph with only graph_depth (no join_where_tables / cross_table_risk),
        # so the static where_drop rule returns "medium", and we can verify that
        # the evidence-based escalation path is correctly suppressed.
        er = {"graph_depth": 4}
        record = _minimal_record(mutation_type="where_drop", er_graph=er)
        ev = _evidence_with_comparison(_comparison(output_relation="different"))
        result = classify_record(record, execution_evidence=ev, use_execution_evidence=False)
        # All escalation suppressed; static rule = medium
        assert result["risk_label"] == "medium"
        assert "risk_evidence" not in result

    def test_performance_label_still_correct_with_risk_evidence(self):
        """Performance labeling must be unaffected by the risk evidence path."""
        record = _minimal_record(mutation_type="where_drop")
        # Combine performance + comparison evidence
        ev = {**_perf_evidence("large", original_ms=2.0, modified_ms=1.0),
              "comparison": _comparison(output_relation="identical")}
        result = classify_record(record, execution_evidence=ev)
        assert result["performance_label"] == "improves"
        assert result["risk_label"] == "medium"  # no escalation on identical

    def test_semantic_label_unaffected_by_risk_evidence(self):
        record = _minimal_record(mutation_type="where_drop")
        ev = _evidence_with_comparison(_comparison(output_relation="different"))
        result = classify_record(record, execution_evidence=ev)
        # semantic label for where_drop is always "broader"
        assert result["semantic_label"] == "broader"

    def test_high_static_risk_cannot_be_lowered_by_evidence(self):
        """Evidence can only escalate; it must never lower a static "high" risk."""
        record = _minimal_record(mutation_type="join_drop")
        ev = _evidence_with_comparison(_comparison(output_relation="identical", both_succeeded=True))
        result = classify_record(record, execution_evidence=ev)
        # join_drop static = high; no triggers → stays high (not lowered)
        assert result["risk_label"] == "high"

    def test_row_count_growth_escalates_medium_to_high(self):
        record = _minimal_record(mutation_type="where_drop")
        comp = _comparison(
            output_relation="broader",
            row_count_original=10,
            row_count_modified=25,
            both_succeeded=True,
        )
        ev = _evidence_with_comparison(comp)
        result = classify_record(record, execution_evidence=ev)
        # where_drop static = medium; Rule B (25 >= 2*10) → high
        assert result["risk_label"] == "high"
