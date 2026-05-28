"""
Tests for execution-based performance labeling.

Covers _performance_label_from_evidence() and the execution_evidence
integration path in classify_record().
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from reasoning_pipeline import _performance_label_from_evidence, classify_record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evidence(scale, original_ms, modified_ms):
    """Build a minimal execution_evidence dict for one scale."""
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


def _minimal_record(mutation_type="where_drop"):
    """Minimal SQLChange record that classify_record() can process without crashing."""
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
        "er_graph": {},
    }


# ---------------------------------------------------------------------------
# _performance_label_from_evidence unit tests
# ---------------------------------------------------------------------------

class TestPerformanceLabelFromEvidence:

    def test_speedup_2_0_is_improves(self):
        ev = _make_evidence("large", original_ms=2.0, modified_ms=1.0)
        result = _performance_label_from_evidence(ev)
        assert result["label"] == "improves"
        assert result["signals"]["speedup_used"] == 2.0
        assert result["signals"]["scale_used"] == "large"

    def test_speedup_0_5_is_degrades(self):
        ev = _make_evidence("large", original_ms=1.0, modified_ms=2.0)
        result = _performance_label_from_evidence(ev)
        assert result["label"] == "degrades"
        assert result["signals"]["speedup_used"] == 0.5

    def test_speedup_1_02_is_neutral(self):
        ev = _make_evidence("large", original_ms=1.02, modified_ms=1.0)
        result = _performance_label_from_evidence(ev)
        assert result["label"] == "neutral"

    def test_speedup_exactly_at_improve_threshold(self):
        # 1.15 is exactly on the boundary → improves
        ev = _make_evidence("large", original_ms=1.15, modified_ms=1.0)
        result = _performance_label_from_evidence(ev)
        assert result["label"] == "improves"

    def test_speedup_exactly_at_degrade_threshold(self):
        # 0.85 is exactly on the boundary → degrades
        ev = _make_evidence("large", original_ms=0.85, modified_ms=1.0)
        result = _performance_label_from_evidence(ev)
        assert result["label"] == "degrades"

    def test_missing_evidence_is_unknown(self):
        result = _performance_label_from_evidence(None)
        assert result["label"] == "unknown"
        assert result["signals"]["scale_used"] is None

    def test_empty_evidence_is_unknown(self):
        result = _performance_label_from_evidence({})
        assert result["label"] == "unknown"

    def test_evidence_with_none_speedup_is_unknown(self):
        ev = {"performance": {"large": {"original_ms": 1.0, "modified_ms": 1.0, "speedup": None}}}
        result = _performance_label_from_evidence(ev)
        assert result["label"] == "unknown"

    def test_large_scale_preferred_over_medium(self):
        ev = {
            "performance": {
                "large": {"original_ms": 2.0, "modified_ms": 1.0, "speedup": 2.0},
                "medium": {"original_ms": 0.8, "modified_ms": 1.0, "speedup": 0.8},
            }
        }
        result = _performance_label_from_evidence(ev)
        assert result["label"] == "improves"
        assert result["signals"]["scale_used"] == "large"

    def test_medium_scale_used_when_large_missing(self):
        ev = {
            "performance": {
                "medium": {"original_ms": 0.5, "modified_ms": 1.0, "speedup": 0.5},
            }
        }
        result = _performance_label_from_evidence(ev)
        assert result["label"] == "degrades"
        assert result["signals"]["scale_used"] == "medium"

    def test_small_scale_used_as_last_resort(self):
        ev = {
            "performance": {
                "small": {"original_ms": 2.0, "modified_ms": 1.0, "speedup": 2.0},
            }
        }
        result = _performance_label_from_evidence(ev)
        assert result["label"] == "improves"
        assert result["signals"]["scale_used"] == "small"

    # Confidence tests

    def test_large_scale_gives_high_confidence(self):
        ev = _make_evidence("large", original_ms=2.0, modified_ms=1.0)
        assert _performance_label_from_evidence(ev)["confidence"] == "high"

    def test_medium_scale_gives_medium_confidence(self):
        ev = _make_evidence("medium", original_ms=2.0, modified_ms=1.0)
        assert _performance_label_from_evidence(ev)["confidence"] == "medium"

    def test_small_scale_gives_low_confidence(self):
        ev = _make_evidence("small", original_ms=2.0, modified_ms=1.0)
        assert _performance_label_from_evidence(ev)["confidence"] == "low"

    def test_noise_floor_forces_low_confidence(self):
        # Both times below 0.05 ms → confidence must be low even at large scale
        ev = _make_evidence("large", original_ms=0.02, modified_ms=0.01)
        result = _performance_label_from_evidence(ev)
        assert result["confidence"] == "low"
        # Label is still computed (improves) even if confidence is low
        assert result["label"] == "improves"

    def test_missing_performance_key_is_unknown(self):
        result = _performance_label_from_evidence({"other_key": {}})
        assert result["label"] == "unknown"


# ---------------------------------------------------------------------------
# classify_record integration tests
# ---------------------------------------------------------------------------

class TestClassifyRecordWithExecutionEvidence:

    def test_execution_evidence_overrides_static_rule(self):
        """
        where_drop would normally get 'improves' from the static rule.
        Provide evidence showing the modified query is actually slower (degrades).
        The execution-based label must win.
        """
        record = _minimal_record(mutation_type="where_drop")
        # Modified is 3x slower
        evidence = _make_evidence("large", original_ms=1.0, modified_ms=3.0)
        result = classify_record(record, execution_evidence=evidence)
        assert result["performance_label"] == "degrades"

    def test_execution_evidence_can_confirm_static_rule(self):
        """Evidence agrees with the static rule → label is still improves."""
        record = _minimal_record(mutation_type="where_drop")
        evidence = _make_evidence("large", original_ms=2.0, modified_ms=1.0)
        result = classify_record(record, execution_evidence=evidence)
        assert result["performance_label"] == "improves"

    def test_performance_evidence_key_present_in_output(self):
        record = _minimal_record()
        evidence = _make_evidence("large", original_ms=2.0, modified_ms=1.0)
        result = classify_record(record, execution_evidence=evidence)
        assert "performance_evidence" in result
        assert result["performance_evidence"]["signals"]["scale_used"] == "large"

    def test_no_evidence_falls_back_to_static_rule(self):
        """Without execution_evidence, static rule must still produce a non-None label."""
        record = _minimal_record(mutation_type="where_drop")
        result = classify_record(record)
        assert result["performance_label"] == "improves"
        assert "performance_evidence" not in result

    def test_use_execution_evidence_false_ignores_evidence(self):
        """Even if evidence is passed, use_execution_evidence=False must suppress it."""
        record = _minimal_record(mutation_type="where_drop")
        # Provide evidence that would yield 'degrades'
        evidence = _make_evidence("large", original_ms=1.0, modified_ms=3.0)
        result = classify_record(record, execution_evidence=evidence, use_execution_evidence=False)
        # Static rule for where_drop returns 'improves', not 'degrades'
        assert result["performance_label"] == "improves"

    def test_unknown_evidence_falls_back_to_static_rule(self):
        """If evidence yields unknown (no valid speedup), static rule is used as fallback."""
        record = _minimal_record(mutation_type="where_drop")
        bad_evidence = {"performance": {"large": {"original_ms": None, "modified_ms": None, "speedup": None}}}
        result = classify_record(record, execution_evidence=bad_evidence)
        # unknown → falls back to static rule → where_drop → improves
        assert result["performance_label"] == "improves"

    def test_other_labels_unaffected_by_execution_evidence(self):
        """Semantic and risk labels must not change when execution_evidence is provided."""
        record = _minimal_record(mutation_type="where_drop")
        evidence = _make_evidence("large", original_ms=1.0, modified_ms=3.0)
        result = classify_record(record, execution_evidence=evidence)
        # where_drop semantic is always 'broader', risk is 'medium' (no cross-table risk)
        assert result["semantic_label"] == "broader"
        assert result["risk_label"] == "medium"

    def test_degrades_is_now_reachable(self):
        """degrades must appear in the output — it was dead code before this feature."""
        record = _minimal_record(mutation_type="limit_add")
        # Provide evidence that limit_add is actually slower (e.g., overhead)
        evidence = _make_evidence("large", original_ms=1.0, modified_ms=2.0)
        result = classify_record(record, execution_evidence=evidence)
        assert result["performance_label"] == "degrades"
