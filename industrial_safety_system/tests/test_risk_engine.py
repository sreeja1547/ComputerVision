"""Unit tests for Stage 5: Explainable Risk Engine."""

import pytest
from industrial_safety_system.core.context_checker import ContextCheckResult, VerifiedViolation
from industrial_safety_system.core.risk_engine import ExplainableRiskEngine


def test_perfect_safety_score():
    engine = ExplainableRiskEngine()
    empty_ctx = ContextCheckResult(
        tracked_workers=[],
        proximity_alerts=[],
        verified_violations=[],
        active_zones=[],
        timestamp=1000.0,
    )
    res = engine.assess(empty_ctx)
    assert res.safety_score == 100.0
    assert res.risk_level == "LOW"
    assert "Full compliance" in res.bullet_reasons[0]


def test_danger_zone_intrusion_penalty():
    engine = ExplainableRiskEngine()
    violation = VerifiedViolation(
        worker_id=1,
        violation_type="danger_zone_intrusion",
        zone_id="ZONE-01-DANGER",
        consecutive_frames=4,
        is_temporal_verified=True,
        evidence_score=0.95,
        details="Encroached danger zone",
    )
    ctx = ContextCheckResult(
        tracked_workers=[],
        proximity_alerts=[],
        verified_violations=[violation],
        active_zones=[],
        timestamp=1000.0,
    )
    res = engine.assess(ctx)
    # 100 - (35 * 1.25) = 100 - 43.8 = 56.2
    assert res.safety_score < 70.0
    assert res.risk_level in ["HIGH", "CRITICAL"]
    assert len(res.contributing_factors) == 1
    assert "Exclusion Danger Zone Intrusion" in res.contributing_factors[0].factor
