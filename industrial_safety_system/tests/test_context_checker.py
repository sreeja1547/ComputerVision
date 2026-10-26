"""Unit tests for Stage 4: Context Checker."""

import pytest
from industrial_safety_system.core.context_checker import (
    ContextVerificationEngine,
    ZoneDefinition,
)
from industrial_safety_system.core.yolo_detector import (
    BoundingBox,
    PPEStatus,
    WorkerDetection,
    MachineryDetection,
    DetectionResult,
)


def test_zone_point_in_polygon():
    zone = ZoneDefinition(
        zone_id="Z1",
        name="Test Danger",
        zone_type="DANGER",
        color_hex="#FF0000",
        points=[[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
        required_ppe=["helmet"],
        prohibited_personnel=True,
        description="Test",
    )
    # Point at (100, 100) in 1000x1000 frame is (0.1, 0.1) -> INSIDE
    assert zone.contains_point((100, 100), 1000, 1000) is True
    # Point at (800, 800) in 1000x1000 frame is (0.8, 0.8) -> OUTSIDE
    assert zone.contains_point((800, 800), 1000, 1000) is False


def test_machinery_proximity_alert():
    engine = ContextVerificationEngine()
    w = WorkerDetection(
        worker_id=1,
        box=BoundingBox(100, 100, 150, 200),
        confidence=0.8,
        ppe=PPEStatus(has_helmet=True, has_vest=True),
    )
    # Machinery very close to worker
    m = MachineryDetection(
        machinery_id=1,
        box=BoundingBox(140, 110, 260, 250),
        class_name="forklift",
        confidence=0.85,
    )
    alerts = engine.check_machinery_proximity([w], [m], 1000, 1000)
    assert len(alerts) >= 1
    assert alerts[0].is_critical is True
