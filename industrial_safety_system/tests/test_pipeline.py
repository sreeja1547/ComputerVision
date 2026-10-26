"""Integration test for full 6-stage end-to-end pipeline."""

import numpy as np
import pytest
from industrial_safety_system.core.pipeline import IndustrialSafetyPipeline


def test_pipeline_execution():
    pipeline = IndustrialSafetyPipeline(camera_id="CAM-01")
    # Synthetic frame (e.g. 720p dark factory frame)
    synthetic_frame = np.full((720, 1280, 3), 40, dtype=np.uint8)

    res = pipeline.process_frame(synthetic_frame, timestamp=100.0)

    assert res.clean_bgr is not None
    assert res.annotated_bgr is not None
    assert res.risk.safety_score >= 0.0
    assert res.risk.risk_level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    assert res.latency.total_ms > 0.0
