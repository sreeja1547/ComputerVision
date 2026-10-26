"""End-to-End Industrial Safety Intelligence Pipeline.

Connects all 6 stages into a unified, high-performance pipeline:
[CCTV Feed] -> [Preprocess] -> [YOLO Detection] -> [Context Check] -> [Risk Engine] -> [Incident Action]
IDENTIFY -> VERIFY -> ASSESS -> EXPLAIN -> ALERT -> RESPOND
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import cv2
import numpy as np

from industrial_safety_system.config.settings import settings
from industrial_safety_system.core.preprocessor import FramePreprocessor, PreprocessedOutput
from industrial_safety_system.core.yolo_detector import (
    YOLOHazardDetector,
    DetectionResult,
)
from industrial_safety_system.core.context_checker import (
    ContextVerificationEngine,
    ContextCheckResult,
)
from industrial_safety_system.core.risk_engine import (
    ExplainableRiskEngine,
    ExplainableRiskAssessment,
)
from industrial_safety_system.core.incident_action import (
    IncidentActionManager,
    IncidentTicket,
)
from industrial_safety_system.utils.visualizer import SafetyVisualizer


@dataclass
class StageLatency:
    preprocess_ms: float
    yolo_ms: float
    context_ms: float
    risk_ms: float
    visualizer_ms: float
    total_ms: float


@dataclass
class PipelineResult:
    timestamp: float
    camera_id: str
    clean_bgr: np.ndarray
    annotated_bgr: np.ndarray
    annotated_rgb: np.ndarray
    preprocess: PreprocessedOutput
    detection: DetectionResult
    context: ContextCheckResult
    risk: ExplainableRiskAssessment
    incident: Optional[IncidentTicket]
    latency: StageLatency


class IndustrialSafetyPipeline:
    """The master pipeline unifying all 6 stages for live video, images, and batch streams."""

    def __init__(
        self,
        camera_id: str = "CAM-01",
        preprocessor: Optional[FramePreprocessor] = None,
        detector: Optional[YOLOHazardDetector] = None,
        context_checker: Optional[ContextVerificationEngine] = None,
        risk_engine: Optional[ExplainableRiskEngine] = None,
        incident_manager: Optional[IncidentActionManager] = None,
        visualizer: Optional[SafetyVisualizer] = None,
    ):
        self.camera_id = camera_id
        self.preprocessor = preprocessor or FramePreprocessor()
        self.detector = detector or YOLOHazardDetector()
        self.context_checker = context_checker or ContextVerificationEngine()
        self.risk_engine = risk_engine or ExplainableRiskEngine()
        self.incident_manager = incident_manager or IncidentActionManager()
        self.visualizer = visualizer or SafetyVisualizer()

        # Set active camera zone configuration
        self.context_checker.set_active_camera(self.camera_id)

    def set_camera(self, camera_id: str):
        self.camera_id = camera_id
        self.context_checker.set_active_camera(camera_id)

    def process_frame(
        self,
        raw_bgr: np.ndarray,
        timestamp: Optional[float] = None,
        fps: float = 30.0,
    ) -> PipelineResult:
        """Processes a single CCTV frame through all 6 pipeline stages."""
        t0 = time.perf_counter()
        now = timestamp or time.time()
        h, w = raw_bgr.shape[:2]

        # Stage 2: Preprocess (Cleanup & Preparation)
        t_prep_start = time.perf_counter()
        prep_out = self.preprocessor.process(raw_bgr)
        t_prep_ms = (time.perf_counter() - t_prep_start) * 1000.0

        # Stage 3: YOLO Detection (Workers, PPE & Hazards)
        t_yolo_start = time.perf_counter()
        det_out = self.detector.detect(prep_out.clean_bgr, timestamp=now)
        t_yolo_ms = (time.perf_counter() - t_yolo_start) * 1000.0

        # Stage 4: Context Check (Multi-Frame & Zone Check)
        t_ctx_start = time.perf_counter()
        ctx_out = self.context_checker.verify(det_out, frame_w=w, frame_h=h)
        t_ctx_ms = (time.perf_counter() - t_ctx_start) * 1000.0

        # Stage 5: Risk Engine (Explainable Risk Level)
        t_risk_start = time.perf_counter()
        risk_out = self.risk_engine.assess(ctx_out)
        t_risk_ms = (time.perf_counter() - t_risk_start) * 1000.0

        # Rendering Overlay
        t_vis_start = time.perf_counter()
        annotated_bgr = self.visualizer.render(
            base_bgr=prep_out.clean_bgr,
            context_res=ctx_out,
            risk=risk_out,
            machinery=det_out.machinery,
            fps=fps,
            camera_id=self.camera_id,
        )
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        t_vis_ms = (time.perf_counter() - t_vis_start) * 1000.0

        # Stage 6: Incident Action (Prioritized Alert & Resolution)
        workers_involved = [w.worker_id for w in ctx_out.tracked_workers if w.worker_id and (w.is_violating or w.current_zone == "DANGER")]
        incident = self.incident_manager.evaluate_and_create_incident(
            risk=risk_out,
            annotated_bgr=annotated_bgr,
            camera_id=self.camera_id,
            workers_involved=workers_involved,
        )

        total_ms = (time.perf_counter() - t0) * 1000.0
        latency = StageLatency(
            preprocess_ms=round(t_prep_ms, 2),
            yolo_ms=round(t_yolo_ms, 2),
            context_ms=round(t_ctx_ms, 2),
            risk_ms=round(t_risk_ms, 2),
            visualizer_ms=round(t_vis_ms, 2),
            total_ms=round(total_ms, 2),
        )

        return PipelineResult(
            timestamp=now,
            camera_id=self.camera_id,
            clean_bgr=prep_out.clean_bgr,
            annotated_bgr=annotated_bgr,
            annotated_rgb=annotated_rgb,
            preprocess=prep_out,
            detection=det_out,
            context=ctx_out,
            risk=risk_out,
            incident=incident,
            latency=latency,
        )
