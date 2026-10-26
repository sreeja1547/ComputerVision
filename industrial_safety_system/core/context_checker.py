"""Stage 4: Context Check Module (Multi-Frame & Zone Verification).

Implements:
- Polygonal Geofencing (Danger Zones, Caution Zones, Safe Walkways)
- Worker Footprint Containment Verification
- Heavy Machinery Proximity & Collision Risk Calculation
- Multi-Frame Temporal Verification & Object Tracking (filters transient false alarms)
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from industrial_safety_system.config.settings import settings
from industrial_safety_system.core.yolo_detector import (
    BoundingBox,
    DetectionResult,
    WorkerDetection,
    MachineryDetection,
)


@dataclass
class ZoneDefinition:
    zone_id: str
    name: str
    zone_type: str  # "DANGER", "CAUTION", "SAFE"
    color_hex: str
    points: List[List[float]]  # Normalized polygon [[x1, y1], [x2, y2], ...]
    required_ppe: List[str]
    prohibited_personnel: bool
    description: str

    def get_pixel_polygon(self, frame_w: int, frame_h: int) -> np.ndarray:
        pts = np.array(
            [[int(x * frame_w), int(y * frame_h)] for x, y in self.points],
            dtype=np.int32,
        )
        return pts

    def contains_point(self, pt: Tuple[float, float], frame_w: int, frame_h: int) -> bool:
        """Tests if pixel coordinate (x, y) is inside the polygon."""
        poly = self.get_pixel_polygon(frame_w, frame_h)
        # cv2.pointPolygonTest returns > 0 for inside, 0 for edge, < 0 for outside
        dist = cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False)
        return dist >= 0


@dataclass
class ProximityAlert:
    worker_id: int
    machinery_id: int
    machinery_name: str
    distance_px: float
    distance_normalized: float
    is_critical: bool
    description: str


@dataclass
class VerifiedViolation:
    worker_id: int
    violation_type: str
    zone_id: Optional[str]
    consecutive_frames: int
    is_temporal_verified: bool
    evidence_score: float
    details: str


@dataclass
class ContextCheckResult:
    tracked_workers: List[WorkerDetection]
    proximity_alerts: List[ProximityAlert]
    verified_violations: List[VerifiedViolation]
    active_zones: List[ZoneDefinition]
    timestamp: float


class ContextVerificationEngine:
    """Evaluates contextual safety rules, polygonal zones, and multi-frame consistency."""

    def __init__(self, zones_config_path: Optional[Path] = None):
        self.zones_by_camera: Dict[str, List[ZoneDefinition]] = {}
        self.current_camera_id: str = "CAM-01"
        self._load_zones_config(zones_config_path)

        # Worker tracking state: {worker_id: {"box": BoundingBox, "consecutive_violations": {violation_type: count}, "last_seen": timestamp}}
        self.tracking_registry: Dict[int, Dict] = {}
        self.next_worker_id = 1
        self.max_tracking_distance_px = 90.0

    def _load_zones_config(self, config_path: Optional[Path]):
        p = config_path or (Path(__file__).resolve().parent.parent / "config" / "zones_config.json")
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                for cam_id, cam_info in data.get("cameras", {}).items():
                    zone_list = []
                    for z in cam_info.get("zones", []):
                        zone_list.append(
                            ZoneDefinition(
                                zone_id=z["zone_id"],
                                name=z["name"],
                                zone_type=z["type"],
                                color_hex=z.get("color", "#FF0000"),
                                points=z["points"],
                                required_ppe=z.get("required_ppe", []),
                                prohibited_personnel=z.get("prohibited_personnel", False),
                                description=z.get("description", ""),
                            )
                        )
                    self.zones_by_camera[cam_id] = zone_list

    def set_active_camera(self, camera_id: str):
        if camera_id in self.zones_by_camera:
            self.current_camera_id = camera_id

    def get_current_zones(self) -> List[ZoneDefinition]:
        return self.zones_by_camera.get(self.current_camera_id, [])

    def _track_workers(self, workers: List[WorkerDetection]) -> List[WorkerDetection]:
        """Simple greedy IoU / centroid worker tracker to maintain consistent IDs across frames."""
        assigned_workers: List[WorkerDetection] = []
        used_ids = set()

        for w in workers:
            w_center = w.box.center
            best_id: Optional[int] = None
            min_dist = float("inf")

            for tracked_id, info in self.tracking_registry.items():
                if tracked_id in used_ids:
                    continue
                t_center = info["box"].center
                dist = math.dist(w_center, t_center)
                iou = w.box.iou(info["box"])

                if dist < self.max_tracking_distance_px or iou > 0.3:
                    if dist < min_dist:
                        min_dist = dist
                        best_id = tracked_id

            if best_id is None:
                best_id = self.next_worker_id
                self.next_worker_id += 1
                self.tracking_registry[best_id] = {
                    "box": w.box,
                    "consecutive_violations": {},
                    "history": [],
                }

            used_ids.add(best_id)
            w.worker_id = best_id
            self.tracking_registry[best_id]["box"] = w.box
            assigned_workers.append(w)

        # Cleanup lost tracks
        lost_ids = [tid for tid in self.tracking_registry if tid not in used_ids]
        for lid in lost_ids:
            # Drop older tracks
            del self.tracking_registry[lid]

        return assigned_workers

    def evaluate_zones(
        self, workers: List[WorkerDetection], frame_w: int, frame_h: int
    ) -> List[VerifiedViolation]:
        """Checks worker foot coordinates against defined zones and required PPE."""
        violations: List[VerifiedViolation] = []
        zones = self.get_current_zones()

        for w in workers:
            foot_pt = w.box.bottom_center
            matched_zone: Optional[ZoneDefinition] = None

            for z in zones:
                if z.contains_point(foot_pt, frame_w, frame_h):
                    matched_zone = z
                    break

            if matched_zone:
                w.current_zone = matched_zone.name
                # Danger Zone check
                if matched_zone.zone_type == "DANGER" and matched_zone.prohibited_personnel:
                    violations.append(
                        VerifiedViolation(
                            worker_id=w.worker_id or 1,
                            violation_type="danger_zone_intrusion",
                            zone_id=matched_zone.zone_id,
                            consecutive_frames=1,
                            is_temporal_verified=False,
                            evidence_score=0.92,
                            details=f"Worker #{w.worker_id} encroached prohibited exclusion zone: {matched_zone.name}",
                        )
                    )

                # Required PPE in zone
                for req in matched_zone.required_ppe:
                    if req == "helmet" and not w.ppe.has_helmet:
                        violations.append(
                            VerifiedViolation(
                                worker_id=w.worker_id or 1,
                                violation_type="missing_helmet_in_zone",
                                zone_id=matched_zone.zone_id,
                                consecutive_frames=1,
                                is_temporal_verified=False,
                                evidence_score=0.88,
                                details=f"Worker #{w.worker_id} in {matched_zone.name} without mandatory hardhat",
                            )
                        )
                    elif req == "vest" and not w.ppe.has_vest:
                        violations.append(
                            VerifiedViolation(
                                worker_id=w.worker_id or 1,
                                violation_type="missing_vest_in_zone",
                                zone_id=matched_zone.zone_id,
                                consecutive_frames=1,
                                is_temporal_verified=False,
                                evidence_score=0.85,
                                details=f"Worker #{w.worker_id} in {matched_zone.name} without mandatory safety vest",
                            )
                        )
            else:
                w.current_zone = "General Floor"
                # Check baseline PPE violations
                if not w.ppe.has_helmet:
                    violations.append(
                        VerifiedViolation(
                            worker_id=w.worker_id or 1,
                            violation_type="missing_helmet",
                            zone_id=None,
                            consecutive_frames=1,
                            is_temporal_verified=False,
                            evidence_score=0.82,
                            details=f"Worker #{w.worker_id} observed without head protection",
                        )
                    )
                if not w.ppe.has_vest:
                    violations.append(
                        VerifiedViolation(
                            worker_id=w.worker_id or 1,
                            violation_type="missing_vest",
                            zone_id=None,
                            consecutive_frames=1,
                            is_temporal_verified=False,
                            evidence_score=0.79,
                            details=f"Worker #{w.worker_id} observed without high-vis safety vest",
                        )
                    )

        return violations

    def check_machinery_proximity(
        self,
        workers: List[WorkerDetection],
        machinery: List[MachineryDetection],
        frame_w: int,
        frame_h: int,
    ) -> List[ProximityAlert]:
        """Computes distance between workers and active machinery/forklifts."""
        alerts: List[ProximityAlert] = []
        frame_diag = math.hypot(frame_w, frame_h)
        prox_thresh_px = frame_diag * settings.thresholds.machinery_proximity_ratio

        for w in workers:
            w_center = w.box.center
            for m in machinery:
                m_center = m.box.center
                dist_px = math.dist(w_center, m_center)
                dist_norm = dist_px / frame_diag

                if dist_px < prox_thresh_px:
                    is_critical = dist_norm < (settings.thresholds.machinery_proximity_ratio * 0.6)
                    alerts.append(
                        ProximityAlert(
                            worker_id=w.worker_id or 1,
                            machinery_id=m.machinery_id,
                            machinery_name=m.class_name.capitalize(),
                            distance_px=round(dist_px, 1),
                            distance_normalized=round(dist_norm, 3),
                            is_critical=is_critical,
                            description=(
                                f"Worker #{w.worker_id} within hazardous proximity "
                                f"({round(dist_norm * 10, 1)}m) of operational {m.class_name.upper()}"
                            ),
                        )
                    )

        return alerts

    def verify(
        self,
        detection_res: DetectionResult,
        frame_w: int,
        frame_h: int,
    ) -> ContextCheckResult:
        """Executes full Stage 4 context check with multi-frame persistence verification."""
        # 1. Track workers
        tracked_workers = self._track_workers(detection_res.workers)

        # 2. Zone evaluation
        raw_violations = self.evaluate_zones(tracked_workers, frame_w, frame_h)

        # 3. Machinery proximity check
        proximity_alerts = self.check_machinery_proximity(
            tracked_workers, detection_res.machinery, frame_w, frame_h
        )

        # Add proximity violations
        for pa in proximity_alerts:
            raw_violations.append(
                VerifiedViolation(
                    worker_id=pa.worker_id,
                    violation_type="machinery_proximity_hazard",
                    zone_id=None,
                    consecutive_frames=1,
                    is_temporal_verified=False,
                    evidence_score=0.95 if pa.is_critical else 0.85,
                    details=pa.description,
                )
            )

        # 4. Multi-frame temporal confirmation
        verified_violations: List[VerifiedViolation] = []
        current_frame_violation_keys = set()

        for v in raw_violations:
            key = (v.worker_id, v.violation_type)
            current_frame_violation_keys.add(key)
            w_state = self.tracking_registry.get(v.worker_id, {})
            v_counts = w_state.setdefault("consecutive_violations", {})
            v_counts[v.violation_type] = v_counts.get(v.violation_type, 0) + 1
            consec = v_counts[v.violation_type]

            # Mark verified if sustained across threshold
            is_verified = consec >= settings.thresholds.temporal_verification_frames
            v.consecutive_frames = consec
            v.is_temporal_verified = is_verified
            verified_violations.append(v)

        # Reset counts for violations that resolved
        for wid, w_state in self.tracking_registry.items():
            v_counts = w_state.get("consecutive_violations", {})
            for v_type in list(v_counts.keys()):
                if (wid, v_type) not in current_frame_violation_keys:
                    v_counts[v_type] = 0

        return ContextCheckResult(
            tracked_workers=tracked_workers,
            proximity_alerts=proximity_alerts,
            verified_violations=verified_violations,
            active_zones=self.get_current_zones(),
            timestamp=detection_res.timestamp,
        )
