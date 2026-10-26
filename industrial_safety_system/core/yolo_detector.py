"""Stage 3: YOLO Detection Module for Workers, PPE & Industrial Hazards.

Performs:
- Person & Worker localization
- Heavy machinery, vehicle, and forklift identification
- Multi-class PPE detection (Hardhats, High-Vis Vests, Gloves, Boots, Goggles)
- Anatomical region context analysis (Head/Torso/Feet) for robust verification
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO

from industrial_safety_system.config.settings import (
    YOLO_BASE_MODEL_PATH,
    YOLO_PPE_MODEL_PATH,
    settings,
)


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Foot position of worker, ideal for ground-plane zone checks."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def to_int_list(self) -> List[int]:
        return [int(self.x1), int(self.y1), int(self.x2), int(self.y2)]

    def to_normalized(self, frame_w: int, frame_h: int) -> List[float]:
        return [
            self.x1 / frame_w,
            self.y1 / frame_h,
            self.x2 / frame_w,
            self.y2 / frame_h,
        ]

    def iou(self, other: BoundingBox) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union_area = (self.width * self.height) + (other.width * other.height) - inter_area
        return inter_area / union_area if union_area > 0 else 0.0


@dataclass
class PPEStatus:
    has_helmet: bool = False
    has_vest: bool = False
    has_gloves: bool = False
    has_boots: bool = True
    has_goggles: bool = False
    confidences: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)


@dataclass
class WorkerDetection:
    worker_id: Optional[int]  # Assigned or tracked ID
    box: BoundingBox
    confidence: float
    ppe: PPEStatus
    current_zone: Optional[str] = None
    is_violating: bool = False
    violations: List[str] = field(default_factory=list)


@dataclass
class MachineryDetection:
    machinery_id: int
    box: BoundingBox
    class_name: str
    confidence: float


@dataclass
class DetectionResult:
    workers: List[WorkerDetection]
    machinery: List[MachineryDetection]
    raw_ppe_detections: List[Dict]
    timestamp: float


class YOLOHazardDetector:
    """Industrial YOLO detection engine integrating Base YOLO and PPE Models."""

    def __init__(
        self,
        base_weights: Optional[Path] = None,
        ppe_weights: Optional[Path] = None,
    ):
        base_p = str(base_weights or YOLO_BASE_MODEL_PATH)
        ppe_p = str(ppe_weights or YOLO_PPE_MODEL_PATH)

        self.base_model = YOLO(base_p)
        self.ppe_model = YOLO(ppe_p) if Path(ppe_p).exists() else None

        # COCO machinery classes of interest
        self.machinery_classes = {"truck", "bus", "car", "train", "motorcycle", "forklift", "tractor"}

    def _analyze_high_vis_vest(self, torso_crop: np.ndarray) -> Tuple[bool, float]:
        """Analyzes torso crop in HSV color space for fluorescent yellow-green and orange high-visibility vest hues."""
        if torso_crop.size == 0 or torso_crop.shape[0] < 8 or torso_crop.shape[1] < 8:
            return False, 0.0

        hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # High-Vis Neon Yellow/Green: Hue 25-50, Saturation > 70, Value > 80
        mask_yellow_green = cv2.inRange(hsv, np.array([25, 70, 80]), np.array([55, 255, 255]))

        # High-Vis Safety Orange: Hue 5-22, Saturation > 90, Value > 80
        mask_orange = cv2.inRange(hsv, np.array([5, 90, 80]), np.array([22, 255, 255]))

        # High-Vis Neon Red/Coral: Hue 0-5 or 170-180
        mask_red1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([5, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([170, 100, 80]), np.array([180, 255, 255]))

        combined = mask_yellow_green | mask_orange | mask_red1 | mask_red2
        coverage = float(np.count_nonzero(combined)) / float(torso_crop.shape[0] * torso_crop.shape[1])

        # If more than 14% of the torso matches certified high-vis colors
        is_vest = coverage > 0.14
        confidence = min(1.0, max(0.40, coverage * 3.5)) if is_vest else max(0.10, 1.0 - (coverage * 4.0))

        return is_vest, confidence

    def _analyze_helmet(self, head_crop: np.ndarray) -> Tuple[bool, float]:
        """Analyzes head crop in HSV for hardhat colors (Yellow, White, Bright Red, Blue, Orange)."""
        if head_crop.size == 0 or head_crop.shape[0] < 8 or head_crop.shape[1] < 8:
            return False, 0.0

        hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Yellow/White/Orange/Blue Hardhats
        mask_yellow = cv2.inRange(hsv, np.array([20, 70, 90]), np.array([38, 255, 255]))
        mask_white = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 45, 255]))
        mask_orange = cv2.inRange(hsv, np.array([7, 100, 100]), np.array([20, 255, 255]))
        mask_blue = cv2.inRange(hsv, np.array([95, 80, 80]), np.array([130, 255, 255]))

        combined = mask_yellow | mask_white | mask_orange | mask_blue
        coverage = float(np.count_nonzero(combined)) / float(head_crop.shape[0] * head_crop.shape[1])

        is_helmet = coverage > 0.22
        confidence = min(1.0, max(0.45, coverage * 3.0)) if is_helmet else max(0.15, 1.0 - (coverage * 3.0))

        return is_helmet, confidence

    def detect(self, bgr_image: np.ndarray, timestamp: float = 0.0) -> DetectionResult:
        """Runs multi-model inference and anatomical region safety evaluation."""
        frame_h, frame_w = bgr_image.shape[:2]

        # 1. Base YOLO inference
        base_res = self.base_model(
            bgr_image,
            conf=settings.thresholds.person_confidence,
            verbose=False,
        )[0]

        detected_workers_boxes: List[Tuple[BoundingBox, float]] = []
        detected_machinery: List[MachineryDetection] = []
        machinery_id_counter = 1

        for b in base_res.boxes:
            cls_id = int(b.cls[0])
            name = base_res.names[cls_id]
            conf = float(b.conf[0])
            xyxy = b.xyxy[0].tolist()
            box = BoundingBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3])

            if name == "person" and conf >= settings.thresholds.person_confidence:
                # Discard micro-artefacts (less than 20x20 px)
                if box.width > 20 and box.height > 25:
                    detected_workers_boxes.append((box, conf))
            elif name in self.machinery_classes and conf >= settings.thresholds.machinery_confidence:
                detected_machinery.append(
                    MachineryDetection(
                        machinery_id=machinery_id_counter,
                        box=box,
                        class_name=name,
                        confidence=conf,
                    )
                )
                machinery_id_counter += 1

        # 2. PPE Model inference (if loaded)
        raw_ppe_detections: List[Dict] = []
        if self.ppe_model:
            ppe_res = self.ppe_model(
                bgr_image,
                conf=settings.thresholds.ppe_confidence,
                verbose=False,
            )[0]
            for b in ppe_res.boxes:
                cls_id = int(b.cls[0])
                name = ppe_res.names[cls_id]
                conf = float(b.conf[0])
                xyxy = b.xyxy[0].tolist()
                raw_ppe_detections.append(
                    {
                        "name": name,
                        "confidence": conf,
                        "box": BoundingBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3]),
                    }
                )

        # 3. Associate PPE detections & regional safety checks per worker
        workers: List[WorkerDetection] = []
        for idx, (w_box, p_conf) in enumerate(detected_workers_boxes, start=1):
            ppe_status = PPEStatus()

            # Find PPE detections overlapping with this worker
            w_ppe_items = [
                d for d in raw_ppe_detections if d["box"].iou(w_box) > 0.05 or (
                    d["box"].x1 >= w_box.x1 - 10 and d["box"].x2 <= w_box.x2 + 10 and
                    d["box"].y1 >= w_box.y1 - 10 and d["box"].y2 <= w_box.y2 + 10
                )
            ]

            # Check for direct PPE detections
            for item in w_ppe_items:
                lbl = item["name"]
                c = item["confidence"]
                if lbl == "helmet":
                    ppe_status.has_helmet = True
                    ppe_status.confidences["helmet"] = max(c, ppe_status.confidences.get("helmet", 0.0))
                elif lbl == "no_helmet":
                    ppe_status.has_helmet = False
                    ppe_status.confidences["helmet"] = max(c, ppe_status.confidences.get("helmet", 0.0))
                elif lbl == "glove":
                    ppe_status.has_gloves = True
                    ppe_status.confidences["gloves"] = max(c, ppe_status.confidences.get("gloves", 0.0))
                elif lbl == "no_glove":
                    ppe_status.has_gloves = False
                    ppe_status.confidences["gloves"] = max(c, ppe_status.confidences.get("gloves", 0.0))
                elif lbl == "goggles":
                    ppe_status.has_goggles = True
                    ppe_status.confidences["goggles"] = max(c, ppe_status.confidences.get("goggles", 0.0))
                elif lbl == "shoes":
                    ppe_status.has_boots = True
                    ppe_status.confidences["boots"] = max(c, ppe_status.confidences.get("boots", 0.0))
                elif lbl == "no_shoes":
                    ppe_status.has_boots = False
                    ppe_status.confidences["boots"] = max(c, ppe_status.confidences.get("boots", 0.0))

            # Anatomical crops for color / feature verification
            ix1, iy1, ix2, iy2 = w_box.to_int_list()
            ix1, iy1 = max(0, ix1), max(0, iy1)
            ix2, iy2 = min(frame_w, ix2), min(frame_h, iy2)
            w_h = iy2 - iy1

            if w_h > 20:
                # Head: top 20%
                head_crop = bgr_image[iy1 : iy1 + int(w_h * 0.22), ix1:ix2]
                helmet_detected, helmet_conf = self._analyze_helmet(head_crop)

                # Torso: 20% to 55%
                torso_crop = bgr_image[iy1 + int(w_h * 0.20) : iy1 + int(w_h * 0.58), ix1:ix2]
                vest_detected, vest_conf = self._analyze_high_vis_vest(torso_crop)

                # Fuse detector + anatomical analysis for helmet
                if "helmet" not in ppe_status.confidences:
                    ppe_status.has_helmet = helmet_detected
                    ppe_status.confidences["helmet"] = helmet_conf

                # Vest is determined by high-visibility analysis
                ppe_status.has_vest = vest_detected
                ppe_status.confidences["vest"] = vest_conf

            # Compile worker violation tags
            w_violations: List[str] = []
            if not ppe_status.has_helmet:
                w_violations.append("missing_helmet")
                ppe_status.reasons.append("Missing protective safety hardhat")
            if not ppe_status.has_vest:
                w_violations.append("missing_vest")
                ppe_status.reasons.append("Missing high-visibility safety vest")
            if not ppe_status.has_gloves and "gloves" in ppe_status.confidences:
                w_violations.append("missing_gloves")
                ppe_status.reasons.append("Operating without safety gloves")
            if not ppe_status.has_boots and "boots" in ppe_status.confidences:
                w_violations.append("missing_boots")
                ppe_status.reasons.append("Unapproved footwear in industrial zone")

            worker = WorkerDetection(
                worker_id=idx,
                box=w_box,
                confidence=p_conf,
                ppe=ppe_status,
                is_violating=len(w_violations) > 0,
                violations=w_violations,
            )
            workers.append(worker)

        return DetectionResult(
            workers=workers,
            machinery=detected_machinery,
            raw_ppe_detections=raw_ppe_detections,
            timestamp=timestamp,
        )
