"""Visualizer module for Industrial Safety HUD Overlays.

Renders:
- Translucent polygonal safety zones (Danger/Caution/Safe)
- Worker bounding boxes color-coded by compliance
- PPE tags (Hardhat, Vest, Boots, Gloves)
- Heavy machinery indicators and proximity collision warning lines
- Top telemetry HUD bar with Safety Score, Risk Level badge, and FPS
- Critical strobe warning border
"""

from __future__ import annotations
from typing import List, Optional
import cv2
import numpy as np

from industrial_safety_system.core.context_checker import (
    ContextCheckResult,
    ZoneDefinition,
    ProximityAlert,
)
from industrial_safety_system.core.risk_engine import ExplainableRiskAssessment
from industrial_safety_system.core.yolo_detector import WorkerDetection, MachineryDetection


class SafetyVisualizer:
    """Renders professional industrial computer vision overlays on CCTV frames."""

    def __init__(self):
        # Color palette (BGR)
        self.COLOR_DANGER = (42, 42, 255)  # Bright Red
        self.COLOR_CAUTION = (0, 184, 255)  # Amber Yellow
        self.COLOR_SAFE = (102, 255, 0)  # Neon Green
        self.COLOR_MACHINERY = (255, 200, 0)  # Cyan-blue
        self.COLOR_COMPLIANT = (50, 220, 50)  # Green
        self.COLOR_NON_COMPLIANT = (30, 30, 240)  # Red
        self.COLOR_NEUTRAL = (200, 200, 200)

    def draw_zones(self, img: np.ndarray, zones: List[ZoneDefinition]) -> np.ndarray:
        """Draws translucent filled polygons with labeled borders."""
        h, w = img.shape[:2]
        overlay = img.copy()

        for z in zones:
            pts = z.get_pixel_polygon(w, h)
            if len(pts) < 3:
                continue

            color = self.COLOR_DANGER if z.zone_type == "DANGER" else (
                self.COLOR_CAUTION if z.zone_type == "CAUTION" else self.COLOR_SAFE
            )

            # Fill translucent
            cv2.fillPoly(overlay, [pts], color)

        # Alpha blend (0.22 opacity)
        cv2.addWeighted(overlay, 0.22, img, 0.78, 0, img)

        # Draw crisp borders and zone labels
        for z in zones:
            pts = z.get_pixel_polygon(w, h)
            color = self.COLOR_DANGER if z.zone_type == "DANGER" else (
                self.COLOR_CAUTION if z.zone_type == "CAUTION" else self.COLOR_SAFE
            )
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

            # Label near top-left of polygon
            min_x = int(np.min(pts[:, 0]))
            min_y = int(np.min(pts[:, 1]))
            label = f"[{z.zone_type}] {z.name}"
            cv2.rectangle(img, (min_x, max(0, min_y - 20)), (min_x + len(label) * 9, min_y), (20, 20, 20), -1)
            cv2.putText(img, label, (min_x + 4, max(12, min_y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        return img

    def draw_machinery(self, img: np.ndarray, machinery: List[MachineryDetection]) -> np.ndarray:
        """Draws machinery bounding boxes with equipment tags."""
        for m in machinery:
            x1, y1, x2, y2 = m.box.to_int_list()
            cv2.rectangle(img, (x1, y1), (x2, y2), self.COLOR_MACHINERY, 2, cv2.LINE_AA)
            label = f"[MACHINERY] {m.class_name.upper()} #{m.machinery_id} ({int(m.confidence * 100)}%)"
            cv2.rectangle(img, (x1, max(0, y1 - 22)), (x1 + len(label) * 9, y1), (30, 30, 30), -1)
            cv2.putText(img, label, (x1 + 4, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_MACHINERY, 1, cv2.LINE_AA)
        return img

    def draw_workers(self, img: np.ndarray, workers: List[WorkerDetection]) -> np.ndarray:
        """Draws worker bounding boxes and PPE status pills."""
        for w in workers:
            x1, y1, x2, y2 = w.box.to_int_list()
            is_violating = w.is_violating or (w.current_zone and "DANGER" in w.current_zone.upper())
            color = self.COLOR_NON_COMPLIANT if is_violating else self.COLOR_COMPLIANT

            # Main bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Top label (Worker ID + Zone)
            zone_txt = f"[{w.current_zone}]" if w.current_zone else ""
            top_label = f"Worker #{w.worker_id} {zone_txt}"
            (tw, th), _ = cv2.getTextSize(top_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, max(0, y1 - 22)), (x1 + tw + 10, y1), (25, 25, 25), -1)
            cv2.putText(img, top_label, (x1 + 5, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # PPE status badges stacked beneath worker box
            ppe = w.ppe
            h_tag = "HELMET: OK" if ppe.has_helmet else "NO HELMET!"
            v_tag = "VEST: OK" if ppe.has_vest else "NO VEST!"
            h_color = self.COLOR_COMPLIANT if ppe.has_helmet else self.COLOR_DANGER
            v_color = self.COLOR_COMPLIANT if ppe.has_vest else self.COLOR_DANGER

            badge_y = y2 + 14
            # Draw helmet badge
            cv2.rectangle(img, (x1, badge_y - 12), (x1 + 105, badge_y + 2), (20, 20, 20), -1)
            cv2.putText(img, h_tag, (x1 + 4, badge_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, h_color, 1, cv2.LINE_AA)

            # Draw vest badge
            cv2.rectangle(img, (x1 + 110, badge_y - 12), (x1 + 200, badge_y + 2), (20, 20, 20), -1)
            cv2.putText(img, v_tag, (x1 + 114, badge_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, v_color, 1, cv2.LINE_AA)

        return img

    def draw_proximity_lines(
        self,
        img: np.ndarray,
        alerts: List[ProximityAlert],
        workers: List[WorkerDetection],
        machinery: List[MachineryDetection],
    ) -> np.ndarray:
        """Renders proximity hazard connection vectors between workers and machinery."""
        w_map = {w.worker_id: w.box.center for w in workers if w.worker_id}
        m_map = {m.machinery_id: m.box.center for m in machinery}

        for pa in alerts:
            if pa.worker_id in w_map and pa.machinery_id in m_map:
                p1 = (int(w_map[pa.worker_id][0]), int(w_map[pa.worker_id][1]))
                p2 = (int(m_map[pa.machinery_id][0]), int(m_map[pa.machinery_id][1]))

                # Line color
                line_color = self.COLOR_DANGER if pa.is_critical else self.COLOR_CAUTION
                cv2.line(img, p1, p2, line_color, 2, cv2.LINE_AA)

                # Midpoint warning badge
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2
                tag = f"COLLISION HAZARD ({pa.distance_px}px)"
                cv2.rectangle(img, (mid_x - 70, mid_y - 12), (mid_x + 90, mid_y + 6), (20, 20, 20), -1)
                cv2.putText(img, tag, (mid_x - 65, mid_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, line_color, 1, cv2.LINE_AA)

        return img

    def draw_hud(
        self,
        img: np.ndarray,
        risk: ExplainableRiskAssessment,
        fps: float = 30.0,
        camera_id: str = "CAM-01",
    ) -> np.ndarray:
        """Renders top industrial telemetry HUD bar."""
        h, w = img.shape[:2]

        # Top banner background
        hud_h = 50
        cv2.rectangle(img, (0, 0), (w, hud_h), (18, 22, 28), -1)
        cv2.line(img, (0, hud_h), (w, hud_h), (50, 60, 75), 1)

        # 1. Camera ID & FPS
        cv2.putText(img, f"CCTV: {camera_id}", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(img, f"FPS: {fps:.1f}", (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 160, 180), 1, cv2.LINE_AA)

        # 2. Risk Level Badge
        risk_color = (
            self.COLOR_DANGER if risk.risk_level == "CRITICAL" else (
                self.COLOR_CAUTION if risk.risk_level in ["HIGH", "MEDIUM"] else self.COLOR_SAFE
            )
        )
        badge_x = int(w * 0.35)
        cv2.rectangle(img, (badge_x, 8), (badge_x + 190, 42), (35, 42, 54), -1)
        cv2.rectangle(img, (badge_x, 8), (badge_x + 190, 42), risk_color, 2)
        cv2.putText(img, f"RISK: {risk.risk_level}", (badge_x + 12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, risk_color, 2, cv2.LINE_AA)

        # 3. Safety Score Gauge
        score_x = int(w * 0.65)
        score_str = f"SAFETY SCORE: {risk.safety_score:.0f}%"
        cv2.putText(img, score_str, (score_x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1, cv2.LINE_AA)

        # Meter bar
        bar_w = 160
        fill_w = int((risk.safety_score / 100.0) * bar_w)
        cv2.rectangle(img, (score_x, 28), (score_x + bar_w, 42), (40, 50, 65), -1)
        cv2.rectangle(img, (score_x, 28), (score_x + fill_w, 42), risk_color, -1)
        cv2.rectangle(img, (score_x, 28), (score_x + bar_w, 42), (100, 120, 140), 1)

        # Critical Strobe Alert border if CRITICAL
        if risk.risk_level == "CRITICAL":
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), self.COLOR_DANGER, 4)

        return img

    def render(
        self,
        base_bgr: np.ndarray,
        context_res: ContextCheckResult,
        risk: ExplainableRiskAssessment,
        machinery: List[MachineryDetection],
        fps: float = 30.0,
        camera_id: str = "CAM-01",
    ) -> np.ndarray:
        """Executes full rendering pipeline and returns annotated BGR image."""
        canvas = base_bgr.copy()

        # 1. Zones
        canvas = self.draw_zones(canvas, context_res.active_zones)

        # 2. Machinery
        canvas = self.draw_machinery(canvas, machinery)

        # 3. Workers & PPE
        canvas = self.draw_workers(canvas, context_res.tracked_workers)

        # 4. Proximity Lines
        canvas = self.draw_proximity_lines(
            canvas, context_res.proximity_alerts, context_res.tracked_workers, machinery
        )

        # 5. Top HUD & Gauge
        canvas = self.draw_hud(canvas, risk, fps=fps, camera_id=camera_id)

        return canvas
