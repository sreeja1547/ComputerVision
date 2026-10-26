"""Stage 2: Frame Preprocessing & Cleanup Module.

Implements:
- Contrast Limited Adaptive Histogram Equalization (CLAHE) for industrial illumination adaptation
- Edge-preserving denoising filter (bilateral/gaussian)
- Dynamic letterbox resizing preserving aspect ratio
- Motion energy gating & background frame differencing
- Illumination quality assessment (shadows, glare, normal)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import cv2
import numpy as np


@dataclass
class PreprocessedOutput:
    clean_bgr: np.ndarray
    clean_rgb: np.ndarray
    analysis_frame: np.ndarray  # Sized for inference e.g. 640x640
    motion_energy: float
    has_motion: bool
    brightness: float
    contrast: float
    lighting_condition: str  # "NORMAL", "UNDEREXPOSED", "OVEREXPOSED"
    scale_ratio: float
    pad_offset: Tuple[int, int]


class FramePreprocessor:
    """Preprocesses CCTV video frames for robust computer vision under challenging industrial factory conditions."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        enable_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        enable_denoise: bool = True,
        motion_threshold: float = 15.0,
    ):
        self.target_size = target_size
        self.enable_clahe = enable_clahe
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
        self.enable_denoise = enable_denoise
        self.motion_threshold = motion_threshold
        self.prev_gray: Optional[np.ndarray] = None

    def assess_lighting(self, gray: np.ndarray) -> Tuple[float, float, str]:
        """Calculates mean brightness and RMS contrast."""
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        if brightness < 60.0:
            condition = "UNDEREXPOSED"
        elif brightness > 200.0:
            condition = "OVEREXPOSED"
        else:
            condition = "NORMAL"
        return brightness, contrast, condition

    def apply_clahe(self, bgr_img: np.ndarray) -> np.ndarray:
        """Applies CLAHE on the L (Luminance) channel in LAB color space to improve visibility without distorting colors."""
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self.clahe.apply(l)
        enhanced_lab = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def apply_denoise(self, bgr_img: np.ndarray) -> np.ndarray:
        """Applies gentle edge-preserving bilateral filter."""
        return cv2.bilateralFilter(bgr_img, d=5, sigmaColor=35, sigmaSpace=35)

    def compute_motion_energy(self, gray: np.ndarray) -> Tuple[float, bool]:
        """Computes motion energy relative to previous frame using absolute difference."""
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray.copy()
            return 0.0, True

        frame_diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        motion_ratio = float(np.count_nonzero(thresh)) / float(gray.size)
        motion_energy = motion_ratio * 100.0
        self.prev_gray = gray.copy()

        has_motion = motion_energy > (self.motion_threshold / 10.0)
        return motion_energy, has_motion

    def letterbox(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Resizes image to target_size with letterbox padding, maintaining aspect ratio."""
        target_w, target_h = self.target_size
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        dx = (target_w - new_w) // 2
        dy = (target_h - new_h) // 2
        canvas[dy : dy + new_h, dx : dx + new_w] = resized

        return canvas, scale, (dx, dy)

    def process(self, raw_bgr: np.ndarray) -> PreprocessedOutput:
        """Executes full stage-2 cleanup and preparation pipeline."""
        clean_bgr = raw_bgr.copy()

        # 1. Denoise
        if self.enable_denoise:
            clean_bgr = self.apply_denoise(clean_bgr)

        # 2. Lighting check
        gray = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY)
        brightness, contrast, lighting_cond = self.assess_lighting(gray)

        # 3. CLAHE enhancement if poor lighting
        if self.enable_clahe and (lighting_cond != "NORMAL" or contrast < 45.0):
            clean_bgr = self.apply_clahe(clean_bgr)
            gray = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY)

        # 4. Motion energy
        motion_energy, has_motion = self.compute_motion_energy(gray)

        # 5. Letterbox resize for YOLO
        analysis_frame, scale, pad = self.letterbox(clean_bgr)
        clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)

        return PreprocessedOutput(
            clean_bgr=clean_bgr,
            clean_rgb=clean_rgb,
            analysis_frame=analysis_frame,
            motion_energy=motion_energy,
            has_motion=has_motion,
            brightness=brightness,
            contrast=contrast,
            lighting_condition=lighting_cond,
            scale_ratio=scale,
            pad_offset=pad,
        )
