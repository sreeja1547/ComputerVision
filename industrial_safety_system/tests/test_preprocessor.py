"""Unit tests for Stage 2: Preprocessor."""

import numpy as np
import pytest
from industrial_safety_system.core.preprocessor import FramePreprocessor


def test_lighting_assessment():
    prep = FramePreprocessor()
    dark_frame = np.zeros((100, 100), dtype=np.uint8)
    b, c, cond = prep.assess_lighting(dark_frame)
    assert cond == "UNDEREXPOSED"

    bright_frame = np.full((100, 100), 220, dtype=np.uint8)
    b, c, cond = prep.assess_lighting(bright_frame)
    assert cond == "OVEREXPOSED"


def test_letterbox_shape():
    prep = FramePreprocessor(target_size=(640, 640))
    sample_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    boxed, scale, pad = prep.letterbox(sample_img)
    assert boxed.shape == (640, 640, 3)
    assert scale < 1.0


def test_motion_energy():
    prep = FramePreprocessor()
    f1 = np.zeros((100, 100), dtype=np.uint8)
    f2 = np.full((100, 100), 255, dtype=np.uint8)

    _, _ = prep.compute_motion_energy(f1)
    energy, has_motion = prep.compute_motion_energy(f2)
    assert energy > 50.0
    assert has_motion is True
