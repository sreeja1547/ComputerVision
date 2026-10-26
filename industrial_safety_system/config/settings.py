"""Configuration settings for Industrial Hazard Detection & Safety-Level Assessment."""

from pathlib import Path
from pydantic import BaseModel, Field

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = BASE_DIR.parent
DATA_DIR = WORKSPACE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
EVIDENCE_DIR = DATA_DIR / "incident_evidence"

# Default External Data Paths
DEFAULT_ARCHIVE_IMAGES_DIR = Path(r"c:\Users\Dell\Downloads\archive\images")
DEFAULT_SAMPLE_VIDEO_PATH = DATA_DIR / "sample_video" / "industrial_cctv_sample.mp4"

# Models
YOLO_BASE_MODEL_PATH = MODELS_DIR / "yolov8n.pt"
YOLO_PPE_MODEL_PATH = MODELS_DIR / "yolov8n_ppe.pt"

# Ensure evidence dir exists
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


class SafetyThresholds(BaseModel):
    # Detection confidences
    person_confidence: float = 0.35
    ppe_confidence: float = 0.25
    machinery_confidence: float = 0.30

    # Proximity threshold (in normalized frame distance, ~2.5m equivalent)
    machinery_proximity_ratio: float = 0.18

    # Multi-frame verification (consecutive frames to confirm violation)
    temporal_verification_frames: int = 3
    temporal_history_window: int = 15

    # Safety scoring penalties (0-100 base)
    penalty_danger_zone_breach: float = 35.0
    penalty_caution_zone_breach: float = 15.0
    penalty_missing_hardhat: float = 25.0
    penalty_missing_vest: float = 20.0
    penalty_missing_boots: float = 10.0
    penalty_missing_gloves: float = 10.0
    penalty_machinery_proximity: float = 25.0
    persistence_penalty_multiplier: float = 1.25


class AppConfig(BaseModel):
    app_title: str = "AI-Based Industrial Inspection Hazard Detection & Safety-Level Assessment"
    app_tagline: str = "Automated Hazard Identification • Verified Risk Analysis • Smart Incident Response"
    version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    thresholds: SafetyThresholds = Field(default_factory=SafetyThresholds)


settings = AppConfig()
