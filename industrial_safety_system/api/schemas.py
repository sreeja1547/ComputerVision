"""Pydantic API request and response schemas."""

from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: Dict[str, bool]
    timestamp: float


class WorkerItem(BaseModel):
    worker_id: int
    bounding_box: List[float]
    confidence: float
    has_helmet: bool
    has_vest: bool
    has_gloves: bool
    has_boots: bool
    current_zone: Optional[str]
    violations: List[str]


class MachineryItem(BaseModel):
    machinery_id: int
    bounding_box: List[float]
    class_name: str
    confidence: float


class RiskEvaluationResponse(BaseModel):
    safety_score: float
    risk_level: str
    primary_hazard: str
    contributing_factors: List[str]
    prescriptive_actions: List[str]
    temporal_confidence: str


class InspectionResponse(BaseModel):
    timestamp: float
    camera_id: str
    safety_score: float
    risk_level: str
    primary_hazard: str
    workers: List[WorkerItem]
    machinery: List[MachineryItem]
    proximity_alerts: List[str]
    explainability_factors: List[str]
    prescriptive_actions: List[str]
    incident_id: Optional[str]
    latency_ms: float


class InspectPathRequest(BaseModel):
    image_path: str = Field(..., description="Absolute path to an image file")
    camera_id: str = "CAM-01"


class IncidentUpdatePayload(BaseModel):
    action: str = Field(..., description="'ACKNOWLEDGE' or 'RESOLVE' or 'FALSE_POSITIVE'")
    responder_name: str = "Duty Safety Officer"
    notes: Optional[str] = "Handled and verified on site."
