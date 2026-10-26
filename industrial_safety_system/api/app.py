"""FastAPI Application for Industrial Inspection Hazard Detection & Safety-Level Assessment.

Provides complete REST endpoints + serves the rich modern Frontend web application.
"""

from __future__ import annotations
import base64
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional
import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
import numpy as np

# Ensure root import path
APP_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = APP_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from industrial_safety_system.config.settings import (
    DEFAULT_ARCHIVE_IMAGES_DIR,
    DEFAULT_SAMPLE_VIDEO_PATH,
    DATA_DIR,
    EVIDENCE_DIR,
    settings,
)
from industrial_safety_system.core.pipeline import IndustrialSafetyPipeline, PipelineResult
from industrial_safety_system.utils.export_report import (
    export_incidents_to_csv,
    export_batch_summary_html,
)
from industrial_safety_system.api.schemas import (
    HealthResponse,
    InspectionResponse,
    InspectPathRequest,
    IncidentUpdatePayload,
    WorkerItem,
    MachineryItem,
)

app = FastAPI(
    title=settings.app_title,
    description="Automated Hazard Identification • Verified Risk Analysis • Smart Incident Response REST API",
    version=settings.version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
pipeline = IndustrialSafetyPipeline()
FRONTEND_DIR = APP_DIR / "frontend"


@app.get("/api/health", response_model=HealthResponse)
def get_health():
    return HealthResponse(
        status="HEALTHY",
        version=settings.version,
        models_loaded={
            "yolov8_base": pipeline.detector.base_model is not None,
            "yolov8_ppe": pipeline.detector.ppe_model is not None,
        },
        timestamp=time.time(),
    )


@app.get("/api/zones")
def get_zones():
    """Returns configured camera profiles and polygon geofences."""
    zones_p = APP_DIR / "config" / "zones_config.json"
    if zones_p.exists():
        with open(zones_p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"cameras": {}}


@app.get("/api/archive/images")
def list_archive_images(
    page: int = Query(1, ge=1),
    page_size: int = Query(16, ge=1, le=100),
    query: Optional[str] = Query(None),
):
    """Lists images from the 2,102 archive dataset with pagination and search."""
    archive_dir = Path(DEFAULT_ARCHIVE_IMAGES_DIR)
    if not archive_dir.exists():
        return {"total": 0, "page": page, "page_size": page_size, "images": []}

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [f.name for f in archive_dir.iterdir() if f.suffix.lower() in valid_exts]
    files.sort()

    if query:
        q = query.lower()
        files = [f for f in files if q in f.lower()]

    total = len(files)
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total)
    slice_files = files[start_idx:end_idx]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, math.ceil(total / page_size)),
        "images": slice_files,
    }


@app.get("/api/archive/image/{filename}")
def get_archive_image_file(filename: str):
    """Serves a specific archive image file for thumbnail/full preview."""
    p = Path(DEFAULT_ARCHIVE_IMAGES_DIR) / filename
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Archive image not found")
    return FileResponse(str(p))


@app.get("/api/evidence/{filename}")
def get_evidence_image_file(filename: str):
    """Serves an annotated evidentiary snapshot image."""
    p = EVIDENCE_DIR / filename
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Evidence file not found")
    return FileResponse(str(p))


@app.get("/api/sample_video")
def get_sample_video():
    """Serves the bundled industrial CCTV sample video."""
    p = Path(DEFAULT_SAMPLE_VIDEO_PATH)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Sample video not found")
    return FileResponse(str(p), media_type="video/mp4")


def _pipeline_result_to_response(res: PipelineResult, camera_id: str, include_annotated_base64: bool = True) -> dict:
    workers_dto = [
        {
            "worker_id": w.worker_id or 1,
            "bounding_box": w.box.to_int_list(),
            "confidence": round(w.confidence, 3),
            "has_helmet": w.ppe.has_helmet,
            "has_vest": w.ppe.has_vest,
            "has_gloves": w.ppe.has_gloves,
            "has_boots": w.ppe.has_boots,
            "current_zone": w.current_zone,
            "violations": w.violations,
        }
        for w in res.context.tracked_workers
    ]

    machinery_dto = [
        {
            "machinery_id": m.machinery_id,
            "bounding_box": m.box.to_int_list(),
            "class_name": m.class_name,
            "confidence": round(m.confidence, 3),
        }
        for m in res.detection.machinery
    ]

    annotated_b64 = None
    if include_annotated_base64:
        _, buffer = cv2.imencode(".jpg", res.annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    return {
        "timestamp": res.timestamp,
        "camera_id": camera_id,
        "safety_score": res.risk.safety_score,
        "risk_level": res.risk.risk_level,
        "risk_color": res.risk.risk_color,
        "primary_hazard": res.risk.primary_hazard,
        "workers": workers_dto,
        "machinery": machinery_dto,
        "proximity_alerts": [pa.description for pa in res.context.proximity_alerts],
        "explainability_factors": res.risk.bullet_reasons,
        "prescriptive_actions": res.risk.prescriptive_actions,
        "temporal_confidence": res.risk.temporal_confidence,
        "incident_id": res.incident.incident_id if res.incident else None,
        "annotated_image": annotated_b64,
        "latency_ms": res.latency.total_ms,
        "latency_breakdown": {
            "preprocess_ms": res.latency.preprocess_ms,
            "yolo_ms": res.latency.yolo_ms,
            "context_ms": res.latency.context_ms,
            "risk_ms": res.latency.risk_ms,
            "visualizer_ms": res.latency.visualizer_ms,
            "total_ms": res.latency.total_ms,
        },
    }


@app.post("/api/inspect/image")
async def inspect_uploaded_image(
    file: UploadFile = File(...),
    camera_id: str = "CAM-01",
):
    """Processes an uploaded image file through all 6 pipeline stages."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file format")

        pipeline.set_camera(camera_id)
        res = pipeline.process_frame(img_bgr, timestamp=time.time())
        return _pipeline_result_to_response(res, camera_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/inspect/path")
def inspect_image_by_path(payload: InspectPathRequest):
    """Processes an image from the local filesystem path."""
    p = Path(payload.image_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Image file not found: {payload.image_path}")

    img_bgr = cv2.imread(str(p))
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Failed to decode image from path")

    pipeline.set_camera(payload.camera_id)
    res = pipeline.process_frame(img_bgr, timestamp=time.time())
    return _pipeline_result_to_response(res, payload.camera_id)


@app.get("/api/incidents")
def get_incidents(
    status: Optional[str] = None,
    risk_level: Optional[str] = None,
):
    """Returns all incident tickets filtered by status and risk level."""
    return pipeline.incident_manager.list_incidents(
        status_filter=status,
        risk_filter=risk_level,
    )


@app.patch("/api/incidents/{incident_id}")
def update_incident(incident_id: str, payload: IncidentUpdatePayload):
    """Updates incident status (ACKNOWLEDGE, RESOLVE, FALSE_POSITIVE)."""
    act = payload.action.upper()
    if act == "ACKNOWLEDGE":
        success = pipeline.incident_manager.acknowledge_incident(incident_id, payload.responder_name)
    elif act in ["RESOLVE", "FALSE_POSITIVE"]:
        is_fp = (act == "FALSE_POSITIVE")
        success = pipeline.incident_manager.resolve_incident(
            incident_id=incident_id,
            resolution_notes=payload.notes or "Resolved via API",
            responder_name=payload.responder_name,
            mark_as_false_positive=is_fp,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Invalid action: {payload.action}")

    if not success:
        raise HTTPException(status_code=404, detail=f"Incident not found: {incident_id}")

    return {"status": "SUCCESS", "incident_id": incident_id, "action": act}


@app.get("/api/incidents/export")
def export_incidents_csv():
    """Exports all incidents as a CSV file."""
    incidents = pipeline.incident_manager.list_incidents()
    csv_str = export_incidents_to_csv(incidents)
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=industrial_incidents_audit.csv"},
    )


@app.post("/api/batch_audit")
def run_batch_audit_api(
    sample_size: int = Query(15, ge=1, le=100),
    camera_id: str = "CAM-01",
):
    """Runs a batch audit on sample archive images and returns aggregated statistics."""
    archive_dir = Path(DEFAULT_ARCHIVE_IMAGES_DIR)
    if not archive_dir.exists():
        raise HTTPException(status_code=404, detail="Archive directory not found")

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted([f for f in archive_dir.iterdir() if f.suffix.lower() in valid_exts])[:sample_size]

    pipeline.set_camera(camera_id)
    records = []
    total_score = 0.0
    critical_cnt = 0
    high_cnt = 0
    medium_cnt = 0
    low_cnt = 0

    for img_p in files:
        raw_bgr = cv2.imread(str(img_p))
        if raw_bgr is None:
            continue
        res = pipeline.process_frame(raw_bgr, timestamp=time.time())
        score = res.risk.safety_score
        total_score += score
        lvl = res.risk.risk_level
        if lvl == "CRITICAL":
            critical_cnt += 1
        elif lvl == "HIGH":
            high_cnt += 1
        elif lvl == "MEDIUM":
            medium_cnt += 1
        else:
            low_cnt += 1

        records.append({
            "source_name": img_p.name,
            "worker_count": len(res.context.tracked_workers),
            "safety_score": score,
            "risk_level": lvl,
            "primary_hazard": res.risk.primary_hazard,
            "violations": [d.factor for d in res.risk.contributing_factors],
        })

    avg_score = round(total_score / len(records), 1) if records else 100.0

    # Save reports
    csv_p = DATA_DIR / "batch_audit_results.csv"
    html_p = DATA_DIR / "batch_audit_report.html"
    with open(html_p, "w", encoding="utf-8") as f:
        f.write(export_batch_summary_html(records))

    return {
        "total_scanned": len(records),
        "avg_safety_score": avg_score,
        "critical_count": critical_cnt,
        "high_count": high_cnt,
        "medium_count": medium_cnt,
        "low_count": low_cnt,
        "records": records,
    }


# Mount Static Frontend
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def serve_frontend_root():
        index_p = FRONTEND_DIR / "index.html"
        if index_p.exists():
            with open(index_p, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse("<h1>Frontend loading...</h1>")
