"""Streamlit Industrial Mission Control & Safety Operations Console.

AI-Based Industrial Inspection Hazard Detection & Safety-Level Assessment
Automated Hazard Identification • Verified Risk Analysis • Smart Incident Response
"""

from __future__ import annotations
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Setup import path
APP_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = APP_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from industrial_safety_system.config.settings import (
    DEFAULT_ARCHIVE_IMAGES_DIR,
    DEFAULT_SAMPLE_VIDEO_PATH,
    settings,
)
from industrial_safety_system.core.cctv_feed import CCTVFeed, FrameData
from industrial_safety_system.core.pipeline import IndustrialSafetyPipeline, PipelineResult
from industrial_safety_system.ui.components import apply_industrial_theme, render_header
from industrial_safety_system.utils.audio_alert import generate_alarm_wav_bytes
from industrial_safety_system.utils.export_report import (
    export_batch_summary_html,
    export_incidents_to_csv,
)

# Page configuration
st.set_page_config(
    page_title="Industrial Safety Intelligence AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_industrial_theme()


@st.cache_resource
def load_pipeline(camera_id: str = "CAM-01") -> IndustrialSafetyPipeline:
    return IndustrialSafetyPipeline(camera_id=camera_id)


def main():
    render_header()
    pipeline = load_pipeline()

    # Sidebar Navigation & Settings
    st.sidebar.markdown("### 🎛️ Operation Controls")
    mode = st.sidebar.radio(
        "Select Operation Station:",
        [
            "🎥 Live CCTV Inspection Station",
            "🗂️ Archive Dataset Inspector (2,102 Images)",
            "🛡️ Zone & Geofencing Manager",
            "🚨 Incident Response & Resolution Center",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Pipeline Thresholds")
    conf_worker = st.sidebar.slider(
        "Worker Detection Conf", 0.15, 0.90, settings.thresholds.person_confidence, 0.05
    )
    conf_ppe = st.sidebar.slider(
        "PPE Inspection Conf", 0.10, 0.90, settings.thresholds.ppe_confidence, 0.05
    )
    temporal_k = st.sidebar.slider(
        "Temporal Verification Frames", 1, 10, settings.thresholds.temporal_verification_frames, 1
    )
    audio_alert_enabled = st.sidebar.checkbox("🔊 Audio Alarm on Critical Risk", value=True)

    # Sync settings
    settings.thresholds.person_confidence = conf_worker
    settings.thresholds.ppe_confidence = conf_ppe
    settings.thresholds.temporal_verification_frames = temporal_k

    # =========================================================================
    # MODE 1: LIVE CCTV INSPECTION STATION
    # =========================================================================
    if mode == "🎥 Live CCTV Inspection Station":
        st.markdown("#### 🎥 Real-Time Industrial CCTV Stream & AI Telemetry")

        col_src, col_cam = st.columns([2, 1])
        with col_src:
            source_choice = st.selectbox(
                "Select CCTV Feed Source:",
                [
                    "Recorded Industrial Video (industrial_cctv_sample.mp4)",
                    "Live Webcam / Camera 0",
                    "Upload Custom Inspection Video",
                ],
            )
        with col_cam:
            cam_choice = st.selectbox("Assign Camera Profile:", ["CAM-01", "CAM-02", "CAM-03"])
            pipeline.set_camera(cam_choice)

        # Video source resolution
        video_source = None
        if "Recorded Industrial Video" in source_choice:
            video_source = str(DEFAULT_SAMPLE_VIDEO_PATH)
        elif "Live Webcam" in source_choice:
            video_source = 0
        else:
            uploaded_vid = st.file_uploader("Upload MP4/AVI Video", type=["mp4", "avi", "mov"])
            if uploaded_vid:
                tmp_path = WORKSPACE_DIR / "data" / f"temp_upload_{uploaded_vid.name}"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_vid.read())
                video_source = str(tmp_path)

        if video_source is None or (isinstance(video_source, str) and not os.path.exists(video_source)):
            st.warning("Please supply or verify the video source.")
            return

        # Setup Feed in Session State
        if "cctv_feed" not in st.session_state or st.session_state.get("last_video_source") != video_source:
            st.session_state["cctv_feed"] = CCTVFeed(video_source, loop_video=True)
            st.session_state["last_video_source"] = video_source

        feed: CCTVFeed = st.session_state["cctv_feed"]

        # Playback Controls
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 4])
        with col_ctrl1:
            run_live = st.checkbox("▶️ Run Continuous Inspection", value=False)
        with col_ctrl2:
            step_frame = st.button("⏭️ Next Frame", use_container_width=True)

        # Display Frame Container
        video_placeholder = st.empty()
        kpi_placeholder = st.empty()
        telemetry_placeholder = st.empty()

        # Frame Processing Step
        if run_live or step_frame:
            frame_data = feed.read_frame()
            if frame_data:
                res: PipelineResult = pipeline.process_frame(
                    raw_bgr=frame_data.image_bgr,
                    timestamp=frame_data.timestamp,
                    fps=frame_data.fps,
                )

                # 1. Video / HUD Display
                with video_placeholder.container():
                    col_view1, col_view2 = st.columns(2)
                    with col_view1:
                        st.image(
                            res.annotated_rgb,
                            caption="Live AI Inspection HUD Overlays (Zones, Workers, PPE, Proximity)",
                            use_container_width=True,
                        )
                    with col_view2:
                        st.image(
                            cv2.cvtColor(res.preprocess.clean_bgr, cv2.COLOR_BGR2RGB),
                            caption=f"Clean Preprocessed Feed | Lighting: {res.preprocess.lighting_condition} | Motion Energy: {res.preprocess.motion_energy:.1f}%",
                            use_container_width=True,
                        )

                # 2. KPI Banner
                with kpi_placeholder.container():
                    k1, k2, k3, k4, k5 = st.columns(5)
                    with k1:
                        st.markdown(
                            f"""<div class="kpi-card"><div class="kpi-title">Safety Score</div>
                            <div class="kpi-val" style="color: {res.risk.risk_color};">{res.risk.safety_score:.0f}%</div></div>""",
                            unsafe_allow_html=True,
                        )
                    with k2:
                        st.markdown(
                            f"""<div class="kpi-card"><div class="kpi-title">Risk Level</div>
                            <div class="risk-badge risk-{res.risk.risk_level}">{res.risk.risk_level}</div></div>""",
                            unsafe_allow_html=True,
                        )
                    with k3:
                        st.markdown(
                            f"""<div class="kpi-card"><div class="kpi-title">Active Workers</div>
                            <div class="kpi-val">{len(res.context.tracked_workers)}</div></div>""",
                            unsafe_allow_html=True,
                        )
                    with k4:
                        st.markdown(
                            f"""<div class="kpi-card"><div class="kpi-title">Machinery Detected</div>
                            <div class="kpi-val">{len(res.detection.machinery)}</div></div>""",
                            unsafe_allow_html=True,
                        )
                    with k5:
                        st.markdown(
                            f"""<div class="kpi-card"><div class="kpi-title">Latency</div>
                            <div class="kpi-val" style="font-size: 22px;">{res.latency.total_ms:.0f} ms</div></div>""",
                            unsafe_allow_html=True,
                        )

                # 3. Explainability Telemetry Panel
                with telemetry_placeholder.container():
                    st.markdown("#### 🔍 Explainable AI (XAI) Risk Assessment Breakdown")
                    col_xai1, col_xai2 = st.columns(2)

                    with col_xai1:
                        st.markdown(f"**Primary Hazard:** `{res.risk.primary_hazard}`")
                        st.markdown(f"**Temporal Verification:** `{res.risk.temporal_confidence}`")
                        st.markdown("**Transparent Factor Attributions:**")
                        for b in res.risk.bullet_reasons:
                            st.markdown(f"<div class='xai-box'>{b}</div>", unsafe_allow_html=True)

                    with col_xai2:
                        st.markdown("**Prescriptive Response Actions:**")
                        for act in res.risk.prescriptive_actions:
                            st.info(f"⚡ {act}")

                        if res.incident:
                            st.error(
                                f"🚨 **AUTOMATED INCIDENT CREATED:** `{res.incident.incident_id}` | Priority: {res.incident.priority} | Status: `{res.incident.status}`"
                            )
                            if audio_alert_enabled and res.risk.risk_level == "CRITICAL":
                                st.audio(generate_alarm_wav_bytes(frequency=920, duration_seconds=0.6), format="audio/wav", autoplay=True)

        else:
            # Idle preview
            first_frame = feed.read_frame()
            if first_frame:
                st.image(first_frame.image_rgb, caption="CCTV Feed Ready. Click '▶️ Run Continuous Inspection' or '⏭️ Next Frame' to analyze.", use_container_width=True)

    # =========================================================================
    # MODE 2: ARCHIVE DATASET INSPECTOR (2,102 IMAGES)
    # =========================================================================
    elif mode == "🗂️ Archive Dataset Inspector (2,102 Images)":
        st.markdown("#### 🗂️ Industrial Archive Dataset Explorer & Batch Inspector")
        archive_path = Path(DEFAULT_ARCHIVE_IMAGES_DIR)

        if not archive_path.exists():
            st.error(f"Archive directory not found: {archive_path}")
            return

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        all_images = sorted([f for f in archive_path.iterdir() if f.suffix.lower() in valid_exts])
        total_images = len(all_images)

        st.success(f"Connected to Industrial Dataset: **{total_images:,} Real-World Inspection Images** at `{archive_path}`")

        tab_single, tab_batch = st.tabs(["🔍 Single Image Deep Inspection", "⚡ 1-Click Batch Safety Audit"])

        # TAB 1: Single Image Deep Inspection
        with tab_single:
            col_search, col_page = st.columns([3, 1])
            with col_search:
                search_query = st.text_input("Filter images by filename keyword:", placeholder="e.g. construction, worker, grinder, farm, building")
            with col_page:
                items_per_page = 12
                filtered_images = [img for img in all_images if search_query.lower() in img.name.lower()] if search_query else all_images
                total_pages = max(1, math.ceil(len(filtered_images) / items_per_page))
                page_num = st.number_input("Page:", min_value=1, max_value=total_pages, value=1)

            st.write(f"Showing page {page_num} of {total_pages} ({len(filtered_images)} matches)")

            start_idx = (page_num - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(filtered_images))
            page_slice = filtered_images[start_idx:end_idx]

            # Render 4 columns grid of thumbnails
            grid_cols = st.columns(4)
            selected_img_path = None

            for i, img_p in enumerate(page_slice):
                col = grid_cols[i % 4]
                with col:
                    try:
                        im = Image.open(img_p)
                        im.thumbnail((240, 160))
                        st.image(im, use_container_width=True)
                        if st.button(f"Inspect #{start_idx + i + 1}", key=f"btn_img_{start_idx + i}", use_container_width=True):
                            selected_img_path = img_p
                    except Exception as e:
                        st.caption(f"Error loading: {img_p.name}")

            if "selected_archive_img" not in st.session_state and selected_img_path:
                st.session_state["selected_archive_img"] = selected_img_path
            elif selected_img_path:
                st.session_state["selected_archive_img"] = selected_img_path

            # Deep Inspection of selected image
            curr_selected = st.session_state.get("selected_archive_img")
            if curr_selected and Path(curr_selected).exists():
                st.markdown("---")
                st.markdown(f"### 🔬 Deep AI Inspection: `{Path(curr_selected).name}`")

                raw_bgr = cv2.imread(str(curr_selected))
                if raw_bgr is not None:
                    # Run full pipeline
                    res = pipeline.process_frame(raw_bgr, timestamp=time.time())

                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.image(res.annotated_rgb, caption="Annotated Safety Analysis", use_container_width=True)
                    with col_res2:
                        st.image(res.preprocess.clean_rgb, caption=f"Preprocessed View | Brightness: {res.preprocess.brightness:.1f}", use_container_width=True)

                    # Telemetry Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Safety Score", f"{res.risk.safety_score:.0f}%")
                    m2.metric("Risk Level", res.risk.risk_level)
                    m3.metric("Workers Detected", len(res.context.tracked_workers))
                    m4.metric("Machinery Detected", len(res.detection.machinery))

                    st.markdown("#### 📋 Explainable Root-Cause Audit")
                    for b in res.risk.bullet_reasons:
                        st.markdown(f"<div class='xai-box'>{b}</div>", unsafe_allow_html=True)

                    st.markdown("#### ⚡ Recommended Remediation")
                    for act in res.risk.prescriptive_actions:
                        st.info(act)

        # TAB 2: Batch Safety Audit
        with tab_batch:
            st.markdown("#### ⚡ Automated High-Throughput Batch Inspection")
            st.write("Scan a subset of the 2,102 dataset images to generate an aggregate plant safety compliance audit report.")

            batch_size = st.slider("Select Batch Size (number of images):", 5, 100, 20, 5)
            if st.button("🚀 Start Automated Batch Safety Audit", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                batch_records = []
                sample_slice = all_images[:batch_size]

                for idx, img_p in enumerate(sample_slice):
                    status_text.text(f"Inspecting [{idx + 1}/{batch_size}]: {img_p.name}...")
                    raw_bgr = cv2.imread(str(img_p))
                    if raw_bgr is None:
                        continue

                    res = pipeline.process_frame(raw_bgr, timestamp=time.time())
                    batch_records.append(
                        {
                            "source_name": img_p.name,
                            "worker_count": len(res.context.tracked_workers),
                            "safety_score": res.risk.safety_score,
                            "risk_level": res.risk.risk_level,
                            "primary_hazard": res.risk.primary_hazard,
                            "violations": [d.factor for d in res.risk.contributing_factors],
                        }
                    )
                    progress_bar.progress((idx + 1) / batch_size)

                status_text.text("✅ Batch Inspection Complete!")
                st.session_state["last_batch_records"] = batch_records

            if "last_batch_records" in st.session_state:
                records = st.session_state["last_batch_records"]
                df = pd.DataFrame(records)

                st.markdown("### 📊 Batch Inspection Analytics & Audit Summary")
                b_kpi1, b_kpi2, b_kpi3, b_kpi4 = st.columns(4)
                b_kpi1.metric("Images Processed", len(df))
                b_kpi2.metric("Average Safety Score", f"{df['safety_score'].mean():.1f}%")
                b_kpi3.metric("Critical Hazards Found", int((df['risk_level'] == 'CRITICAL').sum()))
                b_kpi4.metric("Fully Compliant Frames", int((df['risk_level'] == 'LOW').sum()))

                # Chart
                st.bar_chart(df["risk_level"].value_counts())

                # Table
                st.dataframe(df[["source_name", "worker_count", "safety_score", "risk_level", "primary_hazard"]], use_container_width=True)

                # Export buttons
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    csv_data = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Batch Audit (CSV)", data=csv_data, file_name="batch_safety_audit.csv", mime="text/csv")
                with col_exp2:
                    html_report = export_batch_summary_html(records)
                    st.download_button("📑 Download Executive Audit Report (HTML)", data=html_report, file_name="safety_audit_report.html", mime="text/html")

    # =========================================================================
    # MODE 3: ZONE & GEOFENCING MANAGER
    # =========================================================================
    elif mode == "🛡️ Zone & Geofencing Manager":
        st.markdown("#### 🛡️ Polygonal Geofencing & Exclusion Zone Configuration")

        active_cam = st.selectbox("Select Industrial Camera Profile:", ["CAM-01", "CAM-02", "CAM-03"])
        pipeline.set_camera(active_cam)
        zones = pipeline.context_checker.get_current_zones()

        st.write(f"Configured Zones for **{active_cam}** ({len(zones)} active zones):")

        for z in zones:
            with st.expander(f"📍 [{z.zone_type}] {z.name} ({z.zone_id})", expanded=True):
                col_z1, col_z2 = st.columns([2, 1])
                with col_z1:
                    st.markdown(f"**Description:** {z.description}")
                    st.markdown(f"**Required PPE:** `{', '.join(z.required_ppe) if z.required_ppe else 'None'}`")
                    st.markdown(f"**Prohibited to General Personnel:** `{'YES (Exclusion Zone)' if z.prohibited_personnel else 'NO'}`")
                with col_z2:
                    st.color_picker("Zone Color", z.color_hex, key=f"color_{z.zone_id}", disabled=True)
                    st.write(f"Polygon Vertices: {len(z.points)} points")

    # =========================================================================
    # MODE 4: INCIDENT ACTION & RESOLUTION CENTER
    # =========================================================================
    elif mode == "🚨 Incident Response & Resolution Center":
        st.markdown("#### 🚨 Smart Incident Management & Automated Resolution Workflow")

        inc_mgr = pipeline.incident_manager
        all_incidents = inc_mgr.list_incidents()

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            status_f = st.selectbox("Filter Status:", ["All", "OPEN", "ACKNOWLEDGED", "RESOLVED", "FALSE_POSITIVE"])
        with col_f2:
            risk_f = st.selectbox("Filter Risk Level:", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"])

        filt_stat = None if status_f == "All" else status_f
        filt_risk = None if risk_f == "All" else risk_f
        filtered_inc = inc_mgr.list_incidents(status_filter=filt_stat, risk_filter=filt_risk)

        st.write(f"Showing **{len(filtered_inc)}** incidents (Total logged: {len(all_incidents)})")

        if not filtered_inc:
            st.info("No incident tickets matching the selected filter criteria.")
        else:
            for inc in filtered_inc:
                with st.expander(
                    f"🚨 [{inc.risk_level}] {inc.incident_id} - {inc.primary_hazard} | Status: {inc.status} ({inc.formatted_time})",
                    expanded=(inc.status == "OPEN"),
                ):
                    col_det1, col_det2 = st.columns([3, 2])
                    with col_det1:
                        st.markdown(f"**Camera ID:** `{inc.camera_id}` | **Assigned To:** `{inc.assigned_to}`")
                        st.markdown(f"**Safety Score:** `{inc.safety_score:.0f}%` | **Priority:** `P{inc.priority}`")
                        st.markdown(f"**Recommended Action:**")
                        st.warning(inc.recommended_action)

                        st.markdown("**Violations Evidenced:**")
                        for v in inc.violations:
                            st.write(f"- {v}")

                        # Action form
                        st.markdown("---")
                        st.markdown("##### ✍️ Action & Resolution Workflow")
                        act_col1, act_col2 = st.columns(2)
                        with act_col1:
                            if inc.status == "OPEN":
                                if st.button(f"✅ Acknowledge Incident", key=f"ack_{inc.incident_id}"):
                                    inc_mgr.acknowledge_incident(inc.incident_id, "Duty Safety Officer")
                                    st.rerun()

                        notes = st.text_input("Resolution / Corrective Action Notes:", key=f"notes_{inc.incident_id}", value=inc.resolution_notes or "")
                        with act_col2:
                            if inc.status in ["OPEN", "ACKNOWLEDGED"]:
                                if st.button("🔒 Mark as Resolved", key=f"res_{inc.incident_id}", type="primary"):
                                    inc_mgr.resolve_incident(inc.incident_id, notes or "Verified resolved on plant floor.", "Duty Safety Officer")
                                    st.rerun()
                                if st.button("❌ Flag as False Positive", key=f"fp_{inc.incident_id}"):
                                    inc_mgr.resolve_incident(inc.incident_id, notes or "Manual review flagged false positive.", "Duty Safety Officer", mark_as_false_positive=True)
                                    st.rerun()

                    with col_det2:
                        if inc.evidence_image_path and Path(inc.evidence_image_path).exists():
                            st.image(inc.evidence_image_path, caption=f"Evidentiary Snapshot: {inc.incident_id}", use_container_width=True)
                        else:
                            st.caption("No snapshot image attached.")

            st.markdown("---")
            csv_str = export_incidents_to_csv(all_incidents)
            st.download_button(
                "📥 Download Incident Log (CSV)",
                data=csv_str.encode("utf-8"),
                file_name="industrial_incidents_audit_log.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
