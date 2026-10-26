"""Reusable UI components and dashboard styling for Industrial Safety Console."""

from __future__ import annotations
import streamlit as st


def apply_industrial_theme():
    """Injects custom high-tech dark industrial telemetry CSS."""
    st.markdown(
        """
        <style>
        /* Main background */
        .stApp {
            background-color: #0c1017;
            color: #e2e8f0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        /* Top Header */
        .main-header {
            background: linear-gradient(90deg, #131d2e 0%, #172439 100%);
            border-bottom: 2px solid #2563eb;
            padding: 18px 24px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }
        .main-title {
            font-size: 26px;
            font-weight: 800;
            color: #ffffff;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .main-sub {
            font-size: 13px;
            color: #38bdf8;
            font-weight: 600;
            margin-top: 5px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }

        /* Pipeline Flow Bar */
        .pipeline-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #111827;
            padding: 10px 16px;
            border-radius: 8px;
            border: 1px solid #1f2937;
            margin-bottom: 20px;
        }
        .pipeline-step {
            display: inline-flex;
            align-items: center;
            font-size: 11px;
            font-weight: 700;
            color: #94a3b8;
            padding: 4px 10px;
            border-radius: 4px;
            background: #1e293b;
        }
        .pipeline-step.active {
            background: #0284c7;
            color: #ffffff;
            box-shadow: 0 0 12px rgba(2, 132, 199, 0.5);
        }
        .pipeline-arrow {
            color: #475569;
            font-size: 14px;
        }

        /* KPI Card */
        .kpi-card {
            background: #151d2a;
            border: 1px solid #243044;
            border-radius: 8px;
            padding: 14px 18px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        }
        .kpi-title {
            font-size: 12px;
            color: #94a3b8;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .kpi-val {
            font-size: 28px;
            font-weight: 800;
            color: #ffffff;
            margin: 0;
        }

        /* Risk Badges */
        .risk-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 800;
            font-size: 14px;
            letter-spacing: 1px;
        }
        .risk-CRITICAL { background: #7f1d1d; color: #fca5a5; border: 1px solid #ef4444; }
        .risk-HIGH { background: #7c2d12; color: #fdba74; border: 1px solid #f97316; }
        .risk-MEDIUM { background: #713f12; color: #fde047; border: 1px solid #eab308; }
        .risk-LOW { background: #14532d; color: #86efac; border: 1px solid #22c55e; }

        /* XAI Explanation box */
        .xai-box {
            background: #121926;
            border-left: 4px solid #38bdf8;
            border-radius: 6px;
            padding: 12px 16px;
            margin: 10px 0;
            font-size: 13px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    """Renders the top branding header and pipeline sequence."""
    st.markdown(
        """
        <div class="main-header">
            <div class="main-title">AI-Based Industrial Inspection Hazard Detection & Safety-Level Assessment</div>
            <div class="main-sub">AUTOMATED HAZARD IDENTIFICATION &bull; VERIFIED RISK ANALYSIS &bull; SMART INCIDENT RESPONSE</div>
        </div>
        <div class="pipeline-bar">
            <span class="pipeline-step active">1. CCTV FEED</span>
            <span class="pipeline-arrow">&rarr;</span>
            <span class="pipeline-step active">2. PREPROCESS</span>
            <span class="pipeline-arrow">&rarr;</span>
            <span class="pipeline-step active">3. YOLO DETECTION</span>
            <span class="pipeline-arrow">&rarr;</span>
            <span class="pipeline-step active">4. CONTEXT CHECK</span>
            <span class="pipeline-arrow">&rarr;</span>
            <span class="pipeline-step active">5. RISK ENGINE</span>
            <span class="pipeline-arrow">&rarr;</span>
            <span class="pipeline-step active">6. INCIDENT ACTION</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
