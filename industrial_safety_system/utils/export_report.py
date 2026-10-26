"""Export and reporting utility for industrial safety audits."""

from __future__ import annotations
import csv
import io
import json
from dataclasses import asdict
from datetime import datetime
from typing import List, Dict
from industrial_safety_system.core.incident_action import IncidentTicket


def export_incidents_to_csv(incidents: List[IncidentTicket]) -> str:
    """Exports a list of incident tickets to a standard CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Incident ID",
        "Timestamp",
        "Camera ID",
        "Risk Level",
        "Safety Score (%)",
        "Primary Hazard",
        "Status",
        "Priority",
        "Assigned To",
        "Recommended Action",
        "Resolution Notes",
        "Violations List",
    ])

    for inc in incidents:
        writer.writerow([
            inc.incident_id,
            inc.formatted_time,
            inc.camera_id,
            inc.risk_level,
            inc.safety_score,
            inc.primary_hazard,
            inc.status,
            inc.priority,
            inc.assigned_to,
            inc.recommended_action,
            inc.resolution_notes,
            " | ".join(inc.violations),
        ])

    return output.getvalue()


def export_batch_summary_html(batch_results: List[Dict]) -> str:
    """Generates an executive HTML audit report for a batch inspection run."""
    total = len(batch_results)
    critical_cnt = sum(1 for r in batch_results if r.get("risk_level") == "CRITICAL")
    high_cnt = sum(1 for r in batch_results if r.get("risk_level") == "HIGH")
    medium_cnt = sum(1 for r in batch_results if r.get("risk_level") == "MEDIUM")
    low_cnt = sum(1 for r in batch_results if r.get("risk_level") == "LOW")
    avg_score = sum(r.get("safety_score", 100) for r in batch_results) / total if total > 0 else 100.0

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>Industrial Safety Inspection Audit Report</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 30px; background: #0f141c; color: #e2e8f0; }}
  h1 {{ color: #38bdf8; }}
  .kpi-container {{ display: flex; gap: 20px; margin: 25px 0; }}
  .kpi {{ background: #1e293b; padding: 20px; border-radius: 8px; flex: 1; text-align: center; border-left: 4px solid #38bdf8; }}
  .kpi h2 {{ margin: 0; font-size: 32px; }}
  .kpi p {{ margin: 5px 0 0; color: #94a3b8; font-size: 14px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 25px; background: #1e293b; border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; }}
  th {{ background: #0b1120; color: #94a3b8; text-transform: uppercase; font-size: 12px; }}
  .badge {{ padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
  .badge-CRITICAL {{ background: #7f1d1d; color: #fca5a5; }}
  .badge-HIGH {{ background: #7c2d12; color: #fdba74; }}
  .badge-MEDIUM {{ background: #713f12; color: #fde047; }}
  .badge-LOW {{ background: #14532d; color: #86efac; }}
</style>
</head>
<body>
  <h1>Industrial Inspection & Safety-Level Assessment Report</h1>
  <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | End-to-End Safety Intelligence Pipeline</p>

  <div class="kpi-container">
    <div class="kpi"><h2>{total}</h2><p>Images Analyzed</p></div>
    <div class="kpi"><h2>{avg_score:.1f}%</h2><p>Plant Safety Score</p></div>
    <div class="kpi" style="border-color: #ef4444;"><h2>{critical_cnt}</h2><p>Critical Incidents</p></div>
    <div class="kpi" style="border-color: #f97316;"><h2>{high_cnt}</h2><p>High Hazard Items</p></div>
    <div class="kpi" style="border-color: #22c55e;"><h2>{low_cnt}</h2><p>Compliant Frames</p></div>
  </div>

  <table>
    <thead>
      <tr>
        <th>Image Source</th>
        <th>Workers</th>
        <th>Safety Score</th>
        <th>Risk Level</th>
        <th>Primary Hazard</th>
        <th>Contributing Factors</th>
      </tr>
    </thead>
    <tbody>
"""

    for r in batch_results:
        lvl = r.get("risk_level", "LOW")
        score = r.get("safety_score", 100.0)
        factors = " | ".join(r.get("violations", [])) or "Full Compliance"
        html += f"""      <tr>
        <td><strong>{r.get('source_name', 'N/A')}</strong></td>
        <td>{r.get('worker_count', 0)}</td>
        <td>{score:.0f}%</td>
        <td><span class="badge badge-{lvl}">{lvl}</span></td>
        <td>{r.get('primary_hazard', 'None')}</td>
        <td><small>{factors}</small></td>
      </tr>\n"""

    html += """    </tbody>
  </table>
</body>
</html>"""
    return html
