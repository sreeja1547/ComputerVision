"""Command-line Batch Inspector for Industrial Archive Images."""

import argparse
import sys
import time
from pathlib import Path
import cv2
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from industrial_safety_system.config.settings import DEFAULT_ARCHIVE_IMAGES_DIR, DATA_DIR
from industrial_safety_system.core.pipeline import IndustrialSafetyPipeline
from industrial_safety_system.utils.export_report import (
    export_batch_summary_html,
    export_incidents_to_csv,
)


def run_cli_scanner(source_dir: Path, limit: int = 25, camera_id: str = "CAM-01"):
    print("=" * 75)
    print("AI-BASED INDUSTRIAL INSPECTION HAZARD DETECTION & SAFETY ASSESSMENT")
    print("Batch CLI Inspector")
    print("=" * 75)

    if not source_dir.exists():
        print(f"Error: Directory not found: {source_dir}")
        return

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([f for f in source_dir.iterdir() if f.suffix.lower() in valid_exts])
    print(f"Discovered {len(images):,} images in {source_dir}")
    sample_images = images[:limit] if limit > 0 else images
    print(f"Running 6-Stage End-to-End Safety Pipeline on {len(sample_images)} images...\n")

    pipeline = IndustrialSafetyPipeline(camera_id=camera_id)
    records = []

    for i, img_path in enumerate(sample_images, 1):
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            continue

        res = pipeline.process_frame(raw_bgr, timestamp=time.time())
        risk_lvl = res.risk.risk_level
        score = res.risk.safety_score
        w_cnt = len(res.context.tracked_workers)
        m_cnt = len(res.detection.machinery)

        status_tag = f"[{risk_lvl:8s}]"
        print(f"[{i:03d}/{len(sample_images):03d}] {status_tag} Score: {score:5.1f}% | Workers: {w_cnt} | Machine: {m_cnt} | {img_path.name[:35]:35s} | Hazard: {res.risk.primary_hazard}")

        records.append({
            "source_name": img_path.name,
            "worker_count": w_cnt,
            "safety_score": score,
            "risk_level": risk_lvl,
            "primary_hazard": res.risk.primary_hazard,
            "violations": [d.factor for d in res.risk.contributing_factors],
        })

    print("\n" + "=" * 75)
    print("BATCH INSPECTION COMPLETED")
    print("=" * 75)
    df = pd.DataFrame(records)
    print(f"Total Processed: {len(df)}")
    print(f"Average Safety Score: {df['safety_score'].mean():.1f}%")
    print("Risk Level Breakdown:")
    for lvl, cnt in df["risk_level"].value_counts().items():
        print(f" - {lvl:8s}: {cnt} frames ({cnt / len(df) * 100:.1f}%)")

    # Export reports
    out_csv = DATA_DIR / "batch_audit_cli_results.csv"
    out_html = DATA_DIR / "batch_audit_cli_report.html"
    df.to_csv(out_csv, index=False)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(export_batch_summary_html(records))

    print(f"\nSaved CSV report to:  {out_csv}")
    print(f"Saved HTML report to: {out_html}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Industrial Safety Batch Inspector")
    parser.add_argument("--source", type=str, default=str(DEFAULT_ARCHIVE_IMAGES_DIR), help="Path to archive image directory")
    parser.add_argument("--limit", type=int, default=15, help="Number of images to process (0 for all)")
    parser.add_argument("--camera", type=str, default="CAM-01", help="Camera zone profile")
    args = parser.parse_args()

    run_cli_scanner(Path(args.source), limit=args.limit, camera_id=args.camera)
