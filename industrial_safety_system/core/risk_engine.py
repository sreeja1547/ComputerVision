"""Stage 5: Explainable Risk Engine Module.

Calculates:
- Quantitative Safety Score (0% - 100%)
- Industrial Safety Risk Level: CRITICAL, HIGH, MEDIUM, LOW (SAFE)
- Explainable AI (XAI) transparent attribution breakdown
- Actionable prescriptive remediation recommendations
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from industrial_safety_system.config.settings import settings
from industrial_safety_system.core.context_checker import (
    ContextCheckResult,
    VerifiedViolation,
)


@dataclass
class RiskDeductionItem:
    factor: str
    target: str
    penalty: float
    is_temporal_verified: bool
    explanation: str


@dataclass
class ExplainableRiskAssessment:
    safety_score: float  # 0.0 to 100.0
    risk_level: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    risk_color: str  # Hex color e.g. #FF2A2A
    primary_hazard: str
    contributing_factors: List[RiskDeductionItem]
    bullet_reasons: List[str]
    prescriptive_actions: List[str]
    verified_violations_count: int
    temporal_confidence: str  # e.g. "HIGH (Multi-frame verified)"
    timestamp: float


class ExplainableRiskEngine:
    """Computes transparent, auditable safety levels with explainable root causes."""

    def __init__(self):
        self.thresholds = settings.thresholds

    def determine_risk_level(self, score: float) -> Tuple[str, str]:
        """Maps quantitative score (0-100) to standardized OSHA-grade risk levels."""
        if score < 40.0:
            return "CRITICAL", "#FF2A2A"
        elif score < 70.0:
            return "HIGH", "#FF7A00"
        elif score < 85.0:
            return "MEDIUM", "#FFCC00"
        else:
            return "LOW", "#00E676"

    def assess(self, context_res: ContextCheckResult) -> ExplainableRiskAssessment:
        """Calculates safety score and generates transparent explainability attribution."""
        score = 100.0
        deductions: List[RiskDeductionItem] = []
        bullet_reasons: List[str] = []
        prescriptive_actions: List[str] = []

        # Process each verified violation
        for v in context_res.verified_violations:
            base_penalty = 0.0
            factor_label = ""

            if v.violation_type == "danger_zone_intrusion":
                base_penalty = self.thresholds.penalty_danger_zone_breach
                factor_label = "Exclusion Danger Zone Intrusion"
            elif v.violation_type == "machinery_proximity_hazard":
                base_penalty = self.thresholds.penalty_machinery_proximity
                factor_label = "Machinery Proximity Collision Risk"
            elif "missing_helmet" in v.violation_type:
                base_penalty = self.thresholds.penalty_missing_hardhat
                factor_label = "Missing Safety Hardhat"
            elif "missing_vest" in v.violation_type:
                base_penalty = self.thresholds.penalty_missing_vest
                factor_label = "Missing High-Vis Safety Vest"
            elif "missing_gloves" in v.violation_type:
                base_penalty = self.thresholds.penalty_missing_gloves
                factor_label = "Missing Protective Gloves"
            elif "missing_boots" in v.violation_type:
                base_penalty = self.thresholds.penalty_missing_boots
                factor_label = "Unapproved Footwear"
            else:
                base_penalty = 15.0
                factor_label = "General Safety Non-Compliance"

            # Apply persistence penalty if verified across multiple frames
            actual_penalty = base_penalty
            if v.is_temporal_verified and v.consecutive_frames >= 3:
                actual_penalty = round(base_penalty * self.thresholds.persistence_penalty_multiplier, 1)

            score -= actual_penalty

            verif_tag = f"[Verified over {v.consecutive_frames} frames]" if v.is_temporal_verified else "[Transient/Unconfirmed]"
            deduction_item = RiskDeductionItem(
                factor=factor_label,
                target=f"Worker #{v.worker_id}",
                penalty=actual_penalty,
                is_temporal_verified=v.is_temporal_verified,
                explanation=f"{v.details} {verif_tag} (-{actual_penalty}%)",
            )
            deductions.append(deduction_item)
            bullet_reasons.append(f"• {factor_label}: {v.details} (-{actual_penalty}%)")

            # Prescriptive recommendations
            if v.violation_type == "danger_zone_intrusion":
                prescriptive_actions.append(f"Alert supervisor: Evacuate Worker #{v.worker_id} from {v.zone_id or 'danger zone'} immediately.")
            elif v.violation_type == "machinery_proximity_hazard":
                prescriptive_actions.append("Issue proximity stop-signal to machine operator and notify pedestrian worker.")
            elif "missing_helmet" in v.violation_type:
                prescriptive_actions.append(f"Mandate Worker #{v.worker_id} to equip approved EN397/ANSI Z89.1 hardhat.")
            elif "missing_vest" in v.violation_type:
                prescriptive_actions.append(f"Dispatch Class-2/Class-3 high-vis vest to Worker #{v.worker_id}.")

        # Proximity alerts from context
        for pa in context_res.proximity_alerts:
            if pa.is_critical:
                prescriptive_actions.append(f"CRITICAL: Immediate 3-meter safety perimeter required for {pa.machinery_name}.")

        # Deduplicate prescriptive actions
        prescriptive_actions = list(dict.fromkeys(prescriptive_actions))

        # Clamp score 0 - 100
        score = max(0.0, min(100.0, round(score, 1)))

        risk_level, risk_color = self.determine_risk_level(score)

        if not bullet_reasons:
            bullet_reasons.append("• Full compliance: All personnel equipped with required PPE within authorized corridors.")
            prescriptive_actions.append("Continue routine automated surveillance.")
            primary_hazard = "None - Safe Operations"
        else:
            primary_hazard = deductions[0].factor if deductions else "General Non-Compliance"

        # Temporal confidence evaluation
        verified_count = sum(1 for d in deductions if d.is_temporal_verified)
        if verified_count > 0:
            temporal_confidence = f"HIGH ({verified_count} verified sustained hazards)"
        elif deductions:
            temporal_confidence = "MODERATE (Evaluating multi-frame persistence)"
        else:
            temporal_confidence = "HIGH (Clear baseline)"

        return ExplainableRiskAssessment(
            safety_score=score,
            risk_level=risk_level,
            risk_color=risk_color,
            primary_hazard=primary_hazard,
            contributing_factors=deductions,
            bullet_reasons=bullet_reasons,
            prescriptive_actions=prescriptive_actions,
            verified_violations_count=len(context_res.verified_violations),
            temporal_confidence=temporal_confidence,
            timestamp=context_res.timestamp,
        )
