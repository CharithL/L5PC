"""
monitoring_protocol.py

Phase 7-1: Clinical Monitoring Protocol Specification

Generates the monitoring protocol for DESCARTES Phase A clinical trials
as both JSON (machine-readable) and Markdown (human-readable).

Protocol components:
    1. INSTRUMENTS: MCQ, RDEES, EMA, DES
    2. CONTROL DESIGN: 3-phase within-patient (A-B-A)
    3. OVERRIDE PROTOCOL: Patient wins — always
    4. MICROPHENOMENOLOGY: Integration with qualitative methods
    5. PPI CO-DESIGN: Patient/Public Involvement requirement

This is a SPECIFICATION GENERATOR, not a clinical protocol itself.
The output must be reviewed and approved by:
    - Ethics committee / IRB
    - Clinical team
    - Patient advisory group (PPI)
    - External critical friend

No external dependencies beyond stdlib + json + dataclasses.

Usage:
    python -m descartes.council_controls.priority7_monitoring.monitoring_protocol
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from ..config_council import QAP


# ---------------------------------------------------------------------------
# Instrument definitions
# ---------------------------------------------------------------------------

@dataclass
class InstrumentSpec:
    """Specification for a monitoring instrument."""
    name: str
    full_name: str
    abbreviation: str
    description: str
    purpose_in_descartes: str
    administration: str       # "self-report", "interview", "app-based"
    frequency: str            # "daily", "weekly", "per-session", etc.
    estimated_time_min: int
    scoring: str
    validated: bool
    references: List[str]
    notes: List[str] = field(default_factory=list)


INSTRUMENTS = [
    InstrumentSpec(
        name="Metacognition Questionnaire",
        full_name="Metacognitions Questionnaire-30",
        abbreviation="MCQ",
        description=(
            "30-item self-report measuring metacognitive beliefs and processes. "
            "Five subscales: positive beliefs about worry, negative beliefs about "
            "uncontrollability/danger, cognitive confidence, need to control "
            "thoughts, cognitive self-consciousness."
        ),
        purpose_in_descartes=(
            "Track whether neural surrogate interventions alter metacognitive "
            "awareness or beliefs. Phase A prediction: no change expected in "
            "gradual replacement paradigm. Deviation would be a safety signal."
        ),
        administration="self-report",
        frequency="weekly",
        estimated_time_min=10,
        scoring="5-point Likert per item, subscale and total scores",
        validated=True,
        references=[
            "Wells & Cartwright-Hatton (2004) J Behav Ther Exp Psychiatry",
        ],
        notes=[
            "Administer at same time of day each week to reduce circadian confound",
            "Baseline period: minimum 4 weeks before any intervention",
        ],
    ),
    InstrumentSpec(
        name="Range and Differentiation of Emotional Experience Scale",
        full_name="Range and Differentiation of Emotional Experience Scale",
        abbreviation="RDEES",
        description=(
            "14-item self-report measuring two aspects of emotional complexity: "
            "range (breadth of emotions experienced) and differentiation "
            "(granularity of emotional distinctions)."
        ),
        purpose_in_descartes=(
            "Monitor whether surrogate integration affects emotional granularity. "
            "DESCARTES hypothesis: if emotional experience emerges from neural "
            "computation, gradual substrate changes could alter emotional "
            "differentiation. Track as safety endpoint."
        ),
        administration="self-report",
        frequency="weekly",
        estimated_time_min=5,
        scoring="5-point Likert, separate Range and Differentiation subscores",
        validated=True,
        references=[
            "Kang & Shaver (2004) European Journal of Personality",
        ],
        notes=[
            "Particularly sensitive to changes in emotional experience quality",
            "Compare with EMA ecological data for convergent validity",
        ],
    ),
    InstrumentSpec(
        name="Ecological Momentary Assessment",
        full_name="Ecological Momentary Assessment — Emotional Experience",
        abbreviation="EMA",
        description=(
            "Brief (2-3 min) in-the-moment assessments delivered via smartphone "
            "app at semi-random intervals throughout the day. Captures current "
            "emotional state, cognitive clarity, sense of agency, and any "
            "unusual experiences in real-time."
        ),
        purpose_in_descartes=(
            "Capture moment-to-moment emotional and cognitive experience during "
            "surrogate integration. Unlike retrospective questionnaires, EMA "
            "reduces recall bias. Critical for detecting transient disruptions "
            "that weekly measures would miss."
        ),
        administration="app-based",
        frequency="4-6 times daily (semi-random within waking hours)",
        estimated_time_min=3,
        scoring=(
            "Visual analogue scales (0-100) for valence, arousal, agency, "
            "clarity. Free-text option for unusual experiences. "
            "Time-series analysis across days."
        ),
        validated=True,
        references=[
            "Shiffman, Stone & Hufford (2008) Annual Review of Clinical Psychology",
        ],
        notes=[
            "Participant burden is primary concern — monitor compliance rates",
            "Reduce frequency if compliance drops below 70%",
            "All data encrypted and stored locally until sync",
            "Patient can skip any prompt without consequence",
        ],
    ),
    InstrumentSpec(
        name="Dissociative Experiences Scale",
        full_name="Dissociative Experiences Scale-II",
        abbreviation="DES",
        description=(
            "28-item self-report measuring dissociative experiences including "
            "absorption, depersonalisation, derealization, and amnesia. "
            "Widely used screening tool for dissociative symptoms."
        ),
        purpose_in_descartes=(
            "SAFETY INSTRUMENT: Monitor for dissociative symptoms during "
            "surrogate integration. Theoretical risk: substrate replacement "
            "could induce feelings of depersonalisation or derealization. "
            "Any increase above baseline triggers clinical review."
        ),
        administration="self-report",
        frequency="weekly",
        estimated_time_min=10,
        scoring=(
            "0-100 visual analogue scale per item, mean score. "
            "Clinical cutoff: mean >= 30 warrants clinical interview. "
            "DESCARTES safety threshold: increase of >= 10 points from "
            "baseline triggers review."
        ),
        validated=True,
        references=[
            "Carlson & Putnam (1993) American Journal of Psychiatry",
            "Bernstein & Putnam (1986) J Nervous Mental Disease",
        ],
        notes=[
            "PRIMARY SAFETY INSTRUMENT — never skip administration",
            "Any score increase >= 10 from baseline: pause intervention, "
            "clinical review within 48 hours",
            "Patient informed of safety thresholds in advance",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Control design
# ---------------------------------------------------------------------------

@dataclass
class ControlPhase:
    """One phase of the within-patient control design."""
    phase_label: str       # "A1", "B", "A2"
    phase_name: str        # "Baseline", "Intervention", "Washout"
    description: str
    minimum_duration_weeks: int
    instruments_active: List[str]
    intervention_active: bool


CONTROL_DESIGN = [
    ControlPhase(
        phase_label="A1",
        phase_name="Baseline",
        description=(
            "Pre-intervention baseline. All instruments administered at full "
            "frequency. No surrogate integration. Establishes individual "
            "baseline for each measure. Minimum 4 weeks to capture "
            "natural variability."
        ),
        minimum_duration_weeks=4,
        instruments_active=["MCQ", "RDEES", "EMA", "DES"],
        intervention_active=False,
    ),
    ControlPhase(
        phase_label="B",
        phase_name="Intervention",
        description=(
            "Gradual surrogate integration phase. All instruments continue. "
            "Surrogate replacement proceeds according to Phase A protocol. "
            "Continuous monitoring for safety thresholds. Patient can "
            "pause or withdraw at any time."
        ),
        minimum_duration_weeks=8,
        instruments_active=["MCQ", "RDEES", "EMA", "DES"],
        intervention_active=True,
    ),
    ControlPhase(
        phase_label="A2",
        phase_name="Washout / Follow-up",
        description=(
            "Post-intervention monitoring. Surrogate integration paused "
            "or reversed if technically possible. All instruments continue. "
            "Tracks whether any changes observed during B phase persist, "
            "resolve, or worsen."
        ),
        minimum_duration_weeks=4,
        instruments_active=["MCQ", "RDEES", "EMA", "DES"],
        intervention_active=False,
    ),
]


# ---------------------------------------------------------------------------
# Override protocol
# ---------------------------------------------------------------------------

@dataclass
class OverrideProtocol:
    """Patient override and safety protocol."""
    principle: str
    rules: List[str]
    safety_thresholds: List[str]
    clinical_review_triggers: List[str]
    immediate_stop_triggers: List[str]


OVERRIDE_PROTOCOL = OverrideProtocol(
    principle=(
        "PATIENT WINS: The patient's decision to continue, pause, modify, "
        "or withdraw from the protocol always takes precedence over "
        "scientific objectives. No pressure, no persuasion, no cooling-off "
        "periods for withdrawal decisions."
    ),
    rules=[
        "Patient can pause surrogate integration at any time without reason",
        "Patient can withdraw from study at any time without reason",
        "Patient can skip any instrument administration without consequence",
        "Patient can request reversal of surrogate integration at any time",
        "Patient can request to see all their data at any time",
        "Patient can request deletion of their data (within legal limits)",
        "No incentive structure that penalises withdrawal or non-compliance",
        "Research team cannot override patient decisions for scientific reasons",
        "Family/carer concerns trigger clinical review (not automatic stop)",
    ],
    safety_thresholds=[
        "DES increase >= 10 points from baseline: PAUSE + clinical review",
        "MCQ negative beliefs subscale increase >= 1 SD: clinical review",
        "EMA agency score drop >= 20 points sustained > 48 hours: clinical review",
        "Any patient-reported unusual experience: clinical review within 24 hours",
        "Compliance < 50% for 2 consecutive weeks: welfare check (not penalty)",
    ],
    clinical_review_triggers=[
        "Any safety threshold breached",
        "Patient request",
        "Family/carer concern reported",
        "Clinician observation of behavioural change",
        "Unexpected pattern in EMA time series (automated detection)",
    ],
    immediate_stop_triggers=[
        "Patient requests stop",
        "DES score crosses clinical cutoff (mean >= 30)",
        "Any acute psychiatric emergency",
        "Technical malfunction of surrogate system",
        "Loss of data integrity or security breach",
    ],
)


# ---------------------------------------------------------------------------
# Microphenomenology integration
# ---------------------------------------------------------------------------

@dataclass
class MicrophenomenologySpec:
    """Integration of microphenomenological interview methods."""
    description: str
    purpose: str
    method: str
    frequency: str
    interviewer_requirements: List[str]
    data_handling: str
    integration_with_quantitative: str


MICROPHENOMENOLOGY = MicrophenomenologySpec(
    description=(
        "Microphenomenology is a structured interview technique for eliciting "
        "detailed first-person accounts of lived experience. Based on the "
        "explicitation interview method, it guides participants to describe "
        "the fine-grained structure of specific experiential moments."
    ),
    purpose=(
        "Capture qualitative aspects of experience during surrogate "
        "integration that quantitative instruments cannot detect. "
        "Particularly important for: (a) novel experiential phenomena "
        "not covered by standard scales, (b) subtle changes in the quality "
        "of experience, (c) the lived meaning of any quantitative changes."
    ),
    method=(
        "Semi-structured interview following Petitmengin (2006) protocol. "
        "Focus on specific recent moments identified through EMA data "
        "(e.g., moments of high/low agency, unusual experiences). "
        "Audio-recorded with consent, transcribed, coded using established "
        "microphenomenological coding framework."
    ),
    frequency=(
        "Bi-weekly during intervention phase (B). Monthly during baseline "
        "(A1) and follow-up (A2). Additional sessions triggered by safety "
        "events or patient request."
    ),
    interviewer_requirements=[
        "Trained in microphenomenological / explicitation interview method",
        "Independent from the research team running the surrogate integration",
        "Clinical psychology or qualitative research background",
        "Supervised by experienced microphenomenologist",
    ],
    data_handling=(
        "Audio recordings encrypted at rest. Transcriptions pseudonymised. "
        "Qualitative data stored separately from quantitative data. "
        "Patient can review and redact transcripts before inclusion."
    ),
    integration_with_quantitative=(
        "Microphenomenological findings are NOT used to override quantitative "
        "safety thresholds. They provide ADDITIONAL context: (a) help interpret "
        "quantitative changes, (b) detect phenomena not captured by instruments, "
        "(c) inform protocol modifications via PPI co-design process."
    ),
)


# ---------------------------------------------------------------------------
# PPI co-design
# ---------------------------------------------------------------------------

@dataclass
class PPISpec:
    """Patient and Public Involvement specification."""
    requirement_level: str
    description: str
    roles: List[str]
    minimum_composition: str
    meeting_frequency: str
    decision_authority: List[str]
    compensation: str
    training_provided: List[str]


PPI_CODESIGN = PPISpec(
    requirement_level="MANDATORY",
    description=(
        "Patient and Public Involvement (PPI) is a mandatory component of "
        "the DESCARTES monitoring protocol. PPI members co-design the "
        "protocol, review instruments, and have decision authority over "
        "patient-facing aspects of the research."
    ),
    roles=[
        "Protocol co-design: review and modify all patient-facing materials",
        "Instrument selection: approve/reject proposed instruments",
        "Safety threshold review: input on what constitutes meaningful change",
        "Ongoing oversight: regular review of anonymised aggregate data",
        "Communication review: approve all publications before submission",
        "Adverse event review: independent patient perspective on safety events",
    ],
    minimum_composition=(
        "At least 3 PPI members including: (a) person with lived experience "
        "of epilepsy or neurological condition, (b) carer/family member, "
        "(c) patient advocate with research experience. Diversity in age, "
        "gender, ethnicity, and condition severity."
    ),
    meeting_frequency=(
        "Monthly during protocol design. Quarterly during active study. "
        "Ad hoc meetings for safety events."
    ),
    decision_authority=[
        "Veto power over patient-facing materials and communications",
        "Veto power over instrument selection and frequency",
        "Advisory (not veto) on scientific design decisions",
        "Binding recommendations on safety threshold definitions",
        "Binding recommendations on withdrawal/override procedures",
    ],
    compensation=(
        "PPI members compensated at NIHR-recommended rates (or local "
        "equivalent). Travel expenses covered. No expectation of volunteer "
        "contribution. Payment not contingent on agreement with research team."
    ),
    training_provided=[
        "Plain-language introduction to DESCARTES project and goals",
        "Explanation of surrogate technology at accessible level",
        "Research methods overview relevant to their role",
        "Confidentiality and data protection training",
        "Option to attend relevant conferences (expenses covered)",
    ],
)


# ---------------------------------------------------------------------------
# Protocol assembly
# ---------------------------------------------------------------------------

@dataclass
class MonitoringProtocol:
    """Complete monitoring protocol specification."""
    version: str
    generated_date: str
    status: str
    instruments: List[InstrumentSpec]
    control_design: List[ControlPhase]
    override_protocol: OverrideProtocol
    microphenomenology: MicrophenomenologySpec
    ppi_codesign: PPISpec
    governance_caveats: List[str]


def generate_protocol() -> MonitoringProtocol:
    """Assemble the complete monitoring protocol specification."""
    return MonitoringProtocol(
        version="1.0-DRAFT",
        generated_date=datetime.now().strftime("%Y-%m-%d"),
        status="DRAFT — requires IRB, clinical team, and PPI review",
        instruments=INSTRUMENTS,
        control_design=CONTROL_DESIGN,
        override_protocol=OVERRIDE_PROTOCOL,
        microphenomenology=MICROPHENOMENOLOGY,
        ppi_codesign=PPI_CODESIGN,
        governance_caveats=[
            "This is an AI-generated SPECIFICATION, not a clinical protocol",
            "Must be reviewed and modified by ethics committee / IRB",
            "Must be reviewed and modified by clinical team",
            "Must be co-designed with PPI advisory group",
            "Must be reviewed by external critical friend",
            "Instrument selection subject to PPI veto",
            "Safety thresholds subject to clinical team approval",
            "Protocol must be pre-registered before implementation",
        ],
    )


# ---------------------------------------------------------------------------
# Output: JSON
# ---------------------------------------------------------------------------

def save_protocol_json(protocol: MonitoringProtocol,
                       output_dir: str = None) -> str:
    """Save protocol as machine-readable JSON."""
    if output_dir is None:
        output_dir = os.path.join(QAP.results_dir, "phase7_monitoring")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    filepath = out_path / "monitoring_protocol.json"
    result = asdict(protocol)
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print("JSON saved: {}".format(filepath))
    return str(filepath)


# ---------------------------------------------------------------------------
# Output: Markdown
# ---------------------------------------------------------------------------

def save_protocol_markdown(protocol: MonitoringProtocol,
                           output_dir: str = None) -> str:
    """Save protocol as human-readable Markdown."""
    if output_dir is None:
        output_dir = os.path.join(QAP.results_dir, "phase7_monitoring")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    filepath = out_path / "monitoring_protocol.md"
    lines = []

    lines.append("# DESCARTES Monitoring Protocol Specification")
    lines.append("")
    lines.append("**Version:** {}".format(protocol.version))
    lines.append("**Generated:** {}".format(protocol.generated_date))
    lines.append("**Status:** {}".format(protocol.status))
    lines.append("")

    # Governance caveats
    lines.append("## Governance Caveats")
    lines.append("")
    for c in protocol.governance_caveats:
        lines.append("- {}".format(c))
    lines.append("")

    # Instruments
    lines.append("---")
    lines.append("## 1. Instruments")
    lines.append("")
    for inst in protocol.instruments:
        lines.append("### {} ({})".format(inst.full_name, inst.abbreviation))
        lines.append("")
        lines.append("- **Administration:** {}".format(inst.administration))
        lines.append("- **Frequency:** {}".format(inst.frequency))
        lines.append("- **Time:** ~{} minutes".format(inst.estimated_time_min))
        lines.append("- **Scoring:** {}".format(inst.scoring))
        lines.append("- **Validated:** {}".format("Yes" if inst.validated else "No"))
        lines.append("")
        lines.append("**Description:** {}".format(inst.description))
        lines.append("")
        lines.append("**Purpose in DESCARTES:** {}".format(
            inst.purpose_in_descartes))
        lines.append("")
        if inst.notes:
            lines.append("**Notes:**")
            for n in inst.notes:
                lines.append("- {}".format(n))
            lines.append("")
        lines.append("**References:**")
        for r in inst.references:
            lines.append("- {}".format(r))
        lines.append("")

    # Control design
    lines.append("---")
    lines.append("## 2. Control Design (3-Phase Within-Patient A-B-A)")
    lines.append("")
    for phase in protocol.control_design:
        lines.append("### Phase {} — {}".format(
            phase.phase_label, phase.phase_name))
        lines.append("")
        lines.append("- **Minimum duration:** {} weeks".format(
            phase.minimum_duration_weeks))
        lines.append("- **Instruments:** {}".format(
            ", ".join(phase.instruments_active)))
        lines.append("- **Intervention active:** {}".format(
            "Yes" if phase.intervention_active else "No"))
        lines.append("")
        lines.append("{}".format(phase.description))
        lines.append("")

    # Override protocol
    lines.append("---")
    lines.append("## 3. Override Protocol")
    lines.append("")
    lines.append("**Principle:** {}".format(
        protocol.override_protocol.principle))
    lines.append("")
    lines.append("### Rules")
    for r in protocol.override_protocol.rules:
        lines.append("- {}".format(r))
    lines.append("")
    lines.append("### Safety Thresholds")
    for s in protocol.override_protocol.safety_thresholds:
        lines.append("- {}".format(s))
    lines.append("")
    lines.append("### Clinical Review Triggers")
    for t in protocol.override_protocol.clinical_review_triggers:
        lines.append("- {}".format(t))
    lines.append("")
    lines.append("### Immediate Stop Triggers")
    for t in protocol.override_protocol.immediate_stop_triggers:
        lines.append("- {}".format(t))
    lines.append("")

    # Microphenomenology
    lines.append("---")
    lines.append("## 4. Microphenomenology Integration")
    lines.append("")
    mp = protocol.microphenomenology
    lines.append("**Purpose:** {}".format(mp.purpose))
    lines.append("")
    lines.append("**Method:** {}".format(mp.method))
    lines.append("")
    lines.append("**Frequency:** {}".format(mp.frequency))
    lines.append("")
    lines.append("### Interviewer Requirements")
    for r in mp.interviewer_requirements:
        lines.append("- {}".format(r))
    lines.append("")
    lines.append("**Data Handling:** {}".format(mp.data_handling))
    lines.append("")
    lines.append("**Integration with Quantitative Data:** {}".format(
        mp.integration_with_quantitative))
    lines.append("")

    # PPI
    lines.append("---")
    lines.append("## 5. Patient and Public Involvement (PPI) Co-Design")
    lines.append("")
    ppi = protocol.ppi_codesign
    lines.append("**Requirement Level:** {}".format(ppi.requirement_level))
    lines.append("")
    lines.append("**Description:** {}".format(ppi.description))
    lines.append("")
    lines.append("**Minimum Composition:** {}".format(ppi.minimum_composition))
    lines.append("")
    lines.append("**Meeting Frequency:** {}".format(ppi.meeting_frequency))
    lines.append("")
    lines.append("### Roles")
    for r in ppi.roles:
        lines.append("- {}".format(r))
    lines.append("")
    lines.append("### Decision Authority")
    for d in ppi.decision_authority:
        lines.append("- {}".format(d))
    lines.append("")
    lines.append("**Compensation:** {}".format(ppi.compensation))
    lines.append("")
    lines.append("### Training Provided")
    for t in ppi.training_provided:
        lines.append("- {}".format(t))
    lines.append("")

    content = "\n".join(lines)
    with open(filepath, 'w') as f:
        f.write(content)

    print("Markdown saved: {}".format(filepath))
    return str(filepath)


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_protocol_summary(protocol: MonitoringProtocol):
    """Print protocol summary to console."""
    print("\n" + "=" * 70)
    print("DESCARTES MONITORING PROTOCOL SPECIFICATION")
    print("Version: {}  |  Status: {}".format(
        protocol.version, protocol.status))
    print("=" * 70)

    print("\n--- Instruments ---")
    for inst in protocol.instruments:
        print("  {} ({}) — {} — {}".format(
            inst.abbreviation, inst.administration,
            inst.frequency, inst.purpose_in_descartes[:60] + "..."))

    print("\n--- Control Design ---")
    for phase in protocol.control_design:
        print("  {} ({}): {} weeks min, intervention={}".format(
            phase.phase_label, phase.phase_name,
            phase.minimum_duration_weeks, phase.intervention_active))

    print("\n--- Override Protocol ---")
    print("  Principle: {}".format(
        protocol.override_protocol.principle[:70] + "..."))
    print("  {} rules, {} safety thresholds, {} stop triggers".format(
        len(protocol.override_protocol.rules),
        len(protocol.override_protocol.safety_thresholds),
        len(protocol.override_protocol.immediate_stop_triggers)))

    print("\n--- Microphenomenology ---")
    print("  Frequency: {}".format(protocol.microphenomenology.frequency))

    print("\n--- PPI Co-Design ---")
    print("  Requirement: {}".format(protocol.ppi_codesign.requirement_level))
    print("  {} roles, {} decision authorities".format(
        len(protocol.ppi_codesign.roles),
        len(protocol.ppi_codesign.decision_authority)))

    print("\n--- Governance Caveats ---")
    for c in protocol.governance_caveats:
        print("  - {}".format(c))
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_phase7(output_dir: str = None) -> MonitoringProtocol:
    """Generate and save the monitoring protocol specification."""
    if output_dir is None:
        output_dir = os.path.join(QAP.results_dir, "phase7_monitoring")

    protocol = generate_protocol()
    print_protocol_summary(protocol)
    save_protocol_json(protocol, output_dir)
    save_protocol_markdown(protocol, output_dir)

    return protocol


if __name__ == '__main__':
    run_phase7()
