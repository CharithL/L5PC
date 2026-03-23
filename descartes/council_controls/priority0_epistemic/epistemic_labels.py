"""
epistemic_labels.py

Phase 0: Epistemic Labeling Infrastructure

Tags every DESCARTES result with its epistemic phase and governance status
before it can be reported. Enforces cascading constraints:
- ESTABLISHED requires architecture control + baseline quantification + consent
- Phase B claims always carry philosophical commitment caveat
- Governance fields track consent, DURA review, critical friend oversight, PPI

No external dependencies beyond stdlib and dataclasses.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class EpistemicPhase(Enum):
    PHASE_A = "Gradual replacement with biological validation"
    PHASE_B = "Full substrate transfer — philosophical commitment"


class EpistemicStatus(Enum):
    ESTABLISHED = "Supported by evidence with appropriate controls"
    SUPPORTED = "Consistent with evidence, alternatives not ruled out"
    SPECULATIVE = "Plausible given framework, not directly testable"
    OVERCLAIMED = "Stronger than evidence warrants"


@dataclass
class ResultLabel:
    phase: EpistemicPhase
    status: EpistemicStatus
    claim: str
    evidence: str
    deflationary_alternative: str
    architecture_controlled: bool
    baseline_quantified: bool
    # --- GOVERNANCE FIELDS (v3.0) ---
    consent_verified: bool = False
    dura_reviewed: bool = False
    governance_status: str = "UNCHECKED"  # CLEARED / CAVEAT / BLOCKED
    critical_friend_reviewed: bool = False
    ppi_consulted: bool = False
    caveats: List[str] = field(default_factory=list)


def label_result(result_dict: dict, phase: EpistemicPhase, status: EpistemicStatus,
                 claim: str, evidence: str, deflationary_alternative: str,
                 architecture_controlled: bool = False,
                 baseline_quantified: bool = False,
                 consent_verified: bool = False,
                 dura_reviewed: bool = False,
                 governance_status: str = "UNCHECKED",
                 critical_friend_reviewed: bool = False,
                 ppi_consulted: bool = False,
                 involves_human_data: bool = True) -> dict:
    """Attach a ResultLabel to any result dictionary.

    The label is validated before attachment — status may be downgraded
    and caveats added automatically based on control state.
    """
    label = ResultLabel(
        phase=phase,
        status=status,
        claim=claim,
        evidence=evidence,
        deflationary_alternative=deflationary_alternative,
        architecture_controlled=architecture_controlled,
        baseline_quantified=baseline_quantified,
        consent_verified=consent_verified,
        dura_reviewed=dura_reviewed,
        governance_status=governance_status,
        critical_friend_reviewed=critical_friend_reviewed,
        ppi_consulted=ppi_consulted,
    )

    label = validate_label(label, involves_human_data=involves_human_data)

    result_dict['_epistemic_label'] = {
        'phase': label.phase.name,
        'status': label.status.name,
        'claim': label.claim,
        'evidence': label.evidence,
        'deflationary_alternative': label.deflationary_alternative,
        'architecture_controlled': label.architecture_controlled,
        'baseline_quantified': label.baseline_quantified,
        'consent_verified': label.consent_verified,
        'dura_reviewed': label.dura_reviewed,
        'governance_status': label.governance_status,
        'critical_friend_reviewed': label.critical_friend_reviewed,
        'ppi_consulted': label.ppi_consulted,
        'caveats': label.caveats,
    }

    return result_dict


def validate_label(label: ResultLabel,
                   involves_human_data: bool = True) -> ResultLabel:
    """Validate and enforce constraints on a ResultLabel.

    May downgrade status and add caveats. Never upgrades.
    """
    caveats = list(label.caveats)

    # Architecture control gate
    if not label.architecture_controlled and label.status == EpistemicStatus.ESTABLISHED:
        label.status = EpistemicStatus.SUPPORTED
        caveats.append(
            "Status downgraded from ESTABLISHED to SUPPORTED: "
            "non-oscillatory architecture control not yet completed"
        )

    # Baseline quantification gate
    if not label.baseline_quantified and label.status == EpistemicStatus.ESTABLISHED:
        label.status = EpistemicStatus.SUPPORTED
        caveats.append(
            "Status downgraded from ESTABLISHED to SUPPORTED: "
            "50-seed baseline variance not yet quantified"
        )

    # Phase B automatic caveat
    if label.phase == EpistemicPhase.PHASE_B:
        caveat = (
            "This claim rests on a philosophical commitment (reversed axiom), "
            "not empirical evidence"
        )
        if caveat not in caveats:
            caveats.append(caveat)

    # --- GOVERNANCE VALIDATION (v3.0) ---

    # Consent gate (BLOCKS for human data)
    if involves_human_data and not label.consent_verified:
        if label.status in (EpistemicStatus.ESTABLISHED, EpistemicStatus.SUPPORTED):
            if label.status == EpistemicStatus.ESTABLISHED:
                label.status = EpistemicStatus.SUPPORTED
        caveats.append("Dataset consent for secondary ML use not verified")

    # Critical friend gate
    if not label.critical_friend_reviewed:
        caveats.append(
            "AI quality panel recommendation only — not yet "
            "validated by external human reviewer"
        )

    # PPI gate
    if not label.ppi_consulted:
        caveats.append(
            "Experimental design not reviewed by patient/public advisors"
        )

    label.caveats = caveats
    return label


def format_label(label: ResultLabel) -> str:
    """Produce a human-readable summary for inclusion in result reports."""
    lines = []
    lines.append("=" * 60)
    lines.append("EPISTEMIC LABEL")
    lines.append("=" * 60)
    lines.append("Phase:    {} ({})".format(label.phase.name, label.phase.value))
    lines.append("Status:   {} ({})".format(label.status.name, label.status.value))
    lines.append("Claim:    {}".format(label.claim))
    lines.append("Evidence: {}".format(label.evidence))
    lines.append("Deflationary alternative: {}".format(label.deflationary_alternative))
    lines.append("")
    lines.append("--- Controls ---")
    lines.append("Architecture controlled: {}".format(
        "YES" if label.architecture_controlled else "NO"))
    lines.append("Baseline quantified:     {}".format(
        "YES" if label.baseline_quantified else "NO"))
    lines.append("")
    lines.append("--- Governance (v3.0) ---")
    lines.append("Governance status:       {}".format(label.governance_status))
    lines.append("Consent verified:        {}".format(
        "YES" if label.consent_verified else "NO"))
    lines.append("DURA reviewed:           {}".format(
        "YES" if label.dura_reviewed else "NO"))
    lines.append("Critical friend reviewed: {}".format(
        "YES" if label.critical_friend_reviewed else "NO"))
    lines.append("PPI consulted:           {}".format(
        "YES" if label.ppi_consulted else "NO"))

    if label.caveats:
        lines.append("")
        lines.append("--- Caveats ({}) ---".format(len(label.caveats)))
        for i, c in enumerate(label.caveats, 1):
            lines.append("  {}. {}".format(i, c))

    lines.append("=" * 60)
    return "\n".join(lines)
