"""
governance_checklist.py

Phase 0B-3: Per-Experiment Governance Gate

Every experiment must pass a governance checklist alongside the epistemic
label from Phase 0. This is the external complement to internal quality checks.

BLOCKING: consent_verified for human data
RECOMMENDED (at TRL 2-3): all other checks

Usage:
    from descartes.council_controls.priority0b_governance.governance_checklist import (
        check_governance, governance_caveats
    )
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List


@dataclass
class GovernanceCheck:
    consent_verified: bool = False
    dura_reviewed: bool = False
    ppi_consulted: bool = False
    critical_friend_notified: bool = False
    pre_registered: bool = False
    blinded_analysis: bool = False


def check_governance(experiment_config: dict,
                     consent_audit_path: str = "results/governance/consent_audit.json"
                     ) -> GovernanceCheck:
    """Check governance prerequisites for an experiment.

    experiment_config should contain:
        dataset_name: str (matches consent_audit.json)
        involves_human_data: bool
        experiment_description: str
    """
    check = GovernanceCheck()

    dataset_name = experiment_config.get('dataset_name', '')
    involves_human = experiment_config.get('involves_human_data', True)

    # Check consent audit
    if os.path.exists(consent_audit_path):
        with open(consent_audit_path) as f:
            audits = json.load(f)
        for a in audits:
            if a['dataset_name'] == dataset_name:
                status = a.get('governance_status', 'UNCHECKED')
                if status in ('CLEARED', 'CAVEAT'):
                    check.consent_verified = True
                break

    # Animal data is always consent-verified
    if not involves_human:
        check.consent_verified = True

    return check


def governance_caveats(check: GovernanceCheck) -> List[str]:
    """Generate list of governance caveats for result labeling."""
    caveats = []
    if not check.consent_verified:
        caveats.append("Dataset consent for secondary ML use not verified")
    if not check.critical_friend_notified:
        caveats.append(
            "AI quality panel recommendation only — not yet "
            "validated by external human reviewer"
        )
    if not check.ppi_consulted:
        caveats.append(
            "Experimental design not reviewed by patient/public advisors"
        )
    if not check.pre_registered:
        caveats.append("Protocol not pre-registered on OSF")
    if not check.blinded_analysis:
        caveats.append(
            "Analysis not blinded — condition labels visible during analysis"
        )
    return caveats


def is_blocked(check: GovernanceCheck, involves_human_data: bool = True) -> bool:
    """Check if the experiment is BLOCKED by governance requirements.

    At TRL 2-3, only consent is blocking for human data.
    """
    if involves_human_data and not check.consent_verified:
        return True
    return False


def save_governance_check(experiment_id: str, check: GovernanceCheck,
                          experiment_config: dict,
                          output_dir: str = "results/governance/governance_checks"):
    """Save governance check results."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    result = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat(),
        'config': experiment_config,
        'check': asdict(check),
        'caveats': governance_caveats(check),
        'blocked': is_blocked(check, experiment_config.get('involves_human_data', True)),
    }

    date_str = datetime.now().strftime("%Y%m%d")
    filename = "{}_{}_governance.json".format(experiment_id, date_str)
    with open(out_path / filename, 'w') as f:
        json.dump(result, f, indent=2)

    return result
