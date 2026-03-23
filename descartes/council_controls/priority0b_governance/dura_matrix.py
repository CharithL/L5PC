"""
dura_matrix.py

Phase 0B-2: Dual-Use Risk Assessment (DURA) in FMEA Format

Generates the DURA matrix for the DESCARTES program. Must be completed before
publishing any feasibility claims. Appended as supplementary material to all
manuscripts.

No federal-level dual-use framework in any country covers computational brain
replicas — this is a regulatory vacuum that the program must self-govern within.

Usage:
    python -m descartes.council_controls.priority0b_governance.dura_matrix
"""

import json
from datetime import datetime
from pathlib import Path


THREAT_VECTORS = [
    {
        "id": "DURA-01",
        "threat": "Zombie surrogate deployment",
        "mechanism": (
            "State or malicious actor uses substitution architecture for "
            "undetectable cognitive modification — surrogate passes all "
            "behavioral tests but produces alien internal experience"
        ),
        "severity": 10,
        "probability": 2,
        "detectability": 2,
        "rpn": 40,
        "mitigation": (
            "Publish mandatory-variable verification as inseparable from "
            "the replacement methodology. Make unverified replacement "
            "inherently detectable."
        ),
        "responsible": "PI + critical friends",
        "status": "Unmitigated",
    },
    {
        "id": "DURA-02",
        "threat": "Cognitive vulnerability mapping",
        "mechanism": (
            "Pipeline reverse-engineered to identify variables whose "
            "disruption causes maximal phenomenal destabilization"
        ),
        "severity": 8,
        "probability": 3,
        "detectability": 4,
        "rpn": 96,
        "mitigation": (
            "Withhold individual patient variable weightings from publications. "
            "Publish only generalized mathematical framework. Apply RAND "
            "restorative/augmentative/disruptive classification to all "
            "published methods."
        ),
        "responsible": "PI",
        "status": "Unmitigated",
    },
    {
        "id": "DURA-03",
        "threat": "Non-consensual digital cloning",
        "mechanism": (
            "Unauthorized actors train surrogates on intercepted consumer "
            "BCI data to simulate and predict target behavior"
        ),
        "severity": 7,
        "probability": 4,
        "detectability": 5,
        "rpn": 140,
        "mitigation": (
            "Integrate cryptographic watermarking into surrogate training "
            "pipeline to track provenance. Advocate for neural data "
            "governance legislation."
        ),
        "responsible": "PI + policy engagement",
        "status": "Unmitigated",
    },
    {
        "id": "DURA-04",
        "threat": "Cross-patient weaponization",
        "mechanism": (
            "If surrogates generalize across patients, applicable to any "
            "individual without calibration"
        ),
        "severity": 7,
        "probability": 2,
        "detectability": 3,
        "rpn": 42,
        "mitigation": (
            "Cross-patient generalization failure IS a safeguard. Document "
            "explicitly that individual calibration is required and generic "
            "surrogates do not work."
        ),
        "responsible": "PI",
        "status": "Partially mitigated (by empirical finding)",
    },
    {
        "id": "DURA-05",
        "threat": "Coercive memory manipulation",
        "mechanism": (
            "Validated memory prosthesis methodology used for forced memory "
            "erasure or implantation by state actors"
        ),
        "severity": 9,
        "probability": 2,
        "detectability": 3,
        "rpn": 54,
        "mitigation": (
            "Advocate for neurorights protections. Support emerging "
            "BrainMind Asilomar-for-the-Brain initiatives. Include "
            "anti-coercion safeguards in clinical protocol design."
        ),
        "responsible": "PI + policy engagement",
        "status": "Unmitigated",
    },
    {
        "id": "DURA-06",
        "threat": "Military cognitive enhancement",
        "mechanism": (
            "DARPA-style application of replacement methodology for "
            "soldier enhancement or adversary degradation"
        ),
        "severity": 6,
        "probability": 4,
        "detectability": 5,
        "rpn": 120,
        "mitigation": (
            "Responsible disclosure protocol. Classify all published methods "
            "per RAND restorative/augmentative/disruptive taxonomy. "
            "Emphasize therapeutic intent in all publications."
        ),
        "responsible": "PI",
        "status": "Unmitigated",
    },
]


def generate_dura(output_dir: str = "results/governance"):
    """Generate DURA matrix in JSON, Markdown, and supplementary format."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Sort by RPN descending (highest risk first)
    sorted_threats = sorted(THREAT_VECTORS, key=lambda t: t['rpn'], reverse=True)

    # --- JSON output ---
    dura_json = {
        "title": "DESCARTES Dual-Use Risk Assessment (DURA)",
        "version": "3.0",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "methodology": "Failure Mode and Effects Analysis (FMEA)",
        "rpn_formula": "Severity x Probability x Detectability",
        "scale": "1-10 per factor (higher = worse)",
        "regulatory_context": (
            "No federal-level dual-use framework covers computational brain "
            "replicas. This assessment operates in a regulatory vacuum "
            "requiring self-governance."
        ),
        "threats": sorted_threats,
    }
    with open(out_path / "dura_matrix.json", "w") as f:
        json.dump(dura_json, f, indent=2)

    # --- Markdown table ---
    md_lines = [
        "# DESCARTES Dual-Use Risk Assessment (DURA)",
        "",
        "Version 3.0 | Generated: {}".format(datetime.now().strftime("%Y-%m-%d")),
        "",
        "## Methodology",
        "",
        "Failure Mode and Effects Analysis (FMEA). "
        "RPN = Severity x Probability x Detectability (1-10 each, higher = worse).",
        "",
        "**Regulatory context:** No federal-level dual-use framework in any "
        "country covers computational brain replicas. This is a regulatory vacuum.",
        "",
        "## Threat Matrix",
        "",
        "| ID | Threat | S | P | D | RPN | Status |",
        "|-----|--------|---|---|---|-----|--------|",
    ]
    for t in sorted_threats:
        md_lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            t['id'], t['threat'], t['severity'], t['probability'],
            t['detectability'], t['rpn'], t['status']))

    md_lines.extend([
        "",
        "## Detailed Threat Descriptions",
        "",
    ])
    for t in sorted_threats:
        md_lines.extend([
            "### {} — {} (RPN: {})".format(t['id'], t['threat'], t['rpn']),
            "",
            "**Mechanism:** {}".format(t['mechanism']),
            "",
            "**Mitigation:** {}".format(t['mitigation']),
            "",
            "**Responsible:** {}".format(t['responsible']),
            "",
            "**Status:** {}".format(t['status']),
            "",
        ])

    with open(out_path / "dura_matrix.md", "w") as f:
        f.write("\n".join(md_lines))

    # --- Supplementary material ---
    supp_lines = [
        "# Supplementary Material: Dual-Use Risk Assessment",
        "",
        "## DESCARTES Program — Computational Brain Replica Risk Assessment",
        "",
        "### Methodology",
        "",
        "This assessment uses Failure Mode and Effects Analysis (FMEA) to "
        "enumerate and evaluate dual-use risks associated with the DESCARTES "
        "phenomenal recombination methodology. Each threat vector is scored on "
        "three dimensions (Severity, Probability, Detectability; 1-10 scale, "
        "higher = worse) producing a Risk Priority Number (RPN = S x P x D).",
        "",
        "### Regulatory Context",
        "",
        "At the time of writing (March 2026), no federal-level dual-use framework "
        "in any country explicitly covers computational brain replicas or cognitive "
        "surrogates. The DESCARTES program operates in a regulatory vacuum that "
        "necessitates proactive self-governance. This assessment is informed by:",
        "- OECD Recommendation on Responsible Innovation in Neurotechnology "
        "(OECD/LEGAL/0457)",
        "- Chile 2021 constitutional amendment on neurorights",
        "- Chilean Supreme Court Emotiv ruling (2023)",
        "- Emerging UK regulatory framework (ARIA-Newcastle-MHRA partnership)",
        "- BrainMind Asilomar-for-the-Brain principles",
        "",
        "### Threat Enumeration",
        "",
    ]
    for t in sorted_threats:
        supp_lines.extend([
            "**{}: {}** (RPN: {})".format(t['id'], t['threat'], t['rpn']),
            "",
            "*Mechanism:* {}".format(t['mechanism']),
            "",
            "*Mitigation strategy:* {}".format(t['mitigation']),
            "",
            "*Responsible party:* {}".format(t['responsible']),
            "",
            "*Current status:* {}".format(t['status']),
            "",
        ])

    supp_lines.extend([
        "### Responsible Disclosure Statement",
        "",
        "The DESCARTES program commits to:",
        "1. Publishing this DURA as supplementary material with all manuscripts",
        "2. Classifying all published methods per the RAND "
        "restorative/augmentative/disruptive taxonomy",
        "3. Withholding individual patient variable weightings from publications",
        "4. Seeking external review of this assessment by at least one human "
        "critical friend before manuscript submission",
        "5. Updating this assessment as new threat vectors are identified",
        "",
        "### Acknowledgment of Regulatory Vacuum",
        "",
        "This risk assessment does not constitute legal compliance with any "
        "existing regulatory framework, as no such framework currently governs "
        "computational brain replicas. It represents a good-faith effort at "
        "self-governance pending the development of appropriate oversight "
        "mechanisms.",
    ])

    with open(out_path / "dura_supplementary.md", "w") as f:
        f.write("\n".join(supp_lines))

    # Print summary
    print("\n" + "=" * 60)
    print("DURA MATRIX GENERATED")
    print("=" * 60)
    print("Threats identified: {}".format(len(sorted_threats)))
    print("Max RPN: {} ({})".format(
        sorted_threats[0]['rpn'], sorted_threats[0]['threat']))
    print("Min RPN: {} ({})".format(
        sorted_threats[-1]['rpn'], sorted_threats[-1]['threat']))
    print("Unmitigated: {}".format(
        sum(1 for t in sorted_threats if t['status'] == "Unmitigated")))
    print("\nOutputs:")
    print("  {}".format(out_path / "dura_matrix.json"))
    print("  {}".format(out_path / "dura_matrix.md"))
    print("  {}".format(out_path / "dura_supplementary.md"))
    print("\nNOTE: Requires review by human critical friend before "
          "constituting a governance document.")


if __name__ == '__main__':
    generate_dura()
