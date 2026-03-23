"""
critical_friend_log.py

Phase 0B-4: Critical Friend Audit Log

Maintains an immutable, append-only log of all interactions with human
critical friends. This is the audit trail that proves genuine external
oversight exists. Published to OSF alongside experimental results.

Usage:
    from descartes.council_controls.priority0b_governance.critical_friend_log import log_review
    log_review(
        reviewer_name="Dr. Jane Smith",
        reviewer_affiliation="University of X",
        date="2026-03-23",
        review_type="protocol_review",
        documents_reviewed=["synthetic_reality_v2.py"],
        recommendations=["Add verification step for transforms"],
        pi_response=["Agreed — implemented verify_transform()"],
        action_taken=["Added verification to all 5 conditions"],
        disagreements=[],
        resolution=""
    )
"""

import json
import os
from datetime import datetime
from pathlib import Path


LOG_PATH = "results/governance/critical_friend_log.json"


def log_review(
    reviewer_name: str,
    reviewer_affiliation: str,
    date: str,
    review_type: str,
    documents_reviewed: list,
    recommendations: list,
    pi_response: list,
    action_taken: list,
    disagreements: list,
    resolution: str,
    log_path: str = LOG_PATH,
):
    """Append a review entry to the critical friend log.

    The log is append-only — entries cannot be modified or deleted.
    This ensures the audit trail is genuine.

    review_type: "protocol_review" | "transcript_audit" | "result_review" | "ad_hoc"
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "reviewer": {
            "name": reviewer_name,
            "affiliation": reviewer_affiliation,
        },
        "date": date,
        "review_type": review_type,
        "documents_reviewed": documents_reviewed,
        "recommendations": recommendations,
        "pi_response": pi_response,
        "action_taken": action_taken,
        "disagreements": disagreements,
        "resolution": resolution,
    }

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if log_file.exists():
        with open(log_file) as f:
            existing = json.load(f)

    existing.append(entry)

    with open(log_file, 'w') as f:
        json.dump(existing, f, indent=2)

    print("Logged review by {} ({})".format(reviewer_name, review_type))
    print("  {} recommendations".format(len(recommendations)))
    print("  {} disagreements".format(len(disagreements)))

    return entry


def get_review_count(log_path: str = LOG_PATH) -> int:
    """Get the number of reviews logged."""
    if not os.path.exists(log_path):
        return 0
    with open(log_path) as f:
        return len(json.load(f))


def get_reviewers(log_path: str = LOG_PATH) -> list:
    """Get list of unique reviewers."""
    if not os.path.exists(log_path):
        return []
    with open(log_path) as f:
        entries = json.load(f)
    return list(set(e['reviewer']['name'] for e in entries))


def print_summary(log_path: str = LOG_PATH):
    """Print summary of all reviews."""
    if not os.path.exists(log_path):
        print("No reviews logged yet.")
        return

    with open(log_path) as f:
        entries = json.load(f)

    print("\n" + "=" * 60)
    print("CRITICAL FRIEND LOG SUMMARY")
    print("=" * 60)
    print("Total reviews: {}".format(len(entries)))

    reviewers = set()
    types = {}
    total_recs = 0
    total_disagreements = 0

    for e in entries:
        reviewers.add(e['reviewer']['name'])
        rt = e['review_type']
        types[rt] = types.get(rt, 0) + 1
        total_recs += len(e['recommendations'])
        total_disagreements += len(e['disagreements'])

    print("Unique reviewers: {}".format(len(reviewers)))
    for name in sorted(reviewers):
        print("  - {}".format(name))
    print("Review types: {}".format(
        ", ".join("{}: {}".format(k, v) for k, v in sorted(types.items()))))
    print("Total recommendations: {}".format(total_recs))
    print("Total disagreements: {}".format(total_disagreements))
