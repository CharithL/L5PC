"""
check_ictal_epochs.py

Phase 4A: Check DANDI 000576 for Seizure/Ictal Data

Searches DANDI 000576 (Rutishauser human MTL->frontal) for seizure or
ictal epoch annotations. Uses dandi/pynwb if available, otherwise falls
back to metadata parsing.

GOVERNANCE: Checks consent_audit FIRST. If dataset status is BLOCKED,
exits immediately without touching any data.

Annotations searched:
  - "seizure", "ictal", "aura", "preictal", "postictal",
    "interictal", "epileptiform"

Usage:
    python -m descartes.council_controls.priority4_seizure_aura.check_ictal_epochs
"""

import json
import os
from pathlib import Path

DATASET_NAME = "DANDI 000576"
DANDISET_ID = "000576"
RESULTS_DIR = Path("results/council_controls/phase4_seizure_aura")

ICTAL_KEYWORDS = [
    "seizure", "ictal", "aura", "preictal", "pre-ictal",
    "postictal", "post-ictal", "interictal", "inter-ictal",
    "epileptiform", "epileptic", "onset", "spread",
]


# ---------------------------------------------------------------------------
# Governance gate
# ---------------------------------------------------------------------------

def check_consent_gate(audit_path="results/governance/consent_audit.json"):
    """Check governance status before accessing any data.

    Returns:
        tuple: (status_str, can_proceed: bool)
    """
    if not os.path.exists(audit_path):
        print("WARNING: No consent audit found at {}".format(audit_path))
        print("Running consent audit is REQUIRED before Phase 4.")
        print("Run: python -m descartes.council_controls.priority0b_governance.consent_audit")
        return "UNCHECKED", False

    with open(audit_path) as f:
        audits = json.load(f)

    for a in audits:
        if a.get("dataset_name") == DATASET_NAME:
            status = a.get("governance_status", "UNCHECKED")
            if status == "BLOCKED":
                print("GOVERNANCE BLOCK: {} is BLOCKED.".format(DATASET_NAME))
                print("Reason: {}".format(a.get("justification", "unknown")))
                print("Cannot proceed with Phase 4 analysis.")
                return status, False
            elif status in ("CLEARED", "CAVEAT"):
                print("Consent status for {}: {}".format(DATASET_NAME, status))
                if status == "CAVEAT":
                    print("CAVEAT: {}".format(a.get("justification", "")))
                return status, True
            else:
                print("Unknown governance status: {}".format(status))
                return status, False

    print("WARNING: {} not found in consent audit.".format(DATASET_NAME))
    return "UNCHECKED", False


# ---------------------------------------------------------------------------
# NWB-based search (preferred path)
# ---------------------------------------------------------------------------

def search_nwb_files(dandiset_path):
    """Search NWB files for ictal annotations using pynwb.

    Args:
        dandiset_path: Local path to downloaded DANDI 000576 NWB files.

    Returns:
        list of dicts with ictal epoch info
    """
    try:
        import pynwb
    except ImportError:
        print("pynwb not available — falling back to metadata search.")
        return None

    nwb_files = list(Path(dandiset_path).rglob("*.nwb"))
    if not nwb_files:
        print("No NWB files found in {}".format(dandiset_path))
        return []

    ictal_epochs = []
    for nwb_path in nwb_files:
        print("  Scanning: {}".format(nwb_path.name))
        try:
            io = pynwb.NWBHDF5IO(str(nwb_path), "r")
            nwbfile = io.read()

            # Check epochs / trials / time intervals
            sources_to_check = []
            if hasattr(nwbfile, 'epochs') and nwbfile.epochs is not None:
                sources_to_check.append(("epochs", nwbfile.epochs))
            if hasattr(nwbfile, 'trials') and nwbfile.trials is not None:
                sources_to_check.append(("trials", nwbfile.trials))
            if hasattr(nwbfile, 'intervals'):
                for name, interval in nwbfile.intervals.items():
                    sources_to_check.append((name, interval))

            for source_name, table in sources_to_check:
                # Check column names for ictal keywords
                col_names = table.colnames if hasattr(table, 'colnames') else []
                for col in col_names:
                    col_lower = col.lower()
                    for kw in ICTAL_KEYWORDS:
                        if kw in col_lower:
                            ictal_epochs.append({
                                "file": nwb_path.name,
                                "source": source_name,
                                "column": col,
                                "keyword_match": kw,
                                "n_rows": len(table),
                            })

                # Check cell values in string columns
                for col in col_names:
                    try:
                        values = table[col][:]
                        if hasattr(values[0], 'lower'):
                            for i, v in enumerate(values):
                                v_lower = str(v).lower()
                                for kw in ICTAL_KEYWORDS:
                                    if kw in v_lower:
                                        ictal_epochs.append({
                                            "file": nwb_path.name,
                                            "source": source_name,
                                            "column": col,
                                            "row": i,
                                            "value": str(v)[:200],
                                            "keyword_match": kw,
                                        })
                    except (TypeError, IndexError, KeyError):
                        continue

            # Check processing modules for annotations
            if hasattr(nwbfile, 'processing'):
                for mod_name, mod in nwbfile.processing.items():
                    mod_str = str(mod_name).lower()
                    for kw in ICTAL_KEYWORDS:
                        if kw in mod_str:
                            ictal_epochs.append({
                                "file": nwb_path.name,
                                "source": "processing_module",
                                "module": mod_name,
                                "keyword_match": kw,
                            })

            io.close()

        except Exception as e:
            print("  Error reading {}: {}".format(nwb_path.name, e))

    return ictal_epochs


# ---------------------------------------------------------------------------
# Metadata-based search (fallback)
# ---------------------------------------------------------------------------

def search_dandi_metadata():
    """Search DANDI API metadata for ictal annotations (no NWB download).

    Returns:
        list of dicts with metadata-level ictal indicators
    """
    results = []

    # Try dandi client
    try:
        from dandi.dandiapi import DandiAPIClient

        client = DandiAPIClient()
        dandiset = client.get_dandiset(DANDISET_ID)

        # Check dandiset-level metadata
        meta = dandiset.get_raw_metadata()
        meta_str = json.dumps(meta).lower()
        for kw in ICTAL_KEYWORDS:
            if kw in meta_str:
                results.append({
                    "source": "dandiset_metadata",
                    "keyword_match": kw,
                    "level": "dandiset",
                })

        # Check asset-level metadata
        for asset in dandiset.get_assets():
            asset_meta = asset.get_raw_metadata()
            asset_str = json.dumps(asset_meta).lower()
            for kw in ICTAL_KEYWORDS:
                if kw in asset_str:
                    results.append({
                        "source": "asset_metadata",
                        "asset": asset.path,
                        "keyword_match": kw,
                        "level": "asset",
                    })

        return results

    except ImportError:
        print("dandi client not available.")

    except Exception as e:
        print("DANDI API error: {}".format(e))

    # Final fallback: check local metadata cache
    cache_paths = [
        Path("data/dandi/000576/dandiset.yaml"),
        Path("data/dandi/000576/dandiset.json"),
    ]
    for cp in cache_paths:
        if cp.exists():
            with open(cp) as f:
                text = f.read().lower()
            for kw in ICTAL_KEYWORDS:
                if kw in text:
                    results.append({
                        "source": "local_cache",
                        "file": str(cp),
                        "keyword_match": kw,
                    })

    if not results:
        print("No ictal annotations found via metadata search.")
        print("NOTE: Absence in metadata does not guarantee absence in data.")
        print("Full NWB file inspection recommended.")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check_ictal(dandiset_path=None, output_dir=None):
    """Run ictal epoch check with governance gate.

    Args:
        dandiset_path: Local path to NWB files (optional).
        output_dir: Where to save results.

    Returns:
        dict: {"consent_status", "ictal_found", "ictal_epochs", "method"}
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # GOVERNANCE GATE
    consent_status, can_proceed = check_consent_gate()
    if not can_proceed:
        result = {
            "consent_status": consent_status,
            "ictal_found": False,
            "ictal_epochs": [],
            "method": "blocked_by_governance",
            "note": "Phase 4 halted — consent gate not passed.",
        }
        out_file = out_path / "ictal_check.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        return result

    # Search for ictal data
    print("\nSearching for ictal/seizure annotations in {}...".format(DATASET_NAME))

    ictal_epochs = None
    method = "none"

    # Try NWB files first
    if dandiset_path and Path(dandiset_path).exists():
        ictal_epochs = search_nwb_files(dandiset_path)
        method = "nwb_scan"

    # Fall back to metadata
    if ictal_epochs is None:
        ictal_epochs = search_dandi_metadata()
        method = "metadata_search"

    ictal_found = len(ictal_epochs) > 0

    result = {
        "consent_status": consent_status,
        "ictal_found": ictal_found,
        "n_ictal_annotations": len(ictal_epochs),
        "ictal_epochs": ictal_epochs,
        "method": method,
        "dataset": DATASET_NAME,
    }

    out_file = out_path / "ictal_check.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("PHASE 4A: ICTAL EPOCH CHECK")
    print("=" * 60)
    print("Dataset:          {}".format(DATASET_NAME))
    print("Consent status:   {}".format(consent_status))
    print("Search method:    {}".format(method))
    print("Ictal found:      {}".format(ictal_found))
    print("Annotations:      {}".format(len(ictal_epochs)))

    if ictal_found:
        # Deduplicate keyword matches
        keywords_found = sorted(set(e.get("keyword_match", "") for e in ictal_epochs))
        print("Keywords matched: {}".format(", ".join(keywords_found)))
    else:
        print("\nNo ictal annotations found.")
        print("Phase 4 differential analysis requires ictal epochs.")
        print("Options: (1) Inspect NWB files directly, (2) Use clinical notes.")

    print("\nResults saved: {}".format(out_file))
    return result


if __name__ == "__main__":
    import sys
    dpath = sys.argv[1] if len(sys.argv) > 1 else None
    check_ictal(dandiset_path=dpath)
