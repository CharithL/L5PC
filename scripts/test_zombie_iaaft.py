#!/usr/bin/env python3
"""
Test the ORIGINAL zombie subjects (sub-10, sub-11) with unified iAAFT screening.
Uses pre-existing hidden states from Kyzar/models/kyzar/ — NO retraining.

Sub-10: lstm_h64, CC=0.319, March 18 checkpoint
Sub-11: lstm_h32, CC=0.542, March 18 checkpoint
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

from scripts.run_kyzar_phase3_4 import (
    load_session, run_phase3_iaaft, run_phase4_iaaft
)

KYZAR_DATA = "C:/Users/chari/OneDrive/Documents/Descartes_Cogito/Kyzar/data/kyzar_processed"
KYZAR_MODELS = "C:/Users/chari/OneDrive/Documents/Descartes_Cogito/Kyzar/models/kyzar"

ZOMBIES = [
    (10, 64),  # sub-10: h64, CC=0.319
    (11, 32),  # sub-11: h32, CC=0.542
]

def extract_mandatory(p4_results):
    """Extract mandatory variables — same logic as _extract_mandatory in pipeline."""
    mandatory = []
    for vname, vresult in p4_results.items():
        if not isinstance(vresult, dict):
            continue
        abl = vresult.get('resample_ablation', vresult)
        if abl.get('overall_verdict') == 'MANDATORY':
            mandatory.append(vname)
    return sorted(mandatory)


for subject, hdim in ZOMBIES:
    log.info("=" * 70)
    log.info("ZOMBIE TEST: sub-%d (original LSTM h%d, NOT retrained)", subject, hdim)
    log.info("=" * 70)

    session_data = load_session(KYZAR_DATA, KYZAR_MODELS, subject, hidden_dim=hdim)
    if session_data is None:
        log.error("  Cannot load sub-%d h%d from %s", subject, hdim, KYZAR_MODELS)
        continue

    log.info("  Loaded: H=%s, %d bio vars, %d trials, CC=%.3f",
             session_data['H_trained'].shape,
             len(session_data['bio_names']),
             len(np.unique(session_data['trial_groups'])),
             session_data['model_info'].get('output_cc', 0))

    # Phase 3: iAAFT screening (same method as non-zombie subjects)
    log.info("  Running Phase 3 iAAFT screening (50 surrogates)...")
    p3 = run_phase3_iaaft(session_data, n_surrogates=50)

    # Count passes
    n_pass = sum(1 for v in p3.values()
                 if v.get('iaaft', {}).get('passes_screen', False))
    log.info("  Phase 3 result: %d/18 variables passed dual gate", n_pass)

    # Phase 4: resample ablation on survivors
    log.info("  Running Phase 4 iAAFT ablation...")
    p4 = run_phase4_iaaft(session_data, p3)

    mandatory = extract_mandatory(p4)
    log.info("")
    log.info("  RESULT for sub-%d (original h%d checkpoint):", subject, hdim)
    log.info("  MANDATORY: %s", ', '.join(mandatory) if mandatory else 'NONE (ZOMBIE CONFIRMED)')
    log.info("  Count: %d mandatory variables", len(mandatory))

    # Print screening details for all variables
    log.info("")
    log.info("  Variable screening details:")
    for vname in session_data['bio_names']:
        if vname in p3:
            ia = p3[vname].get('iaaft', {})
            r2 = ia.get('r2_trained', 0)
            p_val = ia.get('p_value', 1)
            passes = ia.get('passes_screen', False)
            status = "PASS" if passes else "FAIL"
            # Check ablation result using correct key
            if vname in p4:
                abl = p4[vname].get('resample_ablation', p4[vname])
                verdict = abl.get('overall_verdict', 'N/A')
                status += f" -> {verdict}"
            log.info("    %-25s R2=%.4f  p=%.3f  %s", vname, r2, p_val, status)

log.info("")
log.info("=" * 70)
log.info("ZOMBIE TEST COMPLETE")
log.info("=" * 70)
