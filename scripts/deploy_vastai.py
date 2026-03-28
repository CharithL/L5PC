#!/usr/bin/env python3
"""
Deploy Synthetic Reality v3 to vast.ai — one command.

Usage:
  1. pip install vastai
  2. vastai set api-key YOUR_KEY
  3. python scripts/deploy_vastai.py

This will:
  - Find the cheapest GPU instance (>=8GB VRAM, <$0.20/hr)
  - Create the instance
  - Upload data + script
  - Run the experiment
  - Download results
  - Destroy the instance

Estimated cost: ~$0.10-0.30 total (30-60 min on GPU)
"""

import subprocess
import sys
import json
import time
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
KYZAR_DATA = BASE_DIR.parent / "Kyzar" / "data" / "kyzar_processed" / "session_sub5_ses2"
SCRIPT = BASE_DIR / "scripts" / "run_synthetic_reality_v3_iaaft.py"
RESULTS_DIR = BASE_DIR / "results" / "synthetic_reality_v3"


def run(cmd, check=True):
    print(f"  > {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"  STDERR: {r.stderr}")
        sys.exit(1)
    return r.stdout.strip()


def main():
    print("=" * 60)
    print("DESCARTES v3 — vast.ai Auto-Deploy")
    print("=" * 60)

    # Check prereqs
    if not KYZAR_DATA.exists():
        print(f"ERROR: Data not found at {KYZAR_DATA}")
        sys.exit(1)
    if not SCRIPT.exists():
        print(f"ERROR: Script not found at {SCRIPT}")
        sys.exit(1)

    # Find cheapest offer
    print("\n1. Searching for cheapest GPU...")
    offers = run(
        'vastai search offers "gpu_ram>=8 num_gpus=1 dph<0.25 inet_down>200 '
        'cuda_vers>=12.0 reliability>0.95" -o dph --raw'
    )
    try:
        offer_list = json.loads(offers)
    except json.JSONDecodeError:
        print("No offers found or vastai not configured. Run: vastai set api-key YOUR_KEY")
        sys.exit(1)

    if not offer_list:
        print("No GPU offers matching criteria. Try relaxing price/specs.")
        sys.exit(1)

    best = offer_list[0]
    print(f"  Best offer: {best.get('gpu_name', '?')} "
          f"${best.get('dph_total', '?'):.3f}/hr "
          f"({best.get('gpu_ram', '?')}GB VRAM)")

    offer_id = best['id']

    # Create instance
    print(f"\n2. Creating instance (offer {offer_id})...")
    result = run(
        f'vastai create instance {offer_id} '
        f'--image pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime '
        f'--disk 20 --raw'
    )
    instance_info = json.loads(result)
    instance_id = instance_info.get('new_contract')
    print(f"  Instance ID: {instance_id}")

    # Wait for instance to be ready
    print("\n3. Waiting for instance to start...")
    for attempt in range(60):
        time.sleep(10)
        status = run(f'vastai show instance {instance_id} --raw')
        info = json.loads(status)
        state = info.get('actual_status', 'unknown')
        print(f"  Status: {state} ({(attempt+1)*10}s)")
        if state == 'running':
            ssh_host = info.get('ssh_host', '')
            ssh_port = info.get('ssh_port', '')
            print(f"  SSH: {ssh_host}:{ssh_port}")
            break
    else:
        print("  Timeout waiting for instance. Destroying...")
        run(f'vastai destroy instance {instance_id}', check=False)
        sys.exit(1)

    # Upload via SCP
    print("\n4. Uploading data and script...")
    ssh_target = f"root@{ssh_host}"
    scp_opts = f"-P {ssh_port} -o StrictHostKeyChecking=no"

    run(f'ssh -p {ssh_port} -o StrictHostKeyChecking=no {ssh_target} '
        f'"mkdir -p /workspace/descartes/data/kyzar_processed/session_sub5_ses2"')
    run(f'scp {scp_opts} "{SCRIPT}" {ssh_target}:/workspace/descartes/')

    # Upload data files
    for fname in ['X_trials.npz', 'Y_trials.npz', 'epoch_masks.npz', 'metadata.json']:
        fpath = KYZAR_DATA / fname
        if fpath.exists():
            run(f'scp {scp_opts} "{fpath}" '
                f'{ssh_target}:/workspace/descartes/data/kyzar_processed/session_sub5_ses2/')

    # Install deps and run
    print("\n5. Installing deps and running experiment...")
    remote_cmd = (
        'cd /workspace/descartes && '
        'pip install -q numpy scipy scikit-learn numba && '
        'python run_synthetic_reality_v3_iaaft.py '
        '--processed-dir /workspace/descartes/data/kyzar_processed '
        '--output-dir /workspace/descartes/results '
        '--source-subject 5 --hidden-dim 64 --n-surrogates 50 '
        '--device cuda 2>&1 | tee v3_results.log'
    )
    run(f'ssh -p {ssh_port} -o StrictHostKeyChecking=no {ssh_target} "{remote_cmd}"')

    # Download results
    print("\n6. Downloading results...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run(f'scp {scp_opts} {ssh_target}:/workspace/descartes/v3_results.log '
        f'"{RESULTS_DIR / "v3_vastai.log"}"')
    run(f'scp -r {scp_opts} {ssh_target}:/workspace/descartes/results/ '
        f'"{RESULTS_DIR}/"', check=False)

    # Destroy instance
    print("\n7. Destroying instance to stop billing...")
    run(f'vastai destroy instance {instance_id}')

    print("\n" + "=" * 60)
    print("DONE! Results at:", RESULTS_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
