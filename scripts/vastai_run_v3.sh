#!/bin/bash
# =================================================================
# vast.ai launcher for Synthetic Reality v3
#
# Usage:
#   1. Install vastai CLI:  pip install vastai
#   2. Set API key:         vastai set api-key YOUR_KEY
#   3. Find a cheap GPU:    vastai search offers 'gpu_ram>=8 num_gpus=1 dph<0.20 inet_down>200 cuda_vers>=12.0' -o dph
#   4. Create instance:     vastai create instance INSTANCE_ID --image pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime --disk 20
#   5. SSH in and run this script
# =================================================================

set -e

echo "=== Setting up DESCARTES Synthetic Reality v3 on vast.ai ==="

# Install dependencies
pip install -q numpy scipy scikit-learn numba

# Create working directory
mkdir -p /workspace/descartes
cd /workspace/descartes

echo "=== Upload your data and script ==="
echo "From your LOCAL machine, run:"
echo ""
echo "  scp -P PORT scripts/run_synthetic_reality_v3_iaaft.py root@SSH_HOST:/workspace/descartes/"
echo "  scp -r Kyzar/data/kyzar_processed/session_sub5_ses2 root@SSH_HOST:/workspace/descartes/data/kyzar_processed/"
echo ""
echo "Then run:"
echo ""
echo "  python run_synthetic_reality_v3_iaaft.py \\"
echo "    --processed-dir /workspace/descartes/data/kyzar_processed \\"
echo "    --output-dir /workspace/descartes/results \\"
echo "    --source-subject 5 \\"
echo "    --hidden-dim 64 \\"
echo "    --n-surrogates 50 \\"
echo "    --device cuda 2>&1 | tee v3_results.log"
echo ""
echo "When done, download results:"
echo "  scp -P PORT root@SSH_HOST:/workspace/descartes/v3_results.log ."
echo "  scp -r -P PORT root@SSH_HOST:/workspace/descartes/results/ results/synthetic_reality_v3/"
