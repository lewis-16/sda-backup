#!/bin/bash
set -e
ENV_NAME=${1:-ravens}
echo "Creating conda env: $ENV_NAME (Python 3.8, for TensorFlow 2.3)"
conda create -n "$ENV_NAME" python=3.8 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
echo "Done. Activate with: conda activate $ENV_NAME"
