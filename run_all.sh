#!/usr/bin/env bash
# run_all.sh - full pipeline for macOS / Linux.
#   1. create the virtual environment (.venv) if it does not exist
#   2. install the pinned requirements
#   3. download the NASA .mat files if they are missing
#   4. run the four scripts in order
#   5. start the Streamlit dashboard (stop it with Ctrl+C)
# Usage:  bash run_all.sh
set -euo pipefail                 # stop at the first error
cd "$(dirname "$0")"              # always run from the project folder

# ---- pick a Python 3.12 or 3.13 interpreter (required by requirements.txt)
PY=""
for cand in python3.12 python3.13 python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && \
       "$cand" -c 'import sys; sys.exit(0 if (3,12) <= sys.version_info[:2] <= (3,13) else 1)'; then
        PY="$cand"; break
    fi
done
if [ -z "$PY" ]; then
    echo "ERROR: Python 3.12 or 3.13 is required (TensorFlow 2.21 does not support newer versions)."
    exit 1
fi
echo "Using $($PY --version)"

# ---- 1. virtual environment
if [ ! -d .venv ]; then
    echo "Creating virtual environment in .venv ..."
    "$PY" -m venv .venv
fi
source .venv/bin/activate

# ---- 2. dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# ---- 3. data (only if any of the four .mat files is missing)
mkdir -p data
if [ ! -f data/B0005.mat ] || [ ! -f data/B0006.mat ] || \
   [ ! -f data/B0007.mat ] || [ ! -f data/B0018.mat ]; then
    echo "Downloading NASA battery data ..."
    TMP="$(mktemp -d)"
    git clone --depth 1 https://github.com/anirudhkhatry/SOH-prediction-using-NASA-Dataset "$TMP/repo"
    cp "$TMP"/repo/B0005.mat "$TMP"/repo/B0006.mat "$TMP"/repo/B0007.mat "$TMP"/repo/B0018.mat data/
    rm -rf "$TMP"
fi

# ---- 4. pipeline
echo "=== Step 1: prepare_data.py ===";  python prepare_data.py
echo "=== Step 2: train_models.py ===";  python train_models.py
echo "=== Step 3: train_lstm.py (takes a few minutes) ==="; python train_lstm.py
echo "=== Step 4: make_plots.py ===";    python make_plots.py

# ---- 5. dashboard
# --server.headless true skips Streamlit's first-run e-mail prompt, which
# would otherwise wait for input. Open the URL below in your browser.
echo "=== Step 5: starting Streamlit - open http://localhost:8501 (Ctrl+C to stop) ==="
streamlit run app.py --server.headless true
