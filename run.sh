#!/bin/bash
# Run the multi-layer autoresearch system.
#
# Usage:
#   ./run.sh              # Full two-layer loop (features + params)
#   ./run.sh 3            # Layer 3 only (hyperparameter tuning)
#   ./run.sh 1            # Full stack (same as no argument)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LAYER="${1:-1}"

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install dependencies if needed
if [ ! -f ".venv/.installed" ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -e .
    touch .venv/.installed
fi

# Load .env
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

echo "============================================================"
echo "Multi-Layer Autoresearch"
echo "  Layer:   $LAYER"
echo "  Tickers: ${TICKERS:-default universe (10 stocks)}"
echo "  Model:   ${MODEL_ID:-gpt-5.4}"
echo "============================================================"

# Fetch FactSet data if credentials are set and cache is missing
if [ -n "$FACTSET_USER_ID" ] && [ -n "$FACTSET_API_KEY" ]; then
    echo "Checking FactSet data cache..."
    python fetch_factset.py
fi

LAYER="$LAYER" python orchestrator.py
