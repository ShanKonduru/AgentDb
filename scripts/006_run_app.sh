#!/usr/bin/env bash
set -euo pipefail

if [ -x ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

exec streamlit run app.py