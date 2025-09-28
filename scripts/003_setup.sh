#!/usr/bin/env bash
set -euo pipefail

PY="python3"
if [ -x ".venv/bin/python" ]; then
	PY=".venv/bin/python"
fi

"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements.txt
