#!/usr/bin/env bash
set -euo pipefail

PT="pytest"
if [ -x ".venv/bin/pytest" ]; then
	PT=".venv/bin/pytest"
fi

"$PT" --cov=. --cov-report=html tests/
