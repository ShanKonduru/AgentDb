#!/usr/bin/env bash
set -euo pipefail

mkdir -p test_reports

PT="pytest"
if [ -x ".venv/bin/pytest" ]; then
	PT=".venv/bin/pytest"
fi

"$PT" --html=test_reports/report.html --self-contained-html tests/
