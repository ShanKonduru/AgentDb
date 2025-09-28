#!/usr/bin/env bash
set -euo pipefail

if [ -n "${VIRTUAL_ENV-}" ]; then
  deactivate
else
  echo "No active virtual environment found."
fi
