#!/bin/bash
set -euo pipefail
cd /workspace
exec /opt/asr-venv/bin/python -I -B /tests/evaluate.py
