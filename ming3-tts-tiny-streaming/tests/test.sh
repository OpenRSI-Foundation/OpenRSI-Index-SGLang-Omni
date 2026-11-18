#!/bin/bash
set -euo pipefail
cd /workspace
exec /opt/sglang/bin/python3 -I -B /tests/evaluate.py
