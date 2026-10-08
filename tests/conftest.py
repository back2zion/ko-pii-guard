import os
import sys

# Make tests/synthetic.py importable.
# Note: Guardrails sends telemetry unless ~/.guardrailsrc contains
# `enable_metrics=false`; CI writes that file before running tests.
sys.path.insert(0, os.path.dirname(__file__))
