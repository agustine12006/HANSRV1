"""
HANSR Dashboard — Isolated Monitoring (FR-015)

This package is strictly observational. It must NEVER be imported by
evaluate.py or train.py. Dashboard failures must never propagate to
or block training/inference.
"""
