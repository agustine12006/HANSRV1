"""
HANSR Master Verification Test Suite — Dry Run across All Phases 1 to 7.
"""

import os
import sys
import time

print("=" * 70)
print("   HANSR MASTER VERIFICATION TEST SUITE (PHASES 1 - 7)")
print("=" * 70)

phases = [
    ("Phase 1: Environment & Scaffolding", "test_phase1.py"),
    ("Phase 2: Model Architecture & Constraints", "test_phase2.py"),
    ("Phase 3: Five-Term Composite Loss", "test_phase3.py"),
    ("Phase 4: Degradation Pipeline & Datasets", "test_phase4.py"),
    ("Phase 5: Training Pipeline & Checkpointing", "test_phase5.py"),
    ("Phase 6: Standalone Inference CLI & Metrics", "test_phase6.py"),
    ("Phase 7: Dashboard Validity & Isolation Gate", "test_phase7.py"),
]

passed_count = 0
start_time = time.time()

for name, script in phases:
    print(f"\n[RUN] {name} ({script})...")
    ret = os.system(f"python {script}")
    if ret == 0:
        print(f"[PASS] {name}")
        passed_count += 1
    else:
        print(f"[FAIL] {name} (exit code {ret})")
        sys.exit(1)

total_elapsed = time.time() - start_time
print("\n" + "=" * 70)
print(f"SUCCESS: All {passed_count}/{len(phases)} Phase Verifications Passed!")
print(f"Total Execution Time: {total_elapsed:.1f} seconds")
print("=" * 70)
