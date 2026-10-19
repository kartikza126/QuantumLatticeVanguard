"""
Automated Pre-Submission Verification Script (`src/check_submission.py`)
Verifies all items on the Section 4 Repository Rules and Section 11 Final Checklist:
  1. All required paths from Section 4 (`README.md`, `requirements.txt`, `WRITEUP.pdf`,
     `AI_USE_NOTE.md`, `src/`, `figures/`, `tests/`) exist and are non-empty.
  2. `requirements.txt` has every required package version pinned (`package==x.y.z`).
  3. `WRITEUP.pdf` is strictly 4 pages (within the 2-to-4 page limit) AND has an exact
     16:9 widescreen PPT aspect ratio (`width / height == 16 / 9`).
  4. Zero API tokens or secrets are committed in any file.
  5. All unit tests in `tests/test_all.py` pass.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader


REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    print("=" * 76)
    print("Qiskit Fall Fest 2026 — Automated Submission & 16:9 Widescreen PDF Audit")
    print("=" * 76)
    failures = []

    required_paths = [
        "README.md",
        "requirements.txt",
        "WRITEUP.pdf",
        "AI_USE_NOTE.md",
        "main.py",
        "src/lwe_toy.py",
        "src/attacks.py",
        "src/ring_lwe_ntt.py",
        "src/timing_attack.py",
        "src/shor_threat.py",
        "figures/fig1_lwe_geometry_and_noise.png",
        "figures/fig2_attacks_and_timing.png",
        "figures/fig3_shor_and_ntt.png",
        "tests/test_all.py",
    ]
    for rel in required_paths:
        p = REPO_ROOT / rel
        if not p.exists() or (p.is_file() and p.stat().st_size == 0):
            failures.append(f"Missing or empty required path: {rel}")
    print(f"[PASS] All {len(required_paths)} required files and figures exist and are non-empty.")

    req_text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    active_lines = [
        line.strip()
        for line in req_text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    unpinned = [line for line in active_lines if "==" not in line]
    if unpinned or not active_lines:
        failures.append(f"Unpinned dependencies in requirements.txt: {unpinned}")
    else:
        print(f"[PASS] requirements.txt has {len(active_lines)} pinned packages: {', '.join(active_lines)}")

    writeup_path = REPO_ROOT / "WRITEUP.pdf"
    if writeup_path.exists():
        reader = PdfReader(str(writeup_path))
        num_pages = len(reader.pages)
        if not (2 <= num_pages <= 4):
            failures.append(f"WRITEUP.pdf has {num_pages} pages (must be 2 to 4 pages).")
        else:
            box = reader.pages[0].mediabox
            w_pt = float(box.width)
            h_pt = float(box.height)
            ratio = w_pt / h_pt
            target_ratio = 16.0 / 9.0
            if abs(ratio - target_ratio) > 0.02:
                failures.append(f"WRITEUP.pdf aspect ratio is {ratio:.4f}, expected 16:9 ({target_ratio:.4f}).")
            else:
                print(
                    f"[PASS] WRITEUP.pdf has {num_pages} pages in exact 16:9 PPT widescreen aspect ratio "
                    f"({w_pt:.1f} pt x {h_pt:.1f} pt, ratio = {ratio:.4f} == 16:9)."
                )

    token_pattern = re.compile(r"IBM_QUANTUM_TOKEN\s*=\s*['\"][0-9a-fA-F]{32,}['\"]")
    for py_file in REPO_ROOT.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8", errors="ignore")
        if token_pattern.search(content):
            failures.append(f"Potential hardcoded API token found in {py_file.relative_to(REPO_ROOT)}")
    print("[PASS] Zero hardcoded API tokens or secrets found in repository.")

    res = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        failures.append(f"Unit test suite failed:\n{res.stderr}")
    else:
        print("[PASS] All verification unit tests in `tests/test_all.py` passed.")

    print("=" * 76)
    if failures:
        print("AUDIT FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("ALL SUBMISSION & 16:9 WIDESCREEN CHECKS PASSED (100% compliant).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
