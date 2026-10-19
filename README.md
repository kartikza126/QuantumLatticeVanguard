# From Shor’s Threat to Kyber’s Shield: Lattice Cryptography from Scratch

**Event:** Qiskit Fall Fest Hackathon 2026 · IISER Bhopal (EECS Club & Physics Club)  
**Track & Problem:** Track 6 — Post-Quantum Cryptography (**Problem 10: Lattice-based cryptography from scratch**)  
**Team Name:** Quantum Lattice Vanguard  
**Python Version:** `Python 3.11.2` (compatible with Python `3.10`–`3.14`)

> **Disclaimer:** Toy parameters are for learning only and must never be used to protect real data.

---

## Project Summary

Shor's quantum algorithm breaks RSA and ECC by finding the period of modular exponentiation in polynomial time. Lattice-based cryptography—standardized in NIST's ML-KEM (Kyber)—replaces factoring with **Learning With Errors (LWE)**: recovering a secret vector $\mathbf{s}$ from noisy linear equations $\mathbf{b} = A\mathbf{s} + \mathbf{e} \pmod q$. In this project, we implement Toy LWE ($n=32, m=64, q=3329$) from scratch, show that noiseless equations ($B=0$) are broken by modular Gaussian elimination in $9.4\text{ ms}$ ($100\%$ recovery) while small noise ($B \ge 1$) drops recovery to $0\%$, and verify decryption failure rates across 20 random seeds against our analytical Gaussian formula. We then compare classical Brute Force ($O(3^n)$) against Kannan-embedding LLL lattice reduction, demonstrate and fix a timing side-channel leak ($1.85\,\mu\text{s/coeff}$) using a constant-time XOR accumulator, speed up structured Ring-LWE polynomial multiplication by $15.7\times$ with the Number Theoretic Transform (NTT), and simulate a 5-qubit Shor factoring circuit for $N=15$ ($a=7 \implies r=4$) comparing noiseless simulation against a noisy IBM Eagle model ($91.9\%$ fidelity).

---

## Clean-Start Reproduction Commands

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Run unit tests (expected runtime: ~1-2 seconds)
pytest tests/

# 4. Run all benchmarks and generate all 3 figures (expected runtime: ~15-30 seconds)
python main.py

# 5. Run automated submission check
python -m src.check_submission
```

---

## Fixed Random Seeds & Statistical Runs

All experiments use fixed `numpy.random.default_rng` seeds for deterministic reproduction:
- **LWE Noise Sweep (`src/lwe_toy.py`):** Averages over **20 random seeds** (`1000..1019` for $q=3329$, `2000..2019` for $q=769$), $250$ trials per seed ($\pm 1\sigma$ error bars).
- **Brute Force vs. LLL (`src/attacks.py`):** Averages over **5 random seeds** per dimension $n \in [2, 18]$ (`seeds = 5000 + 100*n + trial`).
- **Ring-LWE NTT (`src/ring_lwe_ntt.py`):** Averages over **25 random polynomial pairs** per degree $n \in \{64, 128, 256, 512, 1024\}$ (`seeds = 7000..7004`).
- **Timing Leak & Fix (`src/timing_attack.py`):** Averages over **20 timing runs** per prefix length $k \in [0, 256]$ (`seed = 4242`).
- **Shor $N=15$ (`src/shor_threat.py`):** Uses **4,096 shots** (`seed = 1507` ideal, `seed = 20261015` noisy model).

---

## Repository Contents

| Path | Description |
| :--- | :--- |
| `README.md` | Team info, project summary, clean-start commands, expected runtime, and statistical seed info |
| `requirements.txt` | Pinned package versions (`package==x.y.z`) and Python version |
| `WRITEUP.pdf` | 4-page write-up covering Problem, Method, Results, Honest Limits, and AI-Use Note |
| `AI_USE_NOTE.md` | Details on AI tools used, the uncentered `% q` decryption bug, and how unit tests caught it |
| `main.py` | Main script running all experiments and generating all figures in `figures/` |
| `src/` | Source modules (`lwe_toy.py`, `attacks.py`, `ring_lwe_ntt.py`, `timing_attack.py`, `shor_threat.py`, `check_submission.py`) |
| `figures/` | All figures used in `WRITEUP.pdf` (`fig1_lwe_geometry_and_noise.png`, `fig2_attacks_and_timing.png`, `fig3_shor_and_ntt.png`) |
| `tests/` | Unit tests (`tests/test_all.py`) verifying results against exact classical references |
