"""
Track 6 (Problem 10): Lattice-Based Post-Quantum Cryptography From Scratch
Qiskit Fall Fest Hackathon 2026 — IISER Bhopal

Modules:
- lwe_toy: Regev LWE public-key encryption, GF(q) Gaussian elimination, and exact discrete + CLT noise theory.
- attacks: Classical cryptanalysis comparing exhaustive brute force O(3^n) vs. Kannan's embedding + LLL reduction.
- ring_lwe_ntt: Structured Ring-LWE polynomial arithmetic via Negacyclic NTT, FIPS 203 Kyber NTT, and KEM.
- timing_attack: Fujisaki-Okamoto ciphertext verification timing side-channel exploit and constant-time XOR fix.
- shor_threat: Shor's quantum period-finding circuit for N=15 (a=7 and a=11), IBM Eagle noise model, and RSA-2048 scaling.
"""

__version__ = "2.0.0"
