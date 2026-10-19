"""
Module 1: Toy Learning With Errors (LWE) Public-Key Encryption & Exact Analytical Noise Trade-off
Covers Core Tasks 1, 2, and 4 of Problem 10 (Track 6: Post-Quantum Cryptography).

Key Mathematical Contributions:
1. Regev LWE over Z_q (n=32, m=64, q=3329) with ternary secrets s in {-1, 0, 1}^n.
2. Exact Modular Gaussian Elimination over GF(q) proving 100% secret recovery in < 10 ms when B=0,
   and uniform field error amplification ||s_hat - s||_infty ~ q/2 when B >= 1.
3. Exact Discrete Convolution PMF of accumulated noise E = sum_{i=1}^m e_i r_i alongside the
   Central Limit Theorem (CLT) Gaussian tail and modulo-q wrapped Gaussian analytical curve
   (computed via Python's built-in `math.erf` with zero external C/DLL dependencies).

WARNING: Pedagogical implementation for research and benchmarking only. Never use to protect real data.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def _normal_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function Phi(x) = P(Z <= x) for Z ~ N(0, 1),
    computed exactly via Python's built-in `math.erf` (avoids scipy DLL issues on Windows WDAC).
    """
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


@dataclass
class LWEParameters:
    """Parameters for Regev-style Toy LWE public-key encryption over Z_q."""
    n: int = 32          # Secret dimension
    m: int = 64          # Number of LWE equations (samples)
    q: int = 3329        # Prime modulus (exact prime used in NIST FIPS 203 ML-KEM / Kyber)
    B: int = 2           # Uniform discrete error bound: e_i in {-B, ..., +B}


def center_mod(x: np.ndarray | int, q: int) -> np.ndarray | int:
    """
    Map residues from [0, q-1] to the symmetric centered interval (-q/2, q/2].
    
    CRITICAL VERIFICATION FIX (documented in AI_USE_NOTE.md):
    Python's `% q` maps negative inner-product noise (e.g., -3) to `q - 3` (3326).
    Without centering around 0 prior to thresholding `|d| < q/4`, negative noise realizations
    on bit 0 wrap near q and fail ~32% of the time even at B=1.
    """
    r = np.mod(x, q)
    if np.ndim(r) == 0:
        val = int(r)
        return val - q if val > (q // 2) else val
    return np.where(r > (q // 2), r - q, r)


def keygen(
    params: LWEParameters,
    rng: np.random.Generator,
) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    """
    Generate an LWE keypair:
      - Secret key s in {-1, 0, 1}^n
      - Public matrix A in Z_q^{m x n} sampled uniformly at random
      - Error vector e in {-B, ..., B}^m sampled uniformly at random
      - Public vector b = (A @ s + e) mod q
    """
    A = rng.integers(0, params.q, size=(params.m, params.n), dtype=np.int64)
    s = rng.integers(-1, 2, size=params.n, dtype=np.int64)
    if params.B == 0:
        e = np.zeros(params.m, dtype=np.int64)
    else:
        e = rng.integers(-params.B, params.B + 1, size=params.m, dtype=np.int64)
    b = (A @ s + e) % params.q
    return (A, b), s, e


def encrypt_bit(
    pk: Tuple[np.ndarray, np.ndarray],
    bit: int,
    params: LWEParameters,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int]:
    """
    Encrypt a single bit mu in {0, 1} using subset-sum masking r in {0, 1}^m:
      u = A^T @ r mod q
      v = (b^T @ r + mu * floor(q / 2)) mod q
    """
    assert bit in (0, 1), f"Bit must be 0 or 1, got {bit}"
    A, b = pk
    r = rng.integers(0, 2, size=params.m, dtype=np.int64)
    u = (A.T @ r) % params.q
    v = int((np.dot(b, r) + bit * (params.q // 2)) % params.q)
    return u, v


def decrypt_bit(
    sk: np.ndarray,
    ct: Tuple[np.ndarray, int],
    params: LWEParameters,
) -> int:
    """
    Decrypt a single bit from ciphertext (u, v) using secret key s:
      d = center_q(v - s^T @ u) = center_q(e^T @ r + mu * floor(q / 2))
    Decode 0 if |d| < q/4 else 1.
    """
    u, v = ct
    raw_diff = (int(v) - int(np.dot(sk, u))) % params.q
    d_centered = int(center_mod(raw_diff, params.q))
    return 0 if abs(d_centered) < (params.q / 4.0) else 1


def decrypt_bit_uncentered_buggy(
    sk: np.ndarray,
    ct: Tuple[np.ndarray, int],
    params: LWEParameters,
) -> int:
    """
    Uncentered decryption function preserving the exact AI bug documented in AI_USE_NOTE.md.
    """
    u, v = ct
    d_uncentered = (int(v) - int(np.dot(sk, u))) % params.q
    return 0 if abs(d_uncentered) < (params.q / 4.0) else 1


def encrypt_message(
    pk: Tuple[np.ndarray, np.ndarray],
    message_bits: List[int] | np.ndarray,
    params: LWEParameters,
    rng: np.random.Generator,
) -> List[Tuple[np.ndarray, int]]:
    """Encrypt a multi-bit message bit-by-bit."""
    return [encrypt_bit(pk, int(bit), params, rng) for bit in message_bits]


def decrypt_message(
    sk: np.ndarray,
    ciphertexts: List[Tuple[np.ndarray, int]],
    params: LWEParameters,
) -> List[int]:
    """Decrypt a sequence of ciphertexts into bits."""
    return [decrypt_bit(sk, ct, params) for ct in ciphertexts]


def gaussian_elimination_mod_q(
    A: np.ndarray,
    b: np.ndarray,
    q: int,
) -> Tuple[np.ndarray, int, float]:
    """
    Exact Modular Gaussian Elimination over GF(q) using modular inverses `pow(pivot, -1, q)`.
    Returns (recovered_s_centered, rank, runtime_ms).
    """
    t0 = time.perf_counter()
    m, n = A.shape
    aug = np.zeros((m, n + 1), dtype=np.int64)
    aug[:, :n] = np.mod(A, q)
    aug[:, n] = np.mod(b, q)

    pivot_row = 0
    pivot_cols = []

    for col in range(n):
        if pivot_row >= m:
            break
        nonzero_rows = np.where(aug[pivot_row:, col] != 0)[0]
        if len(nonzero_rows) == 0:
            continue
        swap_row = pivot_row + int(nonzero_rows[0])
        if swap_row != pivot_row:
            aug[[pivot_row, swap_row]] = aug[[swap_row, pivot_row]]

        pivot_val = int(aug[pivot_row, col])
        inv_pivot = pow(pivot_val, -1, q)
        aug[pivot_row] = (aug[pivot_row] * inv_pivot) % q

        for r in range(m):
            if r != pivot_row and aug[r, col] != 0:
                factor = int(aug[r, col])
                aug[r] = (aug[r] - factor * aug[pivot_row]) % q

        pivot_cols.append((pivot_row, col))
        pivot_row += 1

    s_hat = np.zeros(n, dtype=np.int64)
    for r_idx, c_idx in pivot_cols:
        s_hat[c_idx] = aug[r_idx, n]

    s_hat_centered = center_mod(s_hat, q)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return s_hat_centered, len(pivot_cols), elapsed_ms


def exact_discrete_failure_rate(B: int, m: int = 64, q: int = 3329) -> float:
    """
    Compute the EXACT discrete probability P(|center_q(sum_{i=1}^m e_i r_i)| >= q/4)
    via m-fold discrete polynomial convolution of the single-sample PMF:
      P(e_i r_i = 0) = (B + 1) / (2B + 1)
      P(e_i r_i = k) = 1 / (2 * (2B + 1))  for k in {-B, ..., B} \\ {0}.
    """
    if B <= 0:
        return 0.0
    single_pmf = np.full(2 * B + 1, 1.0 / (2.0 * (2 * B + 1)), dtype=np.float64)
    single_pmf[B] = float(B + 1) / float(2 * B + 1)

    acc_pmf = np.array([1.0], dtype=np.float64)
    base_pmf = single_pmf.copy()
    exp = m
    while exp > 0:
        if exp & 1:
            acc_pmf = np.convolve(acc_pmf, base_pmf)
        base_pmf = np.convolve(base_pmf, base_pmf)
        exp >>= 1

    max_val = m * B
    support = np.arange(-max_val, max_val + 1, dtype=np.int64)
    centered_support = center_mod(support, q)
    fail_mask = np.abs(centered_support) >= (q / 4.0)
    return float(np.clip(np.sum(acc_pmf[fail_mask]), 0.0, 1.0))


def theoretical_failure_rate(B: int | np.ndarray, m: int = 64, q: int = 3329) -> np.ndarray | float:
    """
    Analytical modulo-q wrapped Gaussian failure probability:
      sigma_total = sqrt(m * B * (B + 1) / 6)
      P_fail(B) = 1 - sum_{k in Z} [Phi((k*q + q/4)/sigma_total) - Phi((k*q - q/4)/sigma_total)].
    """
    B_arr = np.asarray(B, dtype=np.float64)
    scalar_input = (B_arr.ndim == 0)
    B_flat = np.atleast_1d(B_arr)

    p_fail = np.zeros_like(B_flat, dtype=np.float64)
    for idx, b_val in enumerate(B_flat):
        if b_val <= 0:
            p_fail[idx] = 0.0
            continue
        sigma = math.sqrt(m * b_val * (b_val + 1.0) / 6.0)
        p_success = 0.0
        for k in range(-8, 9):
            upper = (k * q + q / 4.0) / sigma
            lower = (k * q - q / 4.0) / sigma
            p_success += (_normal_cdf(upper) - _normal_cdf(lower))
        p_fail[idx] = float(np.clip(1.0 - p_success, 0.0, 1.0))

    return float(p_fail[0]) if scalar_input else p_fail


def clt_unwrapped_failure_rate(B: int | np.ndarray, m: int = 64, q: int = 3329) -> np.ndarray | float:
    """
    Unwrapped single-interval Central Limit Theorem tail approximation:
      P_fail(B) = 2 * (1 - Phi((q / 4) / sigma_total)).
    """
    B_arr = np.asarray(B, dtype=np.float64)
    scalar_input = (B_arr.ndim == 0)
    B_flat = np.atleast_1d(B_arr)
    p_fail = np.zeros_like(B_flat, dtype=np.float64)

    for idx, b_val in enumerate(B_flat):
        if b_val <= 0:
            p_fail[idx] = 0.0
        else:
            sigma = math.sqrt(m * b_val * (b_val + 1.0) / 6.0)
            p_fail[idx] = float(2.0 * (1.0 - _normal_cdf((q / 4.0) / sigma)))

    return float(p_fail[0]) if scalar_input else p_fail


def run_noise_sweep_experiment(
    B_values: List[int] | None = None,
    n_seeds: int = 20,
    trials_per_seed: int = 250,
    n: int = 32,
    m: int = 64,
    q: int = 3329,
) -> Dict:
    """
    Sweep noise bound B across `n_seeds` fixed random seeds (20 seeds x 250 trials = 5,000 trials/point)
    and compare empirical failure rates (mean +/- 1 std) against:
      1. Exact discrete m-fold convolution PMF (`exact_discrete_at_B`)
      2. Analytical modulo-q wrapped Gaussian (`theoretical_wrapped_at_B`)
      3. Unwrapped CLT Gaussian tail (`theoretical_clt_at_B`)
    """
    if B_values is None:
        B_values = [0, 2, 8, 12, 25, 50, 75, 95, 110, 120, 140, 165, 190, 220, 250, 300, 350, 400, 450]

    mean_fail = []
    std_fail = []
    exact_discrete = []
    gauss_elim_success = []
    gauss_elim_l_inf_err = []

    for b_val in B_values:
        params = LWEParameters(n=n, m=m, q=q, B=b_val)
        seed_rates = []
        ge_succ = []
        ge_err = []

        for seed in range(1000, 1000 + n_seeds):
            rng = np.random.default_rng(seed + b_val * 100)
            pk, sk, _ = keygen(params, rng)

            s_hat, _, _ = gaussian_elimination_mod_q(pk[0][:n, :], pk[1][:n], q)
            exact_match = bool(np.array_equal(s_hat, sk))
            ge_succ.append(1.0 if exact_match else 0.0)
            ge_err.append(float(np.max(np.abs(center_mod(s_hat - sk, q)))))

            A, b_vec = pk
            bits = rng.integers(0, 2, size=trials_per_seed, dtype=np.int64)
            R = rng.integers(0, 2, size=(trials_per_seed, m), dtype=np.int64)
            U = (R @ A) % q
            V = (R @ b_vec + bits * (q // 2)) % q

            raw_diffs = (V - (U @ sk)) % q
            centered_diffs = center_mod(raw_diffs, q)
            decoded_bits = np.where(np.abs(centered_diffs) < (q / 4.0), 0, 1)
            seed_rates.append(float(np.mean(decoded_bits != bits)))

        mean_fail.append(float(np.mean(seed_rates)))
        std_fail.append(float(np.std(seed_rates, ddof=1)))
        exact_discrete.append(exact_discrete_failure_rate(b_val, m=m, q=q))
        gauss_elim_success.append(float(np.mean(ge_succ)))
        gauss_elim_l_inf_err.append(float(np.mean(ge_err)))

    q_small = 769
    B_small_vals = [0, 2, 5, 8, 12, 18, 25, 32, 40, 50, 65, 80, 100, 120]
    small_mean_fail = []
    small_std_fail = []
    for b_val in B_small_vals:
        p_small = LWEParameters(n=n, m=m, q=q_small, B=b_val)
        s_rates = []
        for seed in range(2000, 2000 + n_seeds):
            rng = np.random.default_rng(seed + b_val * 100)
            (A_s, b_s), sk_s, _ = keygen(p_small, rng)
            bits = rng.integers(0, 2, size=trials_per_seed, dtype=np.int64)
            R = rng.integers(0, 2, size=(trials_per_seed, m), dtype=np.int64)
            U = (R @ A_s) % q_small
            V = (R @ b_s + bits * (q_small // 2)) % q_small
            centered = center_mod((V - (U @ sk_s)) % q_small, q_small)
            decoded = np.where(np.abs(centered) < (q_small / 4.0), 0, 1)
            s_rates.append(float(np.mean(decoded != bits)))
        small_mean_fail.append(float(np.mean(s_rates)))
        small_std_fail.append(float(np.std(s_rates, ddof=1)))

    b_dense = np.linspace(0, max(B_values), 300)
    b_small_dense = np.linspace(0, 120, 200)

    return {
        "parameters": {"n": n, "m": m, "q": q, "n_seeds": n_seeds, "trials_per_seed": trials_per_seed},
        "B_values": B_values,
        "empirical_mean": mean_fail,
        "empirical_std": std_fail,
        "exact_discrete_at_B": exact_discrete,
        "theoretical_wrapped_at_B": [float(theoretical_failure_rate(b, m=m, q=q)) for b in B_values],
        "theoretical_clt_at_B": [float(clt_unwrapped_failure_rate(b, m=m, q=q)) for b in B_values],
        "gaussian_elim_recovery_rate": gauss_elim_success,
        "gaussian_elim_mean_linf_error": gauss_elim_l_inf_err,
        "B_dense": b_dense.tolist(),
        "theoretical_wrapped_dense": theoretical_failure_rate(b_dense, m=m, q=q).tolist(),
        "theoretical_clt_dense": clt_unwrapped_failure_rate(b_dense, m=m, q=q).tolist(),
        "q769_inset": {
            "q": q_small,
            "B_values": B_small_vals,
            "empirical_mean": small_mean_fail,
            "empirical_std": small_std_fail,
            "B_dense": b_small_dense.tolist(),
            "theoretical_wrapped_dense": theoretical_failure_rate(b_small_dense, m=m, q=q_small).tolist(),
        },
    }
