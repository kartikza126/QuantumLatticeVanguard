"""
Module 2: Classical Cryptanalysis — Exhaustive Brute Force O(3^n) vs. Pure-NumPy LLL Reduction
Covers Core Task 3 of Problem 10 (Track 6: Post-Quantum Cryptography).

Implements:
1. Vectorized exhaustive brute-force search over ternary secrets s in {-1, 0, 1}^n.
2. Kannan's primal embedding of LWE into the Shortest Vector Problem (SVP) solved via a
   self-contained, pure-NumPy Lenstra-Lenstra-Lovasz (LLL) lattice basis reduction algorithm.
"""

from __future__ import annotations

import itertools
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.lwe_toy import LWEParameters, center_mod, keygen


def brute_force_lwe_attack(
    A: np.ndarray,
    b: np.ndarray,
    q: int,
    B: int,
    batch_size: int = 8192,
    timeout_sec: float = 15.0,
) -> Tuple[Optional[np.ndarray], int, float]:
    """
    Exhaustive brute-force search over all 3^n ternary secret vectors s in {-1, 0, 1}^n.
    Checks whether the centered residual ||(b - A @ s') mod_c q||_infty <= B.
    """
    t0 = time.perf_counter()
    m, n = A.shape
    total_candidates = 3 ** n
    checked = 0

    iterator = itertools.product((-1, 0, 1), repeat=n)
    while checked < total_candidates:
        chunk = list(itertools.islice(iterator, batch_size))
        if not chunk:
            break
        S_batch = np.array(chunk, dtype=np.int64)
        checked += len(S_batch)

        residuals = (b[None, :] - (S_batch @ A.T)) % q
        residuals_centered = np.where(residuals > (q // 2), residuals - q, residuals)
        linf_norms = np.max(np.abs(residuals_centered), axis=1)

        matches = np.where(linf_norms <= B)[0]
        if len(matches) > 0:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return S_batch[int(matches[0])], checked, elapsed_ms

        if (time.perf_counter() - t0) > timeout_sec:
            break

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return None, checked, elapsed_ms


def lll_reduce(basis: np.ndarray, delta: float = 0.75, max_iters: int = 50000) -> np.ndarray:
    """
    Pure-NumPy implementation of the Lenstra-Lenstra-Lovasz (LLL) lattice basis reduction algorithm.
    Enforces Gram-Schmidt size reduction (|mu_{i,j}| <= 0.5) and Lovasz condition
    ||b*_k||^2 >= (delta - mu_{k,k-1}^2) ||b*_{k-1}||^2.
    """
    B_mat = np.array(basis, dtype=np.float64, copy=True)
    d = B_mat.shape[0]
    B_star = np.zeros_like(B_mat, dtype=np.float64)
    mu = np.zeros((d, d), dtype=np.float64)
    B_norm_sq = np.zeros(d, dtype=np.float64)

    def update_gso(start_row: int = 0) -> None:
        for i in range(start_row, d):
            vec = B_mat[i].copy()
            for j in range(i):
                if B_norm_sq[j] > 1e-12:
                    mu[i, j] = np.dot(B_mat[i], B_star[j]) / B_norm_sq[j]
                else:
                    mu[i, j] = 0.0
                vec -= mu[i, j] * B_star[j]
            B_star[i] = vec
            B_norm_sq[i] = max(float(np.dot(vec, vec)), 1e-12)

    update_gso(0)
    k = 1
    it = 0
    while k < d and it < max_iters:
        it += 1
        for j in range(k - 1, -1, -1):
            if abs(mu[k, j]) > 0.5:
                q_int = np.round(mu[k, j])
                B_mat[k] = np.round(B_mat[k] - q_int * B_mat[j])
                for l in range(j):
                    mu[k, l] -= q_int * mu[j, l]
                mu[k, j] -= q_int

        if B_norm_sq[k] >= (delta - mu[k, k - 1] ** 2) * B_norm_sq[k - 1]:
            k += 1
        else:
            B_mat[[k, k - 1]] = B_mat[[k - 1, k]]
            update_gso(max(k - 1, 0))
            k = max(k - 1, 1)

    return np.round(B_mat).astype(np.int64)


def kannan_lll_attack(
    A: np.ndarray,
    b: np.ndarray,
    q: int,
    B: int,
    secret_weight: Optional[int] = None,
    embedding_weight: Optional[int] = None,
    delta: float = 0.75,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Kannan's Primal Embedding attack reducing LWE to the Shortest Vector Problem (SVP).
    Constructs the (m + n + 1) x (m + n + 1) row-basis matrix M:
        [ q * I_m      0_{m x n}      0_{m x 1} ]
        [ A^T          w_s * I_n      0_{n x 1} ]
        [ b^T          0_{1 x n}          W     ]
    Because b - A @ s - q * k = e, the integer combination (-k^T, -s^T, 1) * M produces
    the short lattice vector v_short = (e^T, -w_s * s^T, W).
    """
    t0 = time.perf_counter()
    m, n = A.shape
    w_s = 1 if secret_weight is None else secret_weight
    W = 1 if embedding_weight is None else embedding_weight

    d = m + n + 1
    M = np.zeros((d, d), dtype=np.int64)
    M[:m, :m] = q * np.eye(m, dtype=np.int64)
    M[m:m + n, :m] = A.T
    M[m:m + n, m:m + n] = w_s * np.eye(n, dtype=np.int64)
    M[m + n, :m] = b
    M[m + n, m + n] = W

    reduced = lll_reduce(M, delta=delta)

    for row in reduced:
        last_val = int(row[m + n])
        if abs(last_val) == W:
            sign = 1 if last_val == W else -1
            mid = row[m:m + n]
            if np.all(mid % w_s == 0):
                s_cand = (-sign * (mid // w_s)).astype(np.int64)
                res = center_mod((b - A @ s_cand) % q, q)
                if np.max(np.abs(res)) <= B and np.max(np.abs(s_cand)) <= 1:
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    return s_cand, elapsed_ms

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return None, elapsed_ms


def run_attack_scaling_benchmark(
    dimensions: List[int] | None = None,
    q: int = 257,
    B: int = 11,
    n_trials: int = 5,
    max_bruteforce_n: int = 11,
) -> Dict:
    """
    Benchmark Classical Brute Force O(3^n) vs. Kannan's Embedding + LLL across dimensions n in [2, 18].
    """
    if dimensions is None:
        dimensions = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18]

    bf_times_ms = []
    bf_std_ms = []
    bf_success = []
    bf_is_measured = []

    lll_times_ms = []
    lll_std_ms = []
    lll_success = []

    measured_per_cand_ms = 0.0005

    for n in dimensions:
        m = n + 4
        params = LWEParameters(n=n, m=m, q=q, B=B)

        bf_trial_times = []
        bf_trial_succ = []
        lll_trial_times = []
        lll_trial_succ = []

        for trial in range(n_trials):
            rng = np.random.default_rng(5000 + n * 100 + trial)
            pk, sk, _ = keygen(params, rng)
            A, b_vec = pk

            s_lll, t_lll = kannan_lll_attack(
                A, b_vec, q=q, B=B, secret_weight=1, embedding_weight=1, delta=0.75
            )
            lll_trial_times.append(t_lll)
            lll_ok = s_lll is not None and np.array_equal(s_lll, sk)
            lll_trial_succ.append(1.0 if lll_ok else 0.0)

            if n <= max_bruteforce_n:
                s_bf, checked, t_bf = brute_force_lwe_attack(A, b_vec, q=q, B=B)
                bf_trial_times.append(t_bf)
                bf_ok = s_bf is not None and np.max(np.abs(center_mod((b_vec - A @ s_bf) % q, q))) <= B
                bf_trial_succ.append(1.0 if bf_ok else 0.0)
                if n >= 9 and checked > 0:
                    measured_per_cand_ms = t_bf / float(checked)

        lll_times_ms.append(float(np.mean(lll_trial_times)))
        lll_std_ms.append(float(np.std(lll_trial_times, ddof=1)) if n_trials > 1 else 0.0)
        lll_success.append(float(np.mean(lll_trial_succ)))

        if n <= max_bruteforce_n:
            bf_times_ms.append(float(np.mean(bf_trial_times)))
            bf_std_ms.append(float(np.std(bf_trial_times, ddof=1)) if n_trials > 1 else 0.0)
            bf_success.append(float(np.mean(bf_trial_succ)))
            bf_is_measured.append(True)
        else:
            est_ms = 0.5 * (3 ** n) * measured_per_cand_ms
            bf_times_ms.append(float(est_ms))
            bf_std_ms.append(0.0)
            bf_success.append(0.0)
            bf_is_measured.append(False)

    return {
        "parameters": {"q": q, "B": B, "n_trials": n_trials, "max_bruteforce_n": max_bruteforce_n},
        "dimensions": dimensions,
        "bruteforce_time_ms": bf_times_ms,
        "bruteforce_std_ms": bf_std_ms,
        "bruteforce_success_rate": bf_success,
        "bruteforce_is_measured": bf_is_measured,
        "lll_time_ms": lll_times_ms,
        "lll_std_ms": lll_std_ms,
        "lll_success_rate": lll_success,
    }
