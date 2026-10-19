"""
Module 3: Structured Ring-LWE, Fast Negacyclic NTT, & Key Size Comparison
Covers Stretch Goals 1 & 3 of Problem 10 (Track 6: Post-Quantum Cryptography).

Implements:
1. O(n^2) schoolbook negacyclic polynomial multiplication in R_q = Z_q[x] / (x^n + 1).
2. O(n log n) Negacyclic Number Theoretic Transform (NTT) for fast polynomial multiplication.
3. Simplified 256-bit Kyber-style Key Exchange over R_q = Z_q[x] / (x^256 + 1).
4. Key & Ciphertext Size Comparison against RSA-2048 and ECC P-256.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

from src.lwe_toy import center_mod


def find_primitive_2n_root(n: int, q: int) -> int:
    """Find the smallest primitive 2n-th root of unity psi in Z_q^* such that psi^n == -1 (mod q)."""
    assert (q - 1) % (2 * n) == 0, f"2n={2*n} must divide q-1={q-1}"
    exponent = (q - 1) // (2 * n)
    for g in range(2, q):
        psi = pow(g, exponent, q)
        if pow(psi, n, q) == q - 1:
            return psi
    raise ValueError(f"No primitive 2n-th root found for n={n}, q={q}")


def _bit_reverse_indices(n: int) -> np.ndarray:
    """Precompute bit-reversed permutation indices for length n = 2^k."""
    bits = int(np.log2(n))
    indices = np.arange(n, dtype=np.int64)
    rev = np.zeros(n, dtype=np.int64)
    for i in range(bits):
        rev = (rev << 1) | ((indices >> i) & 1)
    return rev


class NegacyclicNTT:
    """
    Fast O(n log n) Negacyclic Number Theoretic Transform over R_q = Z_q[x] / (x^n + 1).
    Uses twist factors psi^i (where psi^n == -1 mod q) so negacyclic polynomial multiplication
    becomes point-wise multiplication: INTT(NTT(a) * NTT(b)).
    """

    def __init__(self, n: int = 256, q: int = 7681):
        assert (n & (n - 1)) == 0, "n must be a power of 2"
        self.n = n
        self.q = q
        self.psi = find_primitive_2n_root(n, q)
        self.psi_inv = pow(self.psi, -1, q)
        self.omega = (self.psi * self.psi) % q
        self.omega_inv = pow(self.omega, -1, q)
        self.n_inv = pow(n, -1, q)
        self.bit_rev = _bit_reverse_indices(n)

        exponents = np.arange(n, dtype=np.int64)
        self.psi_powers = np.array([pow(self.psi, int(i), q) for i in exponents], dtype=np.int64)
        self.psi_inv_powers = np.array([pow(self.psi_inv, int(i), q) for i in exponents], dtype=np.int64)

        self.forward_twiddles = []
        self.inverse_twiddles = []
        m = 2
        while m <= n:
            half = m // 2
            w_m = pow(self.omega, n // m, q)
            w_m_inv = pow(self.omega_inv, n // m, q)
            tw = np.array([pow(w_m, k, q) for k in range(half)], dtype=np.int64)
            tw_inv = np.array([pow(w_m_inv, k, q) for k in range(half)], dtype=np.int64)
            self.forward_twiddles.append(tw)
            self.inverse_twiddles.append(tw_inv)
            m *= 2

    def ntt(self, poly: np.ndarray) -> np.ndarray:
        """Forward negacyclic NTT in O(n log n)."""
        a = (np.asarray(poly, dtype=np.int64) * self.psi_powers) % self.q
        a = a[self.bit_rev].copy()

        m = 2
        stage = 0
        while m <= self.n:
            half = m // 2
            tw = self.forward_twiddles[stage]
            a_view = a.reshape(-1, m)
            u = a_view[:, :half].copy()
            t = (a_view[:, half:] * tw[None, :]) % self.q
            a_view[:, :half] = (u + t) % self.q
            a_view[:, half:] = (u - t) % self.q
            m *= 2
            stage += 1
        return a

    def intt(self, ntt_poly: np.ndarray) -> np.ndarray:
        """Inverse negacyclic NTT in O(n log n)."""
        a = np.asarray(ntt_poly, dtype=np.int64)[self.bit_rev].copy()

        m = 2
        stage = 0
        while m <= self.n:
            half = m // 2
            tw_inv = self.inverse_twiddles[stage]
            a_view = a.reshape(-1, m)
            u = a_view[:, :half].copy()
            t = (a_view[:, half:] * tw_inv[None, :]) % self.q
            a_view[:, :half] = (u + t) % self.q
            a_view[:, half:] = (u - t) % self.q
            m *= 2
            stage += 1

        a = (a * self.n_inv) % self.q
        a = (a * self.psi_inv_powers) % self.q
        return a

    def multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply two polynomials in R_q = Z_q[x]/(x^n + 1) via NTT."""
        return self.intt((self.ntt(a) * self.ntt(b)) % self.q)


def schoolbook_negacyclic_mul(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """
    O(n^2) schoolbook polynomial multiplication in R_q = Z_q[x] / (x^n + 1).
    Serves as the classical reference for verifying NTT multiplication.
    """
    n = len(a)
    a_arr = np.asarray(a, dtype=np.int64) % q
    b_arr = np.asarray(b, dtype=np.int64) % q
    res = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if i == 0:
            res = (res + a_arr[0] * b_arr) % q
        else:
            res[i:] = (res[i:] + a_arr[i] * b_arr[: n - i]) % q
            res[:i] = (res[:i] - a_arr[i] * b_arr[n - i :]) % q
    return res


class SimplifiedKyberKEM:
    """
    Simplified 256-bit Ring-LWE Key Exchange over R_q = Z_q[x] / (x^256 + 1)
    using NegacyclicNTT polynomial arithmetic.
    """

    def __init__(self, B: int = 2):
        self.n = 256
        self.q = 7681
        self.B = B
        self.ntt_engine = NegacyclicNTT(n=self.n, q=self.q)

    def _sample_small(self, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(-self.B, self.B + 1, size=self.n, dtype=np.int64)

    def keygen(self, rng: np.random.Generator) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        a = rng.integers(0, self.q, size=self.n, dtype=np.int64)
        s = self._sample_small(rng)
        e = self._sample_small(rng)
        b = (self.ntt_engine.multiply(a, s) + e) % self.q
        return (a, b), s

    def encaps(
        self,
        pk: Tuple[np.ndarray, np.ndarray],
        rng: np.random.Generator,
        shared_key_bits: np.ndarray | None = None,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        a, b = pk
        if shared_key_bits is None:
            shared_key_bits = rng.integers(0, 2, size=self.n, dtype=np.int64)
        else:
            shared_key_bits = np.asarray(shared_key_bits, dtype=np.int64)

        r = self._sample_small(rng)
        e1 = self._sample_small(rng)
        e2 = self._sample_small(rng)

        u = (self.ntt_engine.multiply(a, r) + e1) % self.q
        v = (self.ntt_engine.multiply(b, r) + e2 + shared_key_bits * (self.q // 2)) % self.q
        return (u, v), shared_key_bits

    def decaps(self, sk: np.ndarray, ct: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        u, v = ct
        su = self.ntt_engine.multiply(sk, u)
        diff = (v - su) % self.q
        centered = center_mod(diff, self.q)
        return np.where(np.abs(centered) < (self.q / 4.0), 0, 1).astype(np.int64)


def get_key_size_comparison_table() -> List[Dict]:
    """Compare key and ciphertext sizes (in bytes) across LWE, Ring-LWE, ML-KEM, RSA, and ECC."""
    return [
        {
            "scheme": "RSA-2048",
            "pk_bytes": 256,
            "sk_bytes": 256,
            "ct_bytes": 256,
            "quantum_safe": False,
        },
        {
            "scheme": "ECC P-256",
            "pk_bytes": 64,
            "sk_bytes": 32,
            "ct_bytes": 64,
            "quantum_safe": False,
        },
        {
            "scheme": "Toy LWE (n=32)",
            "pk_bytes": 3168,
            "sk_bytes": 8,
            "ct_bytes": 12672,
            "quantum_safe": False,
        },
        {
            "scheme": "Standard LWE (n=512)",
            "pk_bytes": 787968,
            "sk_bytes": 768,
            "ct_bytes": 196992,
            "quantum_safe": True,
        },
        {
            "scheme": "Ring-LWE (n=256)",
            "pk_bytes": 768,
            "sk_bytes": 384,
            "ct_bytes": 768,
            "quantum_safe": True,
        },
        {
            "scheme": "ML-KEM-768",
            "pk_bytes": 1184,
            "sk_bytes": 2400,
            "ct_bytes": 1088,
            "quantum_safe": True,
        },
    ]


def run_ntt_benchmark(
    degrees: List[int] | None = None,
    n_repeats: int = 25,
) -> Dict:
    """Benchmark O(n^2) schoolbook multiplication vs. O(n log n) NTT across n in {64..1024}."""
    if degrees is None:
        degrees = [64, 128, 256, 512, 1024]

    schoolbook_mean_ms = []
    schoolbook_std_ms = []
    ntt_mean_ms = []
    ntt_std_ms = []
    speedup_factors = []
    exact_matches = []

    for idx, n in enumerate(degrees):
        q = 257 if n == 64 else (7681 if n <= 256 else 12289)
        ntt = NegacyclicNTT(n=n, q=q)
        rng = np.random.default_rng(7000 + idx)

        sb_times = []
        ntt_times = []
        all_equal = True

        for _ in range(n_repeats):
            a = rng.integers(0, q, size=n, dtype=np.int64)
            b = rng.integers(0, q, size=n, dtype=np.int64)

            t0 = time.perf_counter()
            c_sb = schoolbook_negacyclic_mul(a, b, q)
            sb_times.append((time.perf_counter() - t0) * 1000.0)

            t1 = time.perf_counter()
            c_ntt = ntt.multiply(a, b)
            ntt_times.append((time.perf_counter() - t1) * 1000.0)

            if not np.array_equal(c_sb, c_ntt):
                all_equal = False

        sb_m = float(np.mean(sb_times))
        ntt_m = float(np.mean(ntt_times))
        schoolbook_mean_ms.append(sb_m)
        schoolbook_std_ms.append(float(np.std(sb_times, ddof=1)))
        ntt_mean_ms.append(ntt_m)
        ntt_std_ms.append(float(np.std(ntt_times, ddof=1)))
        speedup_factors.append(sb_m / max(ntt_m, 1e-6))
        exact_matches.append(all_equal)

    kem = SimplifiedKyberKEM(B=2)
    rng_kem = np.random.default_rng(2026)
    bit_errors = 0
    total_bits = 0
    for _ in range(100):
        pk, sk = kem.keygen(rng_kem)
        ct, k_orig = kem.encaps(pk, rng_kem)
        k_rec = kem.decaps(sk, ct)
        bit_errors += int(np.sum(k_orig != k_rec))
        total_bits += kem.n

    return {
        "degrees": degrees,
        "schoolbook_mean_ms": schoolbook_mean_ms,
        "schoolbook_std_ms": schoolbook_std_ms,
        "ntt_mean_ms": ntt_mean_ms,
        "ntt_std_ms": ntt_std_ms,
        "speedup_factors": speedup_factors,
        "exact_algebraic_match": exact_matches,
        "kyber_kem_256bit_test": {
            "trials": 100,
            "total_bits_exchanged": total_bits,
            "bit_errors": bit_errors,
            "bit_error_rate": float(bit_errors) / float(total_bits),
        },
        "size_comparison_table": get_key_size_comparison_table(),
    }
