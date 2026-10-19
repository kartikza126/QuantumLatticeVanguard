"""
Unit & Verification Test Suite (`tests/test_all.py`)
Supports both `pytest` (`pytest tests/`) and standard library `unittest` (`python -m unittest discover -s tests -v`).
"""

from __future__ import annotations

import unittest
import numpy as np

from src.attacks import brute_force_lwe_attack, kannan_lll_attack
from src.lwe_toy import (
    LWEParameters,
    center_mod,
    decrypt_bit,
    decrypt_bit_uncentered_buggy,
    decrypt_message,
    encrypt_bit,
    encrypt_message,
    exact_discrete_failure_rate,
    gaussian_elimination_mod_q,
    keygen,
    theoretical_failure_rate,
)
from src.ring_lwe_ntt import (
    NegacyclicNTT,
    SimplifiedKyberKEM,
    schoolbook_negacyclic_mul,
)
from src.shor_threat import (
    extract_factors_from_measurement,
    simulate_shor_statevector_exact,
)
from src.timing_attack import (
    exploit_timing_oracle_demo,
    verify_ciphertext_constant_time,
    verify_ciphertext_vulnerable,
)


class TestPostQuantumCryptographySuite(unittest.TestCase):
    """Verification test suite for Problem 10 (Track 6)."""

    def test_01_modular_centering_and_ai_bug_regression(self) -> None:
        """Verify `center_mod` fixes the uncentered modulo `% q` bug at B=1."""
        q = 3329
        self.assertEqual(center_mod(3326, q), -3)
        self.assertEqual(center_mod(3, q), 3)
        self.assertEqual(center_mod(0, q), 0)

        params = LWEParameters(n=32, m=64, q=3329, B=1)
        rng = np.random.default_rng(12345)
        pk, sk, _ = keygen(params, rng)

        buggy_bit0_failures = 0
        fixed_bit0_failures = 0
        trials = 400
        for _ in range(trials):
            ct0 = encrypt_bit(pk, 0, params, rng)
            if decrypt_bit_uncentered_buggy(sk, ct0, params) != 0:
                buggy_bit0_failures += 1
            if decrypt_bit(sk, ct0, params) != 0:
                fixed_bit0_failures += 1

        self.assertGreater(buggy_bit0_failures / trials, 0.25)
        self.assertEqual(fixed_bit0_failures, 0)

    def test_02_gaussian_elimination_zero_vs_nonzero_noise(self) -> None:
        """Verify Gaussian elimination recovers s at B=0 and fails at B >= 1."""
        rng = np.random.default_rng(2026)
        params_zero = LWEParameters(n=32, m=64, q=3329, B=0)
        (A0, b0), sk0, _ = keygen(params_zero, rng)
        s_hat0, rank0, _ = gaussian_elimination_mod_q(A0, b0, 3329)
        self.assertEqual(rank0, 32)
        np.testing.assert_array_equal(s_hat0, sk0)

        params_noisy = LWEParameters(n=32, m=64, q=3329, B=2)
        (A1, b1), sk1, _ = keygen(params_noisy, rng)
        s_hat1, _, _ = gaussian_elimination_mod_q(A1, b1, 3329)
        self.assertFalse(np.array_equal(s_hat1, sk1))

    def test_03_exact_discrete_pmf_vs_clt_and_empirical(self) -> None:
        """Verify discrete convolution PMF matches the analytical Gaussian formula."""
        self.assertEqual(exact_discrete_failure_rate(B=10, m=64, q=3329), 0.0)
        p_disc_150 = exact_discrete_failure_rate(B=150, m=64, q=3329)
        p_clt_150 = float(theoretical_failure_rate(150, m=64, q=3329))
        self.assertAlmostEqual(p_disc_150, p_clt_150, delta=0.005)

    def test_04_lwe_message_encryption_correctness(self) -> None:
        """Verify multi-bit LWE encryption/decryption at B=4."""
        rng = np.random.default_rng(31415)
        params = LWEParameters(n=32, m=64, q=3329, B=4)
        pk, sk, _ = keygen(params, rng)
        msg = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1]
        cts = encrypt_message(pk, msg, params, rng)
        dec = decrypt_message(sk, cts, params)
        self.assertEqual(dec, msg)

    def test_05_classical_attacks_bruteforce_and_lll(self) -> None:
        """Verify Brute Force and LLL recover the ternary secret on small n."""
        rng = np.random.default_rng(777)
        params = LWEParameters(n=6, m=12, q=257, B=3)
        (A, b), sk, _ = keygen(params, rng)

        s_bf, checked, _ = brute_force_lwe_attack(A, b, q=257, B=3)
        self.assertIsNotNone(s_bf)
        self.assertLessEqual(checked, 3 ** 6)
        np.testing.assert_array_equal(s_bf, sk)

        s_lll, _ = kannan_lll_attack(A, b, q=257, B=3, delta=0.75)
        self.assertIsNotNone(s_lll)
        np.testing.assert_array_equal(s_lll, sk)

    def test_06_negacyclic_ntt_and_kyber_kem_equivalence(self) -> None:
        """Verify NegacyclicNTT matches schoolbook multiplication and SimplifiedKyberKEM works."""
        rng = np.random.default_rng(8888)
        for n, q in [(64, 257), (256, 7681), (512, 12289)]:
            ntt = NegacyclicNTT(n=n, q=q)
            a = rng.integers(0, q, size=n, dtype=np.int64)
            b = rng.integers(0, q, size=n, dtype=np.int64)
            np.testing.assert_array_equal(ntt.multiply(a, b), schoolbook_negacyclic_mul(a, b, q))

        kem = SimplifiedKyberKEM(B=2)
        for _ in range(10):
            pk, sk = kem.keygen(rng)
            ct, k_orig = kem.encaps(pk, rng)
            k_dec = kem.decaps(sk, ct)
            np.testing.assert_array_equal(k_dec, k_orig)

    def test_07_timing_verification_and_oracle_recovery(self) -> None:
        """Verify constant-time XOR fix and timing oracle recovery."""
        rng = np.random.default_rng(55)
        u = rng.integers(0, 3329, size=256, dtype=np.int64)
        self.assertTrue(verify_ciphertext_vulnerable(u, u.copy()))
        self.assertTrue(verify_ciphertext_constant_time(u, u.copy()))

        for idx in (0, 127, 255):
            u_bad = u.copy()
            u_bad[idx] = (u_bad[idx] + 1) % 3329
            self.assertFalse(verify_ciphertext_vulnerable(u_bad, u))
            self.assertFalse(verify_ciphertext_constant_time(u_bad, u))

        secret_short = np.array([-1, 0, 1, 1, -1, 0, 1, -1], dtype=np.int64)
        rec, _ = exploit_timing_oracle_demo(secret_short, [-1, 0, 1], repeats_per_guess=25, batch_calls=25)
        np.testing.assert_array_equal(rec, secret_short)

    def test_08_shor_n15_a7_statevector_and_continued_fractions(self) -> None:
        """Verify 5-qubit Shor circuit for N=15, a=7 (period r=4 -> factors 3 and 5)."""
        _, probs_a7 = simulate_shor_statevector_exact(a=7)
        for even_b in ("000", "010", "100", "110"):
            self.assertAlmostEqual(probs_a7[even_b], 0.25, places=10)
        for odd_b in ("001", "011", "101", "111"):
            self.assertAlmostEqual(probs_a7[odd_b], 0.0, places=10)

        for nontrivial_b in ("010", "110"):
            res = extract_factors_from_measurement(nontrivial_b, N=15, a=7)
            self.assertEqual(res["candidate_period_r"], 4)
            self.assertEqual(res["factors"], [3, 5])


if __name__ == "__main__":
    unittest.main()
