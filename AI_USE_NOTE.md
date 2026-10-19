# AI-Use Note (`AI_USE_NOTE.md`)

**Event:** Qiskit Fall Fest Hackathon 2026 — IISER Bhopal (Track 6: Post-Quantum Cryptography, Problem 10)  
**Team Name:** Quantum Lattice Vanguard  
**Project Title:** *From Shor’s Threat to Kyber’s Shield: Lattice Cryptography from Scratch*

---

## 1. AI Tools Used

We used an AI coding assistant to help draft module structure, docstrings, and plot formatting, and verified every formula and circuit with 8 unit tests in `tests/test_all.py`.

---

## 2. Specific Error the AI Made: Uncentered Modulo `% q` in LWE Decryption

In the first draft of `decrypt_bit` (in `src/lwe_toy.py`), the AI wrote:

```python
# BUGGY AI DRAFT (preserved as `decrypt_bit_uncentered_buggy` in src/lwe_toy.py):
d_uncentered = (v - np.dot(s, u)) % q
return 0 if abs(d_uncentered) < (q / 4.0) else 1
```

**Why this is wrong:**
When encrypting bit $\mu = 0$, the remainder $v - \mathbf{s}^T\mathbf{u} \equiv \mathbf{e}^T\mathbf{r} \pmod q$ is just the small inner-product noise $E = \sum_{i=1}^m e_i r_i$, which can be negative (for example, $-3$). In Python, `(-3) % 3329` returns `3326` (near $q$), not `-3`. Because `abs(3326) < 832.25` (`q / 4`) is `False`, the AI's code misclassified every negative-noise $\mu = 0$ ciphertext as bit `1`.

---

## 3. How We Caught and Fixed It

- **How we caught it:** When we ran our unit test (`test_01_modular_centering_and_ai_bug_regression` in `tests/test_all.py`) at $B = 1$—where the maximum possible noise is $64 \ll 832.25$ so the failure rate must be $0\%$—the uncentered draft failed with a **15.85% overall error rate (31.7% error rate on bit 0)**.
- **How we fixed it:** We centered residues into $(-q/2, q/2]$ before comparing against $q/4$:
  ```python
  def center_mod(x, q: int):
      r = np.mod(x, q)
      return np.where(r > (q // 2), r - q, r)
  ```
  With centering, the error rate at $B = 1$ dropped to **0.00% across 2,000 trials**.
