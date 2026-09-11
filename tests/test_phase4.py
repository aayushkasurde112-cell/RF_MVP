"""Phase-4 terminal test: demod slicing + GF(2) interleaver detection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src import demod, interleaver

rng = np.random.default_rng(4261)
failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


def ref_rank(m):
    """Brute-force GF(2) rank (test oracle)."""
    m = (m.copy() & 1).astype(np.uint8)
    rk = 0
    for c in range(m.shape[1]):
        piv = next((rr for rr in range(rk, m.shape[0]) if m[rr, c]), None)
        if piv is None:
            continue
        m[[rk, piv]] = m[[piv, rk]]
        for rr in range(m.shape[0]):
            if rr != rk and m[rr, c]:
                m[rr] ^= m[rk]
        rk += 1
    return rk


# ------------------------------------------------------------------ #
print("== demod.slice_hard: round-trip + EVM (sigma=0.12, ~17% EVM theory) ==")
# per-subtype BER expectations follow minimum-distance physics at this SNR:
# d_min/sigma: BPSK 2.0, QPSK 1.41, 16QAM 0.51, 8PSK 0.73/1.18, 64QAM 0.26
# per-subtype sigma scaled to d_min so every constellation is BER-clean;
# EVM theory: 100 * sqrt(2) * sigma (unit-power constellations)
SIGMA = {"BPSK": 0.12, "QPSK": 0.10, "8PSK": 0.055, "16QAM": 0.045, "64QAM": 0.022}
for sub, m in [("BPSK", 1), ("QPSK", 2), ("8PSK", 3), ("16QAM", 4), ("64QAM", 6)]:
    n_sym = 8192
    tx_bits = rng.integers(0, 2, n_sym * m, dtype=np.uint8)
    sym = demod.modulate(tx_bits, sub)
    sg = SIGMA[sub]
    noisy = sym + sg * (rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym))
    rx_bits, evm, diag = demod.slice_hard(noisy, sub)
    ber = float(np.mean(rx_bits[: tx_bits.size] != tx_bits))
    evm_th = 100 * (2 ** 0.5) * sg
    check(f"{sub} round-trip BER==0", ber == 0.0, f"(EVM {evm}%)")
    check(f"{sub} EVM ~ {evm_th:.1f}%", abs(evm - evm_th) < 0.2 * evm_th, f"(measured {evm}%)")
    check(f"{sub} bit count", diag["n_bits"] == n_sym * m and rx_bits.dtype == np.uint8)

bits1, _, _ = demod.slice_hard(np.array([1.0 + 0j]), "BPSK")
check("single-symbol slice", bits1.size == 1 and bits1.dtype == np.uint8)
try:
    demod.slice_hard(np.array([]), "QPSK")
    check("empty input raises", False)
except ValueError:
    check("empty input raises", True)

# ------------------------------------------------------------------ #
print("== interleaver: GF(2) rank vs brute-force oracle ==")
ok = True
for W, R in [(8, 20), (9, 20), (15, 20), (63, 20), (64, 20), (65, 20),
             (100, 30), (130, 90), (70, 80)]:
    mm = rng.integers(0, 2, (R, W), dtype=np.uint8)
    a, b = interleaver.gf2_rank(mm), ref_rank(mm)
    ok &= a == b
    if a != b:
        print(f"   MISMATCH W={W} R={R}: {a} != {b}")
check("packed Gauss-Jordan matches oracle (9 shapes)", ok)
# structured cases
check("rank(I16) == 16", interleaver.gf2_rank(np.eye(16, dtype=np.uint8)) == 16)
check("rank(outer) == 1",
      interleaver.gf2_rank(np.outer([1, 1, 1, 1], [1, 0, 1, 0]).astype(np.uint8)) == 1)
r4 = rng.integers(0, 2, (24, 4), dtype=np.uint8)
check("rank([r, r]) == 4", interleaver.gf2_rank(np.hstack([r4, r4])) == 4)
# GF(2) subtlety: 1-r == 1+r, so [r | 1-r] = [r|r] + all-ones column block
# -> rank 5 (the complement contributes one extra basis vector)
check("rank([r, 1-r]) == 5 (GF(2) complement subtlety)",
      interleaver.gf2_rank(np.hstack([r4, 1 - r4])) == 5)
# wide multi-word packing: 70 free columns needs >= 70 rows
wide = np.zeros((80, 130), dtype=np.uint8)
wide[:, :70] = rng.integers(0, 2, (80, 70))
wide[:, 70:] = wide[:, :60]
check("wide packed rank == 70 (80 rows)", interleaver.gf2_rank(wide) == 70)

print("== interleaver: rank-deficiency spike + reversal ==")
# Realistic regime (mirrors the E2E frame): 64 rows of 64-bit codewords with
# 32 free bits per row -> rank 32 at the true width 64 (deficiency 32);
# multiples of 64 saturate at full rank (min(rows, W) null), random widths
# are general-position full rank.
R, NCOL = 64, 64
info = rng.integers(0, 2, (R, 32), dtype=np.uint8)
cw = np.hstack([info, info[:, ::-1]])          # rank exactly 32
check("sanity: gf2 rank of cw == 32", interleaver.gf2_rank(cw) == 32)
stream = interleaver.block_interleave(cw.ravel(), R, NCOL)   # column-read
w, prof = interleaver.detect_block_width(stream, max_width=128, min_rows=8)
top = sorted(prof.items(), key=lambda kv: -kv[1]["deficiency"])[:3]
print("   profile top (W, rank, deficiency):",
      [(k, v["rank"], v["deficiency"]) for k, v in top])
check("spike at true width 64", w == 64,
      f"(rank {prof[w]['rank']}, deficiency {prof[w]['deficiency']})")
de = interleaver.reverse_block_transpose(stream, R, NCOL)
check("reversal restores codeword rows", np.array_equal(de.reshape(R, NCOL), cw))

print()
if failures:
    print("PHASE-4 TEST: FAILURES ->", failures)
    sys.exit(1)
print("PHASE-4 TEST: ALL PASS")
