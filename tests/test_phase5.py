"""Phase-5 terminal test: Viterbi FEC + bitstream analysis (known dummy stream)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src import bitstream, fec

rng = np.random.default_rng(5526)
failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


# ------------------------------------------------------------------ #
print("== fec: K=7 [171,133] encode/Viterbi round trip ==")
info = rng.integers(0, 2, 500, dtype=np.uint8)
coded = fec.conv_encode(info)                       # 514 codeword rows (tail)
check("encoded length = n + K-1", coded.shape == (506, 2), f"got {coded.shape}")

# (a) clean channel
check("clean decode exact", np.array_equal(fec.viterbi_decode(coded), info))

# (b) hard-decision channel: 3% random coded-bit errors -> ML decode fixes all
err = coded.copy()
flip = rng.choice(err.size, size=int(0.03 * err.size), replace=False)
err.ravel()[flip] ^= 1
dec = fec.viterbi_decode(err)
check("3% channel decode exact", np.array_equal(dec, info),
      f"({flip.size} flipped coded bits)")

# (c) soft decision: BPSK voltages + AWGN (sigma=0.8 -> ~4 dB, way below uncoded)
soft_bits = 1.0 - 2.0 * coded.astype(np.float64)    # bit1 -> -1
soft = soft_bits + 0.6 * rng.standard_normal(soft_bits.shape)
soft_p = (1.0 - soft) / 2.0                          # P(bit=1)
dec_soft = fec.viterbi_decode(np.clip(soft_p, 0, 1))
raw_ber = float(np.mean((soft_p >= 0.5).astype(np.uint8) != coded))
check("soft decode (sigma=0.6) exact", np.array_equal(dec_soft, info),
      f"(raw hard BER before FEC: {raw_ber:.3f})")

# (d) generic K/polys: K=5 [35, 23]
info5 = rng.integers(0, 2, 200, dtype=np.uint8)
c5 = fec.conv_encode(info5, K=5, polys=(0o35, 0o23))
err5 = c5.copy()
err5.ravel()[rng.choice(err5.size, 12, replace=False)] ^= 1
check("K=5 [35,23] variant decode", np.array_equal(fec.viterbi_decode(err5, K=5, polys=(0o35, 0o23)), info5))

# ------------------------------------------------------------------ #
print("== bitstream: frame-period autocorrelation ==")
ASM = 0x1ACFFC1D                                     # CCSDS attached sync marker
asm_bits = np.array([(ASM >> (31 - i)) & 1 for i in range(32)], dtype=np.uint8)
# beacon-style: identical repeated frames -> deep autocorr dip at the period
body = rng.integers(0, 2, 500, dtype=np.uint8)
stream = np.concatenate([asm_bits, body] * 4)        # period = 532
period, z = bitstream.frame_period(stream, d_min=32, d_max=2048)
check("frame period == 532 (32+500)", period == 532, f"(z {z})")
rand1 = rng.integers(0, 2, 1000, dtype=np.uint8)
period_r, z_r = bitstream.frame_period(rand1, d_min=32, d_max=499)
check("random stream -> no period", period_r is None, f"(z {z_r})")

print("== bitstream: sliding 64-bit-window entropy boundary ==")
# structured tail (all-zero padding) after random payload -> entropy drop
seq = np.concatenate([rng.integers(0, 2, 800, dtype=np.uint8),
                      np.zeros(1200, dtype=np.uint8)])
edge, drop = bitstream.entropy_boundary(seq, window=64, step=4)
check("entropy boundary at ~800", 780 <= edge <= 830, f"(edge {edge}, drop {drop:.2f})")
_, h_rand = bitstream.entropy_profile(seq[:800], 64, 64)
_, h_zero = bitstream.entropy_profile(seq[800:], 64, 64)
check("random segment H ~ 1.0", h_rand.mean() > 0.9, f"({h_rand.mean():.3f})")
check("padding segment H ~ 0.0", h_zero.mean() < 0.1, f"({h_zero.mean():.3f})")

print("== bitstream: sync search + CRC-16 dictionary sweep ==")
offs = bitstream.find_sync(stream, asm_bits, tol=1)
check("ASM found at 0/532/1064/1596", offs == [0, 532, 1064, 1596], f"({offs})")

payload = bytes(rng.integers(0, 256, 64, dtype=np.uint8))
for algo in ("CRC-16/CCITT-FALSE", "CRC-16/XMODEM", "CRC-16/MODBUS", "CRC-16/X-25"):
    frame = payload + bitstream.crc16(payload, algo).to_bytes(2, "big")
    hits = bitstream.crc_sweep(frame)
    check(f"crc sweep identifies {algo}", (algo, "big") in hits, f"({hits})")
bad = payload + b"\x00\x00"
check("crc sweep rejects corrupt frame", bitstream.crc_sweep(bad) == [])

print()
if failures:
    print("PHASE-5 TEST: FAILURES ->", failures)
    sys.exit(1)
print("PHASE-5 TEST: ALL PASS")
