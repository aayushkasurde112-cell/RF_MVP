"""Phase 7 -- synthetic reference IQ generator (ground truth chain).

Frame structure (one GROUP, repeated twice -- beacon-style so the blind
frame-period estimator sees a deep autocorrelation dip):

    [ASM(32)] [coded blk1(4096)] [ASM(32)] [coded blk2(4096)] [ASM(32)]

Each 4096-bit coded block is a 64x64 bit block-interleaver transpose of
64 convolutional codewords: 26 payload bits -> K=7 r=1/2 [171o,133o]
encoder with tail -> 64 coded bits per row.  One group therefore carries
64*26*2 = 3328 payload bits = 416 bytes (414 data + CRC-16/CCITT-FALSE).

Waveform: BPSK (bit 0 -> +1, bit 1 -> -1 to match the Phase-4 Gray
slicer), RRC(alpha) pulse shaping at `sps` samples/symbol, complex CFO
tone, complex AWGN at a target SNR, fractional sample timing offset.

The generator returns a ground-truth dict (also written beside the IQ
file as <stem>.truth.json) that `main.py` blind-decodes and compares.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sync                                            # Phase 3 (rrc_taps)
from src import fec, interleaver, bitstream            # Phase 4/5

ASM_BITS = np.array([(0x1ACFFC1D >> (31 - i)) & 1
                     for i in range(32)], dtype=np.uint8)
CRC_ALGO = "CRC-16/CCITT-FALSE"
ROWS_PER_BLOCK = 64
BLOCKS_PER_GROUP = 2
K_INFO = 26                     # per codeword: 26 payload bits
CODEWORD_BITS = 64              # 2 * (26 + K-1) with K=7


def make_payload(n_data_bytes: int = 414, header: bytes = b"NTRO-SYNTH-V1",
                 seed: int = 2026, pad_bytes: int = 32) -> bytes:
    """ASCII header + random body + 0xAA padding, CRC-16 appended.

    The pad uses 0xAA (alternating bits) rather than 0x00: a zero-padded
    payload biases the bitstream (72 % zeros), which after BPSK mapping
    produces a discrete carrier line + baud-spaced clock lines that hijack
    the spectral detector.  0xAA keeps P(bit=1) = 0.5 (no DC bias) while
    remaining a near-zero-entropy segment for the entropy-boundary probe.
    """
    rng = np.random.default_rng(seed)
    n_rand = n_data_bytes - len(header) - pad_bytes
    if n_rand < 0:
        raise ValueError("n_data_bytes too small for header+padding")
    body = rng.integers(0, 256, n_rand, dtype=np.uint8).tobytes()
    data = header + body + bytes([0xAA]) * pad_bytes  # entropy-drop region
    crc = bitstream.crc16(data, CRC_ALGO)
    return data + crc.to_bytes(2, "big")              # 416 bytes


def _payload_to_coded_blocks(payload: bytes) -> list:
    """payload bits -> list of interleaved 4096-bit coded blocks."""
    bits = bitstream.unpack_bytes_to_bits(np.frombuffer(payload, np.uint8))
    need = BLOCKS_PER_GROUP * ROWS_PER_BLOCK * K_INFO
    if bits.size != need:
        raise ValueError(f"payload must be exactly {need} bits "
                         f"({need // 8} bytes), got {bits.size}")
    blocks = []
    for b in range(BLOCKS_PER_GROUP):
        seg = bits[b * ROWS_PER_BLOCK * K_INFO:
                   (b + 1) * ROWS_PER_BLOCK * K_INFO]
        rows = seg.reshape(ROWS_PER_BLOCK, K_INFO)
        coded = np.concatenate([fec.conv_encode(r) for r in rows])  # 64x64
        # block interleave = square bit transpose (matches Phase-4 module)
        blocks.append(interleaver.block_interleave(
            coded.reshape(ROWS_PER_BLOCK, CODEWORD_BITS),
            ROWS_PER_BLOCK, CODEWORD_BITS).ravel())
    return blocks


def generate(out_path: str | Path, *, fs: float = 1.0e6,
             baud: float = 125.0e3, alpha: float = 0.35,
             cfo_hz: float = 7300.0, snr_db: float = 13.0, sps: int = 8,
             amp: float = 0.15, timing_off_samples: float = 0.31,
             seed: int = 11, modulation: str = "BPSK",
             fec_poly: tuple[int, int] = (0o171, 0o133)) -> Dict[str, Any]:
    """Generate the synthetic capture + write <stem>.truth.json.

    Returns the ground-truth dict (payload hex of ONE group, parameters).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    payload = make_payload()
    blocks = _payload_to_coded_blocks(payload)
    # one group: ASM + blk1 + ASM + blk2 + ASM (ASMs delimit both blocks)
    group = np.concatenate([ASM_BITS, blocks[0], ASM_BITS,
                            blocks[1], ASM_BITS])
    # beacon repeat + leading/trailing guard bits (idle line): a real
    # receiver acquires mid-burst and crops edge symbols (matched-filter
    # transient + timing acquisition), so the outermost frame must be
    # protected by dead time or its edge blocks arrive incomplete.
    guard = np.zeros(256, dtype=np.uint8)
    stream = np.concatenate([guard, group, group, guard])

    # ------------------- Baseband modulation + impairments -----------------
    mod_str = modulation.upper()
    if mod_str == "QPSK":
        from src import demod
        if stream.size % 2 != 0:
            stream = np.concatenate([stream, [0]])
        sym = demod.modulate(stream, "QPSK")
    else:
        sym = 1.0 - 2.0 * stream.astype(np.float64)       # bit 0 -> +1 (BPSK)
    up = np.zeros(sym.size * sps, dtype=np.complex128)
    up[::sps] = sym
    taps = sync.rrc_taps(alpha, 8, float(sps))
    x = np.convolve(up, taps, mode="same")            # pulse shaping
    # fractional timing offset: linear-phase shift = frequency-domain ramp,
    # approximated here by a first-order fractional delay on the waveform
    if abs(timing_off_samples) > 1e-9:
        n0 = int(np.floor(timing_off_samples))
        mu = timing_off_samples - n0
        x = np.concatenate([np.zeros(n0, complex), x])[: x.size]
        xp = np.concatenate([x[1:], [0.0]])
        x = (1.0 - mu) * x + mu * xp                  # 1st-order frac delay
    t = np.arange(x.size) / fs
    x *= np.exp(2j * np.pi * cfo_hz * t)              # carrier offset
    # AWGN: sigma^2 per I/Q chosen so P_sig / mean|n|^2 = 10^(snr/10)
    p_sig = float(np.mean(np.abs(x) ** 2))
    sigma = np.sqrt(p_sig / (2.0 * 10.0 ** (snr_db / 10.0)))
    x += sigma * (rng.standard_normal(x.size)
                  + 1j * rng.standard_normal(x.size))
    x = (amp * x).astype(np.complex64)
    x.tofile(out_path)

    truth: Dict[str, Any] = {
        "file": str(out_path),
        "fs_hz": fs, "baud_hz": baud, "sps": sps, "alpha": alpha,
        "cfo_hz": cfo_hz, "snr_db_target": snr_db,
        "modulation": mod_str, "samples": int(x.size),
        "group_bits": int(group.size), "total_bits": int(stream.size),
        "asm_hex": "1ACFFC1D", "crc_algo": CRC_ALGO,
        "fec_polynomials": list(fec_poly),
        "n_blocks_per_group": BLOCKS_PER_GROUP,
        "rows_per_block": ROWS_PER_BLOCK, "k_info": K_INFO,
        "payload_hex": payload.hex(),           # ONE group (416 bytes)
    }
    (out_path.with_suffix("")).with_name(
        out_path.stem + ".truth.json").write_text(json.dumps(truth, indent=2))
    return truth


if __name__ == "__main__":
    gt = generate("testdata/synthetic_bpsk.iq")
    print(f"wrote {gt['file']}: {gt['samples']} samples "
          f"({gt['samples'] / gt['fs_hz'] * 1e3:.1f} ms), "
          f"{gt['total_bits']} bits, group {gt['group_bits']} bits")
