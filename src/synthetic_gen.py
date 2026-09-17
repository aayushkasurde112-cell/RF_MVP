"""Phase 7 -- synthetic reference IQ generator (ground truth chain).

Frame structure (one GROUP, repeated twice -- beacon-style so the blind
frame-period estimator sees a deep autocorrelation dip):

    [ASM(32)] [coded blk1] [ASM(32)] [coded blk2] [ASM(32)]

Supports multiple forward error correction schemes:
  - 'convolutional' (K=7, r=1/2 Viterbi default)
  - 'rs' (Reed-Solomon GF(2^8) e.g. RS(255, 223))
  - 'concatenated' (Outer RS + Inner Convolutional)
  - 'ldpc' (Quasi-cyclic / PEG Min-Sum LDPC)

Supports multiple interleaver schemes:
  - 'block' (Matrix row-write / column-read transpose)
  - 'convolutional' (Ramsey / Forney shift-register bank)
  - 'diagonal' (Diagonal write / row-read)
  - 'pseudo_random' / 'qpp' (Quadratic Permutation Polynomial)
  - 'prbs' (Seeded PRBS shuffler)
  - 'none' (Pass-through)

Waveform: BPSK or QPSK, RRC(alpha) pulse shaping at `sps` samples/symbol,
complex CFO tone, complex AWGN at a target SNR, fractional sample timing offset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    """ASCII header + random body + 0xAA padding, CRC-16 appended."""
    rng = np.random.default_rng(seed)
    n_rand = n_data_bytes - len(header) - pad_bytes
    if n_rand < 0:
        raise ValueError("n_data_bytes too small for header+padding")
    body = rng.integers(0, 256, n_rand, dtype=np.uint8).tobytes()
    data = header + body + bytes([0xAA]) * pad_bytes  # entropy-drop region
    crc = bitstream.crc16(data, CRC_ALGO)
    return data + crc.to_bytes(2, "big")              # 416 bytes


def apply_interleaver(bits: np.ndarray,
                      scheme: str = "block",
                      params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Apply the chosen interleaver scheme to a 1-D bit array."""
    scheme = (scheme or "block").lower()
    params = params or {}
    b = np.asarray(bits, dtype=np.uint8).ravel()

    import math

    if scheme in ("block", "matrix"):
        rows = params.get("rows", 64)
        cols = params.get("cols", 64)
        if rows * cols != b.size:
            s = math.isqrt(b.size)
            if s * s == b.size:
                rows, cols = s, s
            else:
                for r in (64, 32, 16, 8, 4):
                    if b.size % r == 0:
                        rows = r
                        cols = b.size // r
                        break
                else:
                    rows = 1
                    cols = b.size
        return interleaver.block_interleave(b, rows, cols)

    elif scheme in ("convolutional", "conv"):
        branches = params.get("branches", 8)
        delay = params.get("delay", 4)
        return interleaver.conv_interleave(b, branches=branches, delay=delay)

    elif scheme in ("diagonal", "diag"):
        rows = params.get("rows", 64)
        cols = params.get("cols", 64)
        if rows * cols != b.size:
            s = math.isqrt(b.size)
            if s * s == b.size:
                rows, cols = s, s
            else:
                for r in (64, 32, 16, 8, 4):
                    if b.size % r == 0:
                        rows = r
                        cols = b.size // r
                        break
                else:
                    rows = 1
                    cols = b.size
        return interleaver.diag_interleave(b, rows=rows, cols=cols)

    elif scheme in ("pseudo_random", "qpp"):
        n = params.get("n", b.size)
        f1 = params.get("f1", None)
        f2 = params.get("f2", None)
        return interleaver.qpp_interleave(b, n=n, f1=f1, f2=f2)

    elif scheme == "prbs":
        seed = params.get("seed", 42)
        n = params.get("n", b.size)
        return interleaver.prbs_interleave(b, n=n, seed=seed)

    elif scheme in ("none", "raw", "identity"):
        return b

    else:
        raise ValueError(f"Unknown interleaver scheme: {scheme}")


def _payload_to_coded_blocks(payload: bytes,
                             fec_scheme: str = "convolutional",
                             interleaver_scheme: str = "block",
                             fec_params: Optional[Dict[str, Any]] = None,
                             interleaver_params: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
    """Encode payload bytes and interleave into coded blocks."""
    fec_scheme = (fec_scheme or "convolutional").lower()
    fec_params = fec_params or {}
    interleaver_params = interleaver_params or {}

    if fec_scheme in ("convolutional", "viterbi", "conv"):
        bits = bitstream.unpack_bytes_to_bits(np.frombuffer(payload, np.uint8))
        need = BLOCKS_PER_GROUP * ROWS_PER_BLOCK * K_INFO
        if bits.size < need:
            bits = np.pad(bits, (0, need - bits.size))
        elif bits.size > need:
            bits = bits[:need]

        blocks = []
        for b in range(BLOCKS_PER_GROUP):
            seg = bits[b * ROWS_PER_BLOCK * K_INFO: (b + 1) * ROWS_PER_BLOCK * K_INFO]
            rows = seg.reshape(ROWS_PER_BLOCK, K_INFO)
            polys = tuple(fec_params.get("polys", fec.POLY_DEFAULT))
            K = fec_params.get("K", 7)
            coded = np.concatenate([fec.conv_encode(r, K=K, polys=polys) for r in rows])
            intl = apply_interleaver(coded, interleaver_scheme, interleaver_params)
            blocks.append(intl)
        return blocks

    elif fec_scheme in ("rs", "reed_solomon"):
        raw = np.frombuffer(payload, dtype=np.uint8)
        nsym = fec_params.get("nsym", 32)
        k_rs = 255 - nsym
        # Chunk payload into k_rs blocks
        blocks = []
        n_chunks = max(BLOCKS_PER_GROUP, (len(raw) + k_rs - 1) // k_rs)
        for i in range(n_chunks):
            chunk = raw[i * k_rs: (i + 1) * k_rs]
            if len(chunk) < k_rs:
                chunk = np.pad(chunk, (0, k_rs - len(chunk)), constant_values=0xAA)
            coded_bytes = fec.rs_encode(chunk, nsym=nsym)
            coded_bits = np.unpackbits(coded_bytes)
            intl = apply_interleaver(coded_bits, interleaver_scheme, interleaver_params)
            blocks.append(intl)
        return blocks

    elif fec_scheme in ("concatenated", "concat"):
        raw = np.frombuffer(payload, dtype=np.uint8)
        nsym = fec_params.get("rs_nsym", 32)
        k_rs = 255 - nsym
        chunk = raw[:k_rs]
        if len(chunk) < k_rs:
            chunk = np.pad(chunk, (0, k_rs - len(chunk)), constant_values=0xAA)
        coded_bits = fec.concat_encode(chunk, rs_nsym=nsym)
        # Partition into blocks if needed
        half = coded_bits.size // 2
        b1 = apply_interleaver(coded_bits[:half], interleaver_scheme, interleaver_params)
        b2 = apply_interleaver(coded_bits[half:], interleaver_scheme, interleaver_params)
        return [b1, b2]

    elif fec_scheme == "ldpc":
        n_ldpc = fec_params.get("n", 48)
        k_ldpc = fec_params.get("k", 24)
        H = fec_params.get("H", None)
        if H is None:
            H = fec.make_ldpc_parity_check(n=n_ldpc, k=k_ldpc)
        bits = bitstream.unpack_bytes_to_bits(np.frombuffer(payload, np.uint8))
        n_ldpc_blocks = (bits.size + k_ldpc - 1) // k_ldpc
        coded_parts = []
        for i in range(n_ldpc_blocks):
            sub = bits[i * k_ldpc: (i + 1) * k_ldpc]
            if sub.size < k_ldpc:
                sub = np.pad(sub, (0, k_ldpc - sub.size))
            cw = fec.ldpc_encode(sub, H)
            coded_parts.append(cw)
        all_coded = np.concatenate(coded_parts)
        half = all_coded.size // 2
        b1 = apply_interleaver(all_coded[:half], interleaver_scheme, interleaver_params)
        b2 = apply_interleaver(all_coded[half:], interleaver_scheme, interleaver_params)
        return [b1, b2]

    else:
        raise ValueError(f"Unknown FEC scheme: {fec_scheme}")


def generate(out_path: Union[str, Path], *,
             fs: float = 1.0e6,
             baud: float = 125.0e3,
             alpha: float = 0.35,
             cfo_hz: float = 7300.0,
             snr_db: float = 13.0,
             sps: int = 8,
             amp: float = 0.15,
             timing_off_samples: float = 0.31,
             seed: int = 11,
             modulation: str = "BPSK",
             fec_poly: Tuple[int, int] = (0o171, 0o133),
             fec_scheme: str = "convolutional",
             interleaver_scheme: str = "block",
             fec_params: Optional[Dict[str, Any]] = None,
             interleaver_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate synthetic capture with specified FEC & Interleaver schemes.

    Writes .iq file and beside it <stem>.truth.json.
    Returns ground-truth dict.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    payload = make_payload()
    blocks = _payload_to_coded_blocks(
        payload,
        fec_scheme=fec_scheme,
        interleaver_scheme=interleaver_scheme,
        fec_params=fec_params,
        interleaver_params=interleaver_params,
    )

    # Frame assembly: ASM + block_0 + ASM + block_1 + ASM ...
    frame_parts = [ASM_BITS]
    for blk in blocks:
        frame_parts.append(blk)
        frame_parts.append(ASM_BITS)
    group = np.concatenate(frame_parts)

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

    if abs(timing_off_samples) > 1e-9:
        n0 = int(np.floor(timing_off_samples))
        mu = timing_off_samples - n0
        x = np.concatenate([np.zeros(n0, complex), x])[: x.size]
        xp = np.concatenate([x[1:], [0.0]])
        x = (1.0 - mu) * x + mu * xp                  # 1st-order frac delay

    t = np.arange(x.size) / fs
    x *= np.exp(2j * np.pi * cfo_hz * t)              # carrier offset

    p_sig = float(np.mean(np.abs(x) ** 2))
    sigma = np.sqrt(p_sig / (2.0 * 10.0 ** (snr_db / 10.0)))
    x += sigma * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
    x = (amp * x).astype(np.complex64)
    x.tofile(out_path)

    truth: Dict[str, Any] = {
        "file": str(out_path),
        "fs_hz": fs, "baud_hz": baud, "sps": sps, "alpha": alpha,
        "cfo_hz": cfo_hz, "snr_db_target": snr_db,
        "modulation": mod_str, "samples": int(x.size),
        "group_bits": int(group.size), "total_bits": int(stream.size),
        "asm_hex": "1ACFFC1D", "crc_algo": CRC_ALGO,
        "fec_scheme": fec_scheme,
        "interleaver_scheme": interleaver_scheme,
        "fec_polynomials": list(fec_poly),
        "fec_params": fec_params or {},
        "interleaver_params": interleaver_params or {},
        "n_blocks_per_group": len(blocks),
        "rows_per_block": ROWS_PER_BLOCK, "k_info": K_INFO,
        "payload_hex": payload.hex(),
    }
    (out_path.with_suffix("")).with_name(
        out_path.stem + ".truth.json").write_text(json.dumps(truth, indent=2))
    return truth


if __name__ == "__main__":
    gt = generate("testdata/synthetic_bpsk.iq")
    print(f"wrote {gt['file']}: {gt['samples']} samples "
          f"({gt['samples'] / gt['fs_hz'] * 1e3:.1f} ms), "
          f"{gt['total_bits']} bits, group {gt['group_bits']} bits, "
          f"fec={gt['fec_scheme']}, interleaver={gt['interleaver_scheme']}")
