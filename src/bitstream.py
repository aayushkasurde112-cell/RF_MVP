"""
bitstream.py -- Blind bitstream analysis (Phase 5, NTRO PS-26147)
=================================================================

* **Frame-length estimation** via bitstream autocorrelation: the mismatch
  statistic p(d) = mean(x[i] != x[i+d]) dips sharply when d equals the
  period of any repeating structure (repeated sync words / ASM markers
  give p(frame_len) ~ 0).  Computed on the bipolar signal with a dot
  product, so it is one numpy correlation per lag.

* **Header/payload boundary** via sliding-window Shannon entropy (64-bit
  window): Bernoulli entropy H2 = -p log2 p - (1-p) log2(1-p) of each
  window.  Structured segments (ASCII headers, zero padding) have
  strongly biased bits -> low H2; random/compressed payload sits at
  H2 ~ 1.  The boundary is the steepest entropy edge.

* **CRC-16 dictionary sweep**: a registry of common CRC-16 variants
  (CCITT-FALSE, XMODEM, AUG-CCITT, IBM/ARC, MODBUS, USB, X-25, DNP),
  checked both byte orders against the trailing 2 bytes of a frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "frame_period",
    "entropy_profile",
    "entropy_boundary",
    "find_sync",
    "crc16",
    "CRC16_REGISTRY",
    "crc_sweep",
    "pack_bits_to_bytes",
    "unpack_bytes_to_bits",
]


# --------------------------------------------------------------------------- #
# Frame-period autocorrelation
# --------------------------------------------------------------------------- #
def frame_period(bits: np.ndarray, d_min: int = 8, d_max: int = 8192,
                 max_span: int = 65536, z_min: float = 5.0
                 ) -> Tuple[Optional[int], float]:
    """Estimate the repeating frame period from the mismatch autocorrelation.

    Statistic: for each lag d the mismatch rate p(d) over the overlapping
    span; under the no-structure null, p(d) ~ 0.5 with std
    0.5/sqrt(n-d), so the lag score is the z-score

        z(d) = (0.5 - p(d)) * sqrt(n - d) / 0.5

    (a repeated marker/frame dips p(d) BELOW 0.5 by its duty cycle).  The
    z-normalization matters: sync markers are a tiny duty cycle in long
    frames, so raw dips are indistinguishable from variance noise.
    Returns (period_bits or None, best_z); None when best_z < `z_min`.
    """
    x = (np.asarray(bits).astype(np.int64).ravel() & 1).astype(np.float64)
    n = min(x.size, max_span)
    x = 2.0 * x[:n] - 1.0                           # bipolar
    d_max = min(d_max, n // 2)                      # need >= 2 repetitions
    if d_max <= d_min:
        return None, 0.0
    best_d, best_z = None, 0.0
    for d in range(d_min, d_max + 1):
        seg = x[: n - d]
        matches = (float(np.dot(seg, x[d:])) + (n - d)) / 2.0
        p = 1.0 - matches / (n - d)
        z = (0.5 - p) * np.sqrt(n - d) / 0.5
        if z > best_z:
            best_z, best_d = z, d
    return (best_d if best_z >= z_min else None), round(float(best_z), 2)


# --------------------------------------------------------------------------- #
# Sliding-window Shannon entropy
# --------------------------------------------------------------------------- #
def entropy_profile(bits: np.ndarray, window: int = 64,
                    step: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Bernoulli Shannon entropy (bits/bit) over sliding windows.

    Vectorized with a cumulative sum of the bit stream; returns
    (centers (n,), H (n,)) with H in [0, 1].
    """
    x = (np.asarray(bits).astype(np.int64).ravel() & 1).astype(np.float64)
    if x.size < window:
        raise ValueError(f"need >= {window} bits, got {x.size}")
    c = np.concatenate([[0.0], np.cumsum(x)])
    starts = np.arange(0, x.size - window + 1, step)
    ones = c[starts + window] - c[starts]
    p = np.clip(ones / window, 1e-12, 1 - 1e-12)
    h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    centers = starts + window // 2
    return centers, h


def entropy_boundary(bits: np.ndarray, window: int = 64,
                     step: int = 4) -> Tuple[int, float]:
    """Locate the strongest entropy EDGE (header/payload boundary).

    Returns (bit_index_of_edge, entropy_drop) where the edge maximizes the
    downward step of the smoothed entropy profile -- the transition from
    random-like bits into structured bits (e.g. zero padding).
    """
    centers, h = entropy_profile(bits, window, step)
    kern = np.ones(5) / 5.0
    hs = np.convolve(h, kern, mode="same")
    grad = np.diff(hs)
    i = int(np.argmin(grad))                    # steepest drop
    edge = int(centers[min(i, centers.size - 1)])
    return edge, float(hs[i] - hs[i + 1])


# --------------------------------------------------------------------------- #
# Sync-word search
# --------------------------------------------------------------------------- #
def find_sync(bits: np.ndarray, pattern, tol: int = 2) -> List[int]:
    """Offsets where `pattern` (bit vector or int w/ width) matches within
    `tol` bit errors.  Standard markers (e.g. CCSDS ASM 0x1ACFFC1D) make
    frame-grid acquisition semi-blind: the marker is from a dictionary, its
    position is measured."""
    b = np.asarray(bits, dtype=np.uint8).ravel() & 1
    if isinstance(pattern, int):
        width = max(1, pattern.bit_length())
        pat = np.array([(pattern >> (width - 1 - i)) & 1
                        for i in range(width)], dtype=np.uint8)
    else:
        pat = np.asarray(pattern, dtype=np.uint8).ravel() & 1
    if pat.size == 0 or b.size < pat.size:
        return []
    # mismatch count via sliding correlation of the bipolar signals
    xb = 2.0 * b.astype(np.float64) - 1.0
    xp = 2.0 * pat.astype(np.float64) - 1.0
    corr = np.correlate(xb, xp, mode="valid")       # (L - w + 1,)
    mism = (pat.size - corr) / 2.0
    return [int(i) for i in np.flatnonzero(mism <= tol)]


# --------------------------------------------------------------------------- #
# CRC-16 registry + sweep
# --------------------------------------------------------------------------- #
def _crc16_bitwise(data: bytes, poly: int, init: int, refin: bool,
                   refout: bool, xorout: int, width: int = 16) -> int:
    crc = init
    for byte in data:
        if refin:
            byte = int("{:08b}".format(byte)[::-1], 2)
        crc ^= byte << (width - 8)
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    if refout:
        crc = int("{:016b}".format(crc)[::-1], 2)
    return crc ^ xorout


CRC16_REGISTRY: Dict[str, Dict] = {
    # name: poly, init, refin, refout, xorout  (standard catalogue values)
    "CRC-16/CCITT-FALSE": (0x1021, 0xFFFF, False, False, 0x0000),
    "CRC-16/XMODEM":      (0x1021, 0x0000, False, False, 0x0000),
    "CRC-16/AUG-CCITT":   (0x1021, 0x1D0F, False, False, 0x0000),
    "CRC-16/IBM":         (0x8005, 0x0000, True, True, 0x0000),
    "CRC-16/MODBUS":      (0x8005, 0xFFFF, True, True, 0x0000),
    "CRC-16/USB":         (0x8005, 0xFFFF, True, True, 0xFFFF),
    "CRC-16/X-25":        (0x1021, 0xFFFF, True, True, 0xFFFF),
    "CRC-16/DNP":         (0x3D65, 0x0000, True, True, 0xFFFF),
}


def crc16(data: bytes, algo: str = "CRC-16/CCITT-FALSE") -> int:
    poly, init, refin, refout, xorout = CRC16_REGISTRY[algo]
    return _crc16_bitwise(data, poly, init, refin, refout, xorout)


def crc_sweep(frame: bytes) -> List[Tuple[str, str]]:
    """Try every registry CRC against the trailing 2 bytes of `frame`,
    in both byte orders.  Returns [(algo, byteorder)] that validate."""
    if len(frame) < 3:
        return []
    data, tail = frame[:-2], frame[-2:]
    hits = []
    for algo in CRC16_REGISTRY:
        val = crc16(data, algo)
        for order, ref in (("big", val.to_bytes(2, "big")),
                           ("little", val.to_bytes(2, "little"))):
            if tail == ref:
                hits.append((algo, order))
    return hits


# --------------------------------------------------------------------------- #
# bit/byte packing helpers
# --------------------------------------------------------------------------- #
def pack_bits_to_bytes(bits: np.ndarray) -> bytes:
    """MSB-first bit -> bytes (tail zero-padded)."""
    b = np.asarray(bits, dtype=np.uint8).ravel() & 1
    pad = (-b.size) % 8
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(b).tobytes()


def unpack_bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))
