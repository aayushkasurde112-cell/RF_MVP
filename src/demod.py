"""
demod.py -- Constellation slicing (Phase 4, NTRO PS-26147)
==========================================================

Nearest-neighbor (minimum Euclidean distance) slicing of the synchronized
1-sample-per-symbol complex array produced by ``sync.synchronize_*``.

Outputs:
  * hard bits, using the module's FIXED bit labeling (the synthetic
    generator in Phase 7 modulates with the inverse map, so blind chain
    output == ground-truth bits by construction);
  * constellation EVM (rms distance to the nearest ideal point, normalized
    by ideal rms power) -- the physical-layer quality metric.

Labeling conventions (MSB first per symbol):
  BPSK : bit 0 -> +1, bit 1 -> -1
  QPSK : Gray-2 over phase index k in {0..3}, point = exp(j(pi/4 + k pi/2))
  8PSK : Gray-3 over phase index
  QAM  : Gray per axis on odd levels (-(L-1) .. +(L-1)), bits = [I-bits | Q-bits]
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["constellation_table", "modulate", "slice_hard"]


def _gray(i: int) -> int:
    """Binary-reflected Gray code of integer i."""
    return i ^ (i >> 1)


def _bits_of(value: int, width: int) -> np.ndarray:
    """MSB-first bit vector of `value` with `width` bits."""
    return np.array([(value >> (width - 1 - b)) & 1 for b in range(width)],
                    dtype=np.uint8)


def constellation_table(subtype: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    """(points (M,) complex, labels (M, m) uint8) for a modulation subtype.

    Unknown/None falls back to QPSK (the safest linear assumption).
    """
    sub = (subtype or "QPSK").upper()
    if sub == "BPSK":
        pts = np.array([1.0 + 0j, -1.0 + 0j])
        lab = np.array([[0], [1]], dtype=np.uint8)
    elif sub == "8PSK":
        pts = np.exp(2j * np.pi * np.arange(8) / 8)
        lab = np.array([_bits_of(_gray(k), 3) for k in range(8)], dtype=np.uint8)
    elif sub in ("16QAM", "64QAM"):
        n = 4 if sub == "16QAM" else 8          # levels per axis
        m = 2 if n == 4 else 3                  # bits per axis
        lv = (2.0 * np.arange(n) - (n - 1))     # -(n-1) .. (n-1) odd levels
        g = np.array([_bits_of(_gray(i), m) for i in range(n)], dtype=np.uint8)
        pts = np.array([complex(a, b) for a in lv for b in lv])
        pts = pts / np.sqrt(np.mean(np.abs(pts) ** 2))
        # bit order per symbol: [I-axis bits | Q-axis bits], level-major
        lab = np.array([np.concatenate([g[i], g[j]])
                        for i in range(n) for j in range(n)], dtype=np.uint8)
    else:  # QPSK (also covers 4PSK aliases)
        pts = np.exp(1j * (np.pi / 4 + np.arange(4) * np.pi / 2))
        lab = np.array([_bits_of(_gray(k), 2) for k in range(4)], dtype=np.uint8)
    return pts, lab


def modulate(bits: np.ndarray, subtype: Optional[str]) -> np.ndarray:
    """Inverse map: bit vector -> constellation points (len(bits) % m pads)."""
    pts, lab = constellation_table(subtype)
    m = lab.shape[1]
    n_sym = int(np.ceil(len(bits) / m))
    b = np.zeros(n_sym * m, dtype=np.uint8)
    b[: len(bits)] = np.asarray(bits, dtype=np.uint8).ravel()[: n_sym * m]
    b = b.reshape(n_sym, m)
    # symbol index = integer value of the label (labels enumerate the table)
    idx = (b * (1 << np.arange(m - 1, -1, -1, dtype=np.uint64))).sum(axis=1)
    # labels were generated in Gray order -> invert through the label table
    table_idx = {tuple(row): i for i, row in enumerate(lab)}
    return pts[[table_idx[tuple(row)] for row in b]]


def slice_hard(symbols: np.ndarray, subtype: Optional[str]) -> Tuple[np.ndarray, float, Dict]:
    """Nearest-neighbor slicing -> (hard bits uint8, EVM %, diagnostics).

    EVM (%) = 100 * sqrt(mean |r - d|^2 / mean |d|^2) with d the nearest
    ideal points -- i.e. rms error over rms reference, the standard
    modulation-quality figure (compare against the noise floor
    100 * 10^(-SNR/20)).
    """
    z = np.asarray(symbols, dtype=np.complex128).ravel()
    if z.size == 0:
        raise ValueError("empty symbol array")
    pts, lab = constellation_table(subtype)
    m = lab.shape[1]

    d_idx = np.argmin(np.abs(pts[None, :] - z[:, None]), axis=1)
    ideal = pts[d_idx]
    p_ref = float(np.mean(np.abs(ideal) ** 2))
    evm = 100.0 * float(np.sqrt(np.mean(np.abs(z - ideal) ** 2) / max(p_ref, 1e-12)))
    bits = lab[d_idx].reshape(-1)
    return bits, round(evm, 2), {
        "n_symbols": int(z.size),
        "bits_per_symbol": int(m),
        "n_bits": int(bits.size),
        "modulation": (subtype or "QPSK").upper(),
    }
