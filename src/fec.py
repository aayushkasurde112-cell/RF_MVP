"""
fec.py -- Blind FEC support: convolutional coding + Viterbi (Phase 5)
=====================================================================
(NTRO PS-26147)

Industry-standard rate-1/2, constraint-length-7 convolutional code with
generator polynomials 171o (1111001) and 133o (1011011) -- the CCSDS/IEEE
802.11 legacy mother code, and the default hypothesis for blind
reconstruction.  Generic over K and polynomials; tables are built once.

Conventions (shared with the Phase-7 synthetic generator):
  * register: input bit is the NEWEST bit (MSB side);
    out_j = parity(register_with_input & poly_j), emitted [out_0, out_1]
    per information bit (interleaved "G0,G1" framing);
  * tail=True zero-terminates the register (K-1 tail bits) so the decoder
    traceback starts from the known all-zero state -- a standard, blind-
    friendly frame structure.

Viterbi implementation: ACS is vectorized over the 2^(K-1) states with
numpy per time step (add-compare-select via scatter-min), traceback via a
parent table.  Accepts hard bits (0/1) or soft values (floats in [0, 1],
interpreted as P(bit=1); cost = |r - b|), so a future LLR front-end drops
in without changing the decoder.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["conv_encode", "viterbi_decode", "code_tables", "POLY_DEFAULT"]

POLY_DEFAULT = (0o171, 0o133)          # K=7, r=1/2 (171, 133 octal)
_TABLE_CACHE: Dict = {}


def code_tables(K: int = 7, polys: Tuple[int, ...] = POLY_DEFAULT):
    """Next-state / output tables for a feed-forward convolutional code.

    Returns (next_state (S, 2), out_bits (S, 2, n_polys) uint8) with S = 2^(K-1).
    """
    key = (K, polys)
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]
    s_count = 1 << (K - 1)
    mask = s_count - 1
    nxt = np.zeros((s_count, 2), dtype=np.int32)
    out = np.zeros((s_count, 2, len(polys)), dtype=np.uint8)
    for s in range(s_count):
        for b in (0, 1):
            reg = (b << (K - 1)) | s          # new bit enters at the MSB
            nxt[s, b] = (reg >> 1) & mask     # oldest bit falls off
            for j, poly in enumerate(polys):
                x = reg & poly
                out[s, b, j] = bin(x).count("1") & 1
    _TABLE_CACHE[key] = (nxt, out)
    return nxt, out


def conv_encode(bits: np.ndarray, K: int = 7,
                polys: Tuple[int, ...] = POLY_DEFAULT,
                tail: bool = True) -> np.ndarray:
    """Convolutional encode; returns interleaved (T', n_polys) uint8.

    tail=True flushes the register with K-1 zeros -> T' = T + K - 1 rows.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel() & 1
    if tail:
        bits = np.concatenate([bits, np.zeros(K - 1, dtype=np.uint8)])
    nxt, out = code_tables(K, polys)
    n_polys = out.shape[2]
    coded = np.empty((bits.size, n_polys), dtype=np.uint8)
    state = 0
    for i, b in enumerate(bits):
        b = int(b)
        coded[i] = out[state, b]
        state = int(nxt[state, b])
    return coded


def viterbi_decode(coded: np.ndarray, K: int = 7,
                   polys: Tuple[int, ...] = POLY_DEFAULT,
                   tail: bool = True) -> np.ndarray:
    """Viterbi ML sequence decode.

    coded : (T, n_polys) array; hard bits (0/1 ints) or soft values
            (floats in [0, 1] = P(bit = 1); Euclidean metric, the hard-
            decision-compatible special case of an LLR metric).
    tail=True forces the traceback to start from the all-zero state
    (matching a zero-terminated encode) and strips the K-1 tail bits.

    Returns the information bit estimates (uint8).
    """
    c = np.asarray(coded, dtype=np.float64)
    if c.ndim == 1:                        # tolerate interleaved flat input
        c = c.reshape(-1, len(polys))
    T, n_polys = c.shape
    nxt, out = code_tables(K, polys)
    S = nxt.shape[0]

    # branch metrics: (T, S, 2) via table gather -- vectorized per step
    hard = np.array_equal(c, np.round(c)) and c.max() <= 1 and c.min() >= 0 \
        and np.issubdtype(np.asarray(coded).dtype, np.integer)
    bm = np.empty((T, S, 2), dtype=np.float64)
    outs = out.astype(np.float64)                     # (S, 2, n_polys)
    for b in (0, 1):
        # metric[t, s, b] = sum_p |c[t, p] - out_p(s, b)|  -> broadcast over T,S
        bm[:, :, b] = np.abs(c[:, None, :] - outs[None, :, b, :]).sum(axis=2)
    if hard:
        bm = np.rint(bm)                              # exact Hamming metric

    INF = np.inf
    cost = np.full(S, INF)
    cost[0] = 0.0
    bit_arr = np.zeros((T, S), dtype=np.uint8)   # winning input bit per (t, state)
    sel_arr = np.zeros((T, S), dtype=np.uint8)   # winning predecessor selector (0=p0, 1=p1)

    # For input b, next state ns = b<<(K-2) | (p>>1) is reached from exactly
    # two source states p0 = ((ns & hi_mask) << 1) and p0 + 1 -- BOTH carry
    # the same input bit b, so the ACS must store the branch bit AND which
    # predecessor won (storing take1 as the "parent bit" is a classic bug).
    hi_mask = (1 << (K - 2)) - 1
    p0 = ((np.arange(S) & hi_mask) << 1).astype(np.int64)
    ns_idx = np.arange(S)

    for t in range(T):
        cost_prev = cost
        cost = np.full(S, INF)
        for b in (0, 1):
            c0 = cost_prev[p0] + bm[t, p0, b]
            c1 = cost_prev[p0 + 1] + bm[t, p0 + 1, b]
            take1 = c1 < c0                       # tie -> predecessor p0
            new_cost = np.where(take1, c1, c0)
            valid = (ns_idx >> (K - 2)) == b      # top state bit == input bit
            new_cost = np.where(valid, new_cost, INF)
            new_bit = np.where(valid, b, 0).astype(np.uint8)
            new_sel = np.where(valid, take1, 0).astype(np.uint8)
            if b == 0:
                cost = new_cost
                bit_arr[t] = new_bit
                sel_arr[t] = new_sel
            else:
                better = new_cost < cost
                cost = np.where(better, new_cost, cost)
                bit_arr[t] = np.where(better, new_bit, bit_arr[t])
                sel_arr[t] = np.where(better, new_sel, sel_arr[t])

    # traceback: input bit + predecessor-pair selector fully determine the path
    state = 0 if tail else int(np.argmin(cost))
    info = np.empty(T, dtype=np.uint8)
    for t in range(T - 1, -1, -1):
        info[t] = bit_arr[t, state]
        state = ((state & hi_mask) << 1) + int(sel_arr[t, state])

    if tail:
        info = info[: T - (K - 1)]
    return info
