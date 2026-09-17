"""
fec.py -- Multi-scheme FEC: Convolutional/Viterbi + RS + Concatenated + LDPC
============================================================================
(NTRO PS-26147)

Supported schemes
-----------------
1. **Convolutional / Viterbi** (original) -- rate-1/2, K=7, [171o, 133o]
   CCSDS/IEEE 802.11 mother code.  Generic over K and polynomials.
2. **Reed-Solomon (RS) block codes** -- GF(2^m) arithmetic implemented
   from scratch using numpy (no external galois library required).
   Default: RS(255, 223, t=16) over GF(2^8) with the CCSDS generator.
3. **Concatenated codes** -- inner Viterbi + outer RS, the classic deep-
   space / DVB-S1 scheme.
4. **LDPC** -- Belief Propagation (Min-Sum) decoder for a configurable
   parity-check matrix.  Ships with a compact (48,24) regular LDPC code
   for verification; can load arbitrary H matrices.

Conventions:
  * All encoders accept uint8 bit or byte arrays, return uint8.
  * All decoders accept hard bits (uint8 0/1); soft-input is supported
    where noted.
  * RS works at the BYTE level (symbols in GF(2^8)); the bit-level
    interface packs/unpacks automatically.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "conv_encode", "viterbi_decode", "code_tables", "POLY_DEFAULT",
    "rs_encode", "rs_decode",
    "concat_encode", "concat_decode",
    "ldpc_encode", "ldpc_decode", "make_ldpc_parity_check",
]

# =========================================================================== #
#  1. Convolutional / Viterbi  (original, untouched)
# =========================================================================== #
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


# =========================================================================== #
#  2. Reed-Solomon over GF(2^8)  --  pure numpy implementation
# =========================================================================== #
# GF(2^8) with the standard CCSDS/AES primitive polynomial x^8+x^4+x^3+x+1 = 0x11D
_GF_EXP = np.zeros(512, dtype=np.int32)   # anti-log table
_GF_LOG = np.zeros(256, dtype=np.int32)   # log table
_GF_INIT = False


def _gf_init(prim: int = 0x11D):
    """Build GF(2^8) log/antilog tables with primitive polynomial `prim`."""
    global _GF_EXP, _GF_LOG, _GF_INIT
    if _GF_INIT:
        return
    x = 1
    for i in range(255):
        _GF_EXP[i] = x
        _GF_LOG[x] = i
        x <<= 1
        if x & 0x100:
            x ^= prim
    # Double the exp table for convenience in multiplication
    for i in range(255, 512):
        _GF_EXP[i] = _GF_EXP[i - 255]
    _GF_LOG[0] = -1  # log(0) is undefined; use sentinel
    _GF_INIT = True


def _gf_mul(a: int, b: int) -> int:
    """Multiply two GF(2^8) elements."""
    if a == 0 or b == 0:
        return 0
    return int(_GF_EXP[_GF_LOG[a] + _GF_LOG[b]])


def _gf_pow(a: int, n: int) -> int:
    """a^n in GF(2^8)."""
    if n == 0:
        return 1
    if a == 0:
        return 0
    return int(_GF_EXP[(_GF_LOG[a] * n) % 255])


def _gf_inv(a: int) -> int:
    """Multiplicative inverse in GF(2^8)."""
    if a == 0:
        raise ZeroDivisionError("inverse of zero in GF(2^8)")
    return int(_GF_EXP[255 - _GF_LOG[a]])


def _gf_poly_mul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Multiply two polynomials over GF(2^8)."""
    r = np.zeros(len(p) + len(q) - 1, dtype=np.int32)
    for i, pi in enumerate(p):
        if pi == 0:
            continue
        for j, qj in enumerate(q):
            if qj == 0:
                continue
            r[i + j] ^= _gf_mul(int(pi), int(qj))
    return r


def _gf_poly_eval(poly: np.ndarray, x: int) -> int:
    """Evaluate polynomial at x in GF(2^8) using Horner's method."""
    result = 0
    for coeff in poly:
        result = _gf_mul(result, x) ^ int(coeff)
    return result


def _rs_generator(nsym: int, fcr: int = 0) -> np.ndarray:
    """Generator polynomial for RS code with `nsym` check symbols.

    g(x) = prod_{i=fcr}^{fcr+nsym-1} (x - alpha^i)
    """
    g = np.array([1], dtype=np.int32)
    for i in range(fcr, fcr + nsym):
        g = _gf_poly_mul(g, np.array([1, _gf_pow(2, i)], dtype=np.int32))
    return g


def rs_encode(data: np.ndarray, nsym: int = 32, fcr: int = 0) -> np.ndarray:
    """Reed-Solomon encode: append `nsym` check symbols to `data`.

    data: 1-D array of GF(2^8) symbols (uint8, length <= 255 - nsym).
    Returns codeword of length len(data) + nsym.
    """
    _gf_init()
    data = np.asarray(data, dtype=np.int32).ravel()
    n = len(data)
    if n + nsym > 255:
        raise ValueError(f"RS(n+nsym={n + nsym}) exceeds GF(2^8) field size 255")

    gen = _rs_generator(nsym, fcr)
    # Polynomial division: data * x^nsym mod gen
    msg = np.concatenate([data, np.zeros(nsym, dtype=np.int32)])
    for i in range(n):
        coef = msg[i]
        if coef == 0:
            continue
        for j in range(1, len(gen)):
            msg[i + j] ^= _gf_mul(int(gen[j]), int(coef))
    # Check symbols are the remainder
    check = msg[n:]
    return np.concatenate([data, check]).astype(np.uint8)


def _rs_syndromes(msg: np.ndarray, nsym: int, fcr: int = 0) -> np.ndarray:
    """Compute syndromes S_i = msg(alpha^i) for i in [fcr, fcr+nsym)."""
    _gf_init()
    synd = np.zeros(nsym, dtype=np.int32)
    for i in range(nsym):
        synd[i] = _gf_poly_eval(msg, _gf_pow(2, fcr + i))
    return synd


def _rs_berlekamp_massey(synd: np.ndarray, nsym: int) -> np.ndarray:
    """Berlekamp-Massey algorithm to find the error-locator polynomial."""
    # sigma(x) = 1 + s1*x + s2*x^2 + ...
    sigma = np.zeros(nsym + 1, dtype=np.int32)
    old_sigma = np.zeros(nsym + 1, dtype=np.int32)
    sigma[0] = 1
    old_sigma[0] = 1
    L = 0
    for n in range(nsym):
        delta = int(synd[n])
        for i in range(1, L + 1):
            delta ^= _gf_mul(int(sigma[i]), int(synd[n - i]))
        old_sigma = np.roll(old_sigma, 1)
        old_sigma[0] = 0
        if delta == 0:
            continue
        if 2 * L <= n:
            new_sigma = sigma.copy()
            for i in range(nsym + 1):
                sigma[i] ^= _gf_mul(delta, int(old_sigma[i]))
            old_sigma = new_sigma.copy()
            inv_d = _gf_inv(delta)
            for i in range(nsym + 1):
                old_sigma[i] = _gf_mul(int(old_sigma[i]), inv_d)
            L = n + 1 - L
        else:
            for i in range(nsym + 1):
                sigma[i] ^= _gf_mul(delta, int(old_sigma[i]))
    # Trim trailing zeros
    deg = nsym
    while deg > 0 and sigma[deg] == 0:
        deg -= 1
    return sigma[:deg + 1]


def _rs_find_errors(sigma: np.ndarray, n: int) -> List[int]:
    """Chien search: find roots of sigma(x) -> error positions."""
    _gf_init()
    errs = []
    for i in range(n):
        if _gf_poly_eval(sigma, _gf_pow(2, i)) == 0:
            errs.append(n - 1 - i)
    return errs


def _rs_forney(synd: np.ndarray, sigma: np.ndarray,
               positions: List[int], n: int, fcr: int = 0) -> List[int]:
    """Forney algorithm: compute error magnitudes."""
    _gf_init()
    nsym = len(synd)
    # Error evaluator polynomial: Omega = S * sigma mod x^nsym
    omega = np.zeros(nsym, dtype=np.int32)
    for i in range(nsym):
        tmp = int(synd[i])
        for j in range(1, min(i + 1, len(sigma))):
            tmp ^= _gf_mul(int(sigma[j]), int(synd[i - j]))
        omega[i] = tmp

    # Formal derivative of sigma
    magnitudes = []
    for pos in positions:
        xi = _gf_pow(2, n - 1 - pos)
        xi_inv = _gf_inv(xi)
        # Evaluate omega at xi_inv
        omega_val = 0
        for i in range(nsym):
            omega_val ^= _gf_mul(int(omega[i]), _gf_pow(xi_inv, i))
        # Evaluate sigma' (formal derivative) at xi_inv
        sigma_deriv = 0
        for i in range(1, len(sigma), 2):
            sigma_deriv ^= _gf_mul(int(sigma[i]), _gf_pow(xi_inv, i - 1))
        if sigma_deriv == 0:
            magnitudes.append(0)
        else:
            # e_k = X_k * Omega(X_k^-1) / sigma'(X_k^-1) * alpha^(-fcr*(pos))
            mag = _gf_mul(_gf_mul(xi, omega_val), _gf_inv(sigma_deriv))
            magnitudes.append(mag)
    return magnitudes


def rs_decode(codeword: np.ndarray, nsym: int = 32,
              fcr: int = 0) -> Tuple[np.ndarray, int]:
    """Reed-Solomon decode.

    Returns (corrected_data, n_errors).  n_errors = -1 if uncorrectable.
    """
    _gf_init()
    msg = np.asarray(codeword, dtype=np.int32).ravel().copy()
    n = len(msg)

    synd = _rs_syndromes(msg, nsym, fcr)
    if np.all(synd == 0):
        return msg[:n - nsym].astype(np.uint8), 0

    sigma = _rs_berlekamp_massey(synd, nsym)
    err_pos = _rs_find_errors(sigma, n)

    if len(err_pos) != len(sigma) - 1:
        log.warning("RS decode: %d roots found for degree-%d locator -> uncorrectable",
                    len(err_pos), len(sigma) - 1)
        return msg[:n - nsym].astype(np.uint8), -1

    magnitudes = _rs_forney(synd, sigma, err_pos, n, fcr)
    for pos, mag in zip(err_pos, magnitudes):
        if 0 <= pos < n:
            msg[pos] ^= mag

    # Verify syndromes are now zero
    synd2 = _rs_syndromes(msg, nsym, fcr)
    if not np.all(synd2 == 0):
        log.warning("RS decode: syndromes non-zero after correction -> uncorrectable")
        return msg[:n - nsym].astype(np.uint8), -1

    return msg[:n - nsym].astype(np.uint8), len(err_pos)


# =========================================================================== #
#  3. Concatenated codes (inner Viterbi + outer RS)
# =========================================================================== #
def concat_encode(data_bytes: np.ndarray, rs_nsym: int = 32,
                  conv_K: int = 7, conv_polys: Tuple[int, ...] = POLY_DEFAULT,
                  rs_fcr: int = 0) -> np.ndarray:
    """Concatenated encode: RS outer -> bit unpack -> conv inner.

    data_bytes: 1-D uint8 array of message bytes.
    Returns: coded bit stream (uint8, 0/1).
    """
    _gf_init()
    data = np.asarray(data_bytes, dtype=np.uint8).ravel()
    # Outer RS encode (byte level)
    rs_coded = rs_encode(data, nsym=rs_nsym, fcr=rs_fcr)
    # Unpack to bits
    bits = np.unpackbits(rs_coded)
    # Inner convolutional encode
    coded = conv_encode(bits, K=conv_K, polys=conv_polys, tail=True)
    return coded.ravel().astype(np.uint8)


def concat_decode(coded_bits: np.ndarray, rs_nsym: int = 32,
                  n_rs_symbols: int = 255, conv_K: int = 7,
                  conv_polys: Tuple[int, ...] = POLY_DEFAULT,
                  rs_fcr: int = 0) -> Tuple[np.ndarray, Dict]:
    """Concatenated decode: inner Viterbi -> bit pack -> outer RS.

    Returns (data_bytes, {"viterbi_bits": int, "rs_errors": int}).
    """
    coded = np.asarray(coded_bits, dtype=np.uint8).ravel()
    n_polys = len(conv_polys)
    if coded.size % n_polys != 0:
        coded = coded[:coded.size - coded.size % n_polys]
    coded_2d = coded.reshape(-1, n_polys)

    # Inner Viterbi decode
    info_bits = viterbi_decode(coded_2d, K=conv_K, polys=conv_polys, tail=True)

    # Pack bits to bytes
    n_bytes = n_rs_symbols
    need_bits = n_bytes * 8
    if info_bits.size < need_bits:
        log.warning("concat_decode: Viterbi output %d bits < %d needed for RS(%d)",
                    info_bits.size, need_bits, n_rs_symbols)
        # Pad with zeros
        info_bits = np.concatenate([info_bits,
                                    np.zeros(need_bits - info_bits.size, dtype=np.uint8)])
    rs_bytes = np.packbits(info_bits[:need_bits])

    # Outer RS decode
    data, n_err = rs_decode(rs_bytes, nsym=rs_nsym, fcr=rs_fcr)
    return data, {"viterbi_bits": int(info_bits.size), "rs_errors": n_err}


# =========================================================================== #
#  4. LDPC -- Belief Propagation (Min-Sum variant)
# =========================================================================== #
def make_ldpc_parity_check(n: int = 48, k: int = 24,
                           seed: int = 2026) -> np.ndarray:
    """Generate a systematic (n, k) LDPC parity-check matrix H = [P | I_{n-k}].

    Uses a pseudo-random parity construction with column weight ~3 for information
    bits and an identity structure for parity bits, ensuring straightforward systematic
    encoding and full row rank.
    """
    m = n - k  # number of check equations
    rng = np.random.default_rng(seed)
    P = np.zeros((m, k), dtype=np.uint8)
    for j in range(k):
        idx = rng.choice(m, size=min(3, m), replace=False)
        P[idx, j] = 1
    # Ensure every check equation has at least degree 2
    for i in range(m):
        if P[i].sum() < 2:
            cols = rng.choice(k, size=min(2, k), replace=False)
            P[i, cols] = 1
    H = np.hstack([P, np.eye(m, dtype=np.uint8)])
    return H


def _ldpc_encode_systematic(msg_bits: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Systematic LDPC encode using parity check matrix H.

    If H is in systematic form [P | I_m], parity = (P @ msg) % 2.
    Otherwise, uses Gaussian elimination over GF(2) to solve H @ cw = 0.
    """
    m, n = H.shape
    k = n - m
    msg = np.asarray(msg_bits, dtype=np.uint8).ravel() & 1
    if msg.size != k:
        raise ValueError(f"LDPC encode: message must be {k} bits, got {msg.size}")

    # Check if H already has I_m in the right-most m columns
    if np.array_equal(H[:, k:], np.eye(m, dtype=H.dtype)):
        P = H[:, :k]
        parity = (P @ msg) % 2
        return np.concatenate([msg, parity]).astype(np.uint8)

    # General Gaussian elimination over GF(2)
    Hw = H.copy().astype(np.uint8)
    pivots = {}  # row -> col
    r = 0
    for c in range(n):
        p_row = -1
        for ro in range(r, m):
            if Hw[ro, c]:
                p_row = ro
                break
        if p_row == -1:
            continue
        if p_row != r:
            Hw[[r, p_row]] = Hw[[p_row, r]]
        for ro in range(m):
            if ro != r and Hw[ro, c]:
                Hw[ro] ^= Hw[r]
        pivots[r] = c
        r += 1
        if r == m:
            break

    pivot_cols = set(pivots.values())
    free_cols = [c for c in range(n) if c not in pivot_cols]

    cw = np.zeros(n, dtype=np.uint8)
    for j, c in enumerate(free_cols):
        if j < k:
            cw[c] = msg[j]
    for r in range(len(pivots)):
        c = pivots[r]
        cw[c] = int((Hw[r, free_cols[:k]] @ msg) % 2)

    return cw


def ldpc_encode(msg_bits: np.ndarray, H: np.ndarray) -> np.ndarray:
    """LDPC encode using the parity-check matrix H.

    Returns systematic codeword of length n = H.shape[1].
    """
    return _ldpc_encode_systematic(msg_bits, H)


def ldpc_decode(received: np.ndarray, H: np.ndarray,
                max_iter: int = 50, snr_db: float = 10.0) -> Tuple[np.ndarray, int]:
    """Min-Sum LDPC decoder (hard-decision input).

    Parameters
    ----------
    received : (n,) uint8, hard bits 0/1.
    H : (m, n) parity check matrix.
    max_iter : maximum BP iterations.
    snr_db : assumed SNR for LLR initialization (higher = more trust in input).

    Returns (decoded_bits (n,), n_iterations_used).
    """
    m, n = H.shape
    r = np.asarray(received, dtype=np.float64).ravel()
    if r.size != n:
        raise ValueError(f"LDPC decode: received {r.size} bits, H has {n} columns")

    # Convert hard bits to LLRs: L = (-1)^bit * 2*SNR_linear
    snr_lin = 10.0 ** (snr_db / 10.0)
    llr = (1.0 - 2.0 * r) * 2.0 * snr_lin   # +ve for bit=0, -ve for bit=1

    # Build adjacency: which checks connect to which variables
    check_nodes = [np.flatnonzero(H[i]) for i in range(m)]
    var_nodes = [np.flatnonzero(H[:, j]) for j in range(n)]

    # Messages: check-to-variable (R) and variable-to-check (Q)
    R = np.zeros((m, n), dtype=np.float64)
    Q = np.zeros((m, n), dtype=np.float64)

    # Initialize Q with channel LLRs
    for j in range(n):
        for i in var_nodes[j]:
            Q[i, j] = llr[j]

    for iteration in range(max_iter):
        # Check node update (Min-Sum approximation)
        for i in range(m):
            vn = check_nodes[i]
            if len(vn) < 2:
                continue
            for j in vn:
                # Product of signs and min of magnitudes, excluding j
                others = [v for v in vn if v != j]
                sign = 1.0
                min_abs = np.inf
                for v in others:
                    sign *= np.sign(Q[i, v]) if Q[i, v] != 0 else 1.0
                    min_abs = min(min_abs, abs(Q[i, v]))
                # Scaling factor 0.75 for min-sum normalization
                R[i, j] = sign * min_abs * 0.75

        # Variable node update
        for j in range(n):
            cn = var_nodes[j]
            total = llr[j] + sum(R[i, j] for i in cn)
            for i in cn:
                Q[i, j] = total - R[i, j]

        # Hard decision
        total_llr = llr.copy()
        for j in range(n):
            for i in var_nodes[j]:
                total_llr[j] += R[i, j]
        decoded = (total_llr < 0).astype(np.uint8)

        # Syndrome check
        syndrome = (H @ decoded) % 2
        if np.all(syndrome == 0):
            return decoded, iteration + 1

    # Failed to converge -- return best guess
    total_llr = llr.copy()
    for j in range(n):
        for i in var_nodes[j]:
            total_llr[j] += R[i, j]
    decoded = (total_llr < 0).astype(np.uint8)
    return decoded, max_iter
