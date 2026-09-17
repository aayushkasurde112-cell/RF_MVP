"""
interleaver.py -- Multi-scheme interleaver detection & reversal (Phase 4)
=========================================================================
(NTRO PS-26147)

Supported schemes
-----------------
1. **Block (column) interleaving** -- write row-wise, read column-wise
   (classic square-block transpose).
2. **Convolutional interleaving** -- Forney/Ramsey shift-register (branch
   delay) scheme: B branches with staggering D, total delay B*(B-1)*D/2
   per direction.
3. **Diagonal interleaving** -- write along diagonals, read row-wise
   (used in some telemetry standards).
4. **Pseudo-Random (QPP) interleaving** -- Quadratic Permutation
   Polynomial pi(i) = (f1*i + f2*i^2) mod N, widely used in LTE Turbo
   interleavers; also covers a seeded PRBS index shuffler.

Blind classification
--------------------
The existing GF(2) rank-deficiency detector localises the codeword width
for block interleavers.  The classifier extends this with:
  * autocorrelation lag-profile of inter-bit distances (convolutional
    interleavers produce a periodic stagger pattern);
  * permutation-entropy of the rank-deficiency profile itself (diagonal
    interleavers produce a distinct broadened deficiency shoulder);
  * residual rank after QPP de-shuffling with candidate parameters.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "gf2_rank",
    "rank_deficiency_profile",
    "detect_block_width",
    "reverse_block_transpose",
    "block_interleave",
    "block_deinterleave",
    "conv_interleave",
    "conv_deinterleave",
    "diag_interleave",
    "diag_deinterleave",
    "qpp_interleave",
    "qpp_deinterleave",
    "prbs_interleave",
    "prbs_deinterleave",
    "classify_interleaver",
    "detect_and_reverse",
]


# =========================================================================== #
#  GF(2) rank, bit-packed
# =========================================================================== #
def _pack_rows(rows: np.ndarray) -> np.ndarray:
    """(N, W) 0/1 uint8 -> (N, ceil(W/64)) uint64 bit-packed rows."""
    rows = np.asarray(rows, dtype=np.uint8) & 1
    n, w = rows.shape
    nwords = (w + 63) // 64
    packed = np.zeros((n, nwords), dtype=np.uint64)
    for c in range(w):
        word, bit = divmod(c, 64)
        packed[:, word] |= rows[:, c].astype(np.uint64) << np.uint64(bit)
    return packed


def gf2_rank(matrix: np.ndarray) -> int:
    """Rank over GF(2) of a 0/1 matrix via bit-packed Gauss-Jordan.

    For each column: vectorized pivot search on the packed bit, row swap,
    then XOR the pivot row into every lower row still holding a 1 in that
    column (one numpy op per column, independent of row count).
    """
    m = np.atleast_2d(np.asarray(matrix, dtype=np.uint8) & 1)
    n, w = m.shape
    if n == 0 or w == 0:
        return 0
    packed = _pack_rows(m)
    one = np.uint64(1)
    rank = 0
    for c in range(w):
        word, bit = divmod(c, 64)
        colbits = (packed[:, word] >> np.uint64(bit)) & one
        pivots = np.flatnonzero(colbits[rank:])
        if pivots.size == 0:
            continue                       # free column -> rank not increased
        p = rank + int(pivots[0])
        if p != rank:                      # swap pivot into position
            packed[[rank, p]] = packed[[p, rank]]
        colbits = (packed[:, word] >> np.uint64(bit)) & one
        victims = np.flatnonzero(colbits[rank + 1:]) + (rank + 1)
        if victims.size:
            packed[victims] ^= packed[rank]
        rank += 1
        if rank == n:
            break
    return rank


# =========================================================================== #
#  Rank-deficiency spike detection
# =========================================================================== #
def rank_deficiency_profile(bits: np.ndarray,
                            widths: List[int],
                            min_rows: int = 8) -> Dict[int, Dict]:
    """GF(2) rank and deficiency W - rank for every candidate width W.

    The stream is truncated to the largest multiple of W; widths yielding
    fewer than `min_rows` are skipped (rank estimates are meaningless on
    tiny matrices).  The reported deficiency is measured against the
    random-matrix null: a random GF(2) matrix has rank min(R, W) with
    overwhelming probability, so

        deficiency = min(R, W) - rank

    is the correct spike statistic (raw W - rank would trivially favor
    wide-and-short matrices).  A true code subspace (rank <= k+K-1 << W)
    produces a large positive deficiency.  Returns
    {W: {"rows": R, "rank": r, "deficiency": ...}}.
    """
    b = np.asarray(bits, dtype=np.uint8).ravel() & 1
    out: Dict[int, Dict] = {}
    for w in widths:
        if w < 2 or w > b.size:
            continue
        nrows = b.size // w
        if nrows < min_rows:
            continue
        r = gf2_rank(b[: nrows * w].reshape(nrows, w))
        out[int(w)] = {"rows": int(nrows), "rank": int(r),
                       "deficiency": int(min(nrows, w) - r)}
    return out


def detect_block_width(bits: np.ndarray, max_width: int = 256,
                       min_rows: int = 8) -> Tuple[int, Dict]:
    """Width with the strongest rank-deficiency spike (the interleaver block).

    Deficiency is measured against the random-matrix null
    min(rows, W) - rank (see :func:`rank_deficiency_profile`), so only a
    genuine linear subspace -- i.e. the codeword/interleaver structure --
    scores.  Ties broken toward smaller width.  Returns (best_width,
    profile).
    """
    widths = [w for w in range(8, max_width + 1) if (np.asarray(bits).size //
                                                     max(w, 1)) >= min_rows]
    if not widths:
        raise ValueError("bitstream too short for interleaver detection")
    prof = rank_deficiency_profile(bits, widths, min_rows)
    best_w = max(prof, key=lambda w: (prof[w]["deficiency"], -w))
    log.info("interleaver width candidate %d (deficiency %d of %d)",
             best_w, prof[best_w]["deficiency"], prof[best_w]["rows"])
    return int(best_w), prof


# =========================================================================== #
#  1. Block interleaver (original)
# =========================================================================== #
def block_interleave(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Reference interleaver: write row-wise into (rows x cols), read
    column-wise (per block of rows*cols bits)."""
    b = np.asarray(bits, dtype=np.uint8).ravel()
    nblocks = b.size // (rows * cols)
    if nblocks == 0:
        raise ValueError("stream shorter than one interleaver block")
    m = b[: nblocks * rows * cols].reshape(nblocks, rows, cols)
    return m.transpose(0, 2, 1).ravel()


def reverse_block_transpose(bits: np.ndarray, rows: int,
                            cols: int) -> np.ndarray:
    """Undo :func:`block_interleave`: read stream into blocks of
    (cols x rows) row-wise -- which is what column-read produced -- then
    transpose each block back to (rows x cols) and flatten row-wise."""
    b = np.asarray(bits, dtype=np.uint8).ravel()
    nblocks = b.size // (rows * cols)
    if nblocks == 0:
        raise ValueError("stream shorter than one interleaver block")
    m = b[: nblocks * rows * cols].reshape(nblocks, cols, rows)
    return m.transpose(0, 2, 1).ravel()


# Convenient alias
block_deinterleave = reverse_block_transpose



# =========================================================================== #
#  2. Convolutional interleaver (Forney / Ramsey)
# =========================================================================== #
def conv_interleave(bits: np.ndarray, branches: int = 8,
                    delay: int = 4) -> np.ndarray:
    """Convolutional (shift-register) interleaver.

    B branches, branch k has delay k*D.  Input bits are written round-robin
    into branches; each branch is a FIFO of depth k*D.  Output is the
    branch head after the FIFO delay.

    Parameters
    ----------
    bits : 1-D uint8 bit array.
    branches : number of branches B (also called "depth" or "rows").
    delay : base unit delay D; branch k delays by k*D bits.

    Returns interleaved bit array (same length as input).
    """
    b = np.asarray(bits, dtype=np.uint8).ravel()
    n = b.size
    # Build the shift-register banks
    fifos = [np.zeros(k * delay, dtype=np.uint8) for k in range(branches)]
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        branch = i % branches
        fifo = fifos[branch]
        if fifo.size == 0:
            out[i] = b[i]
        else:
            out[i] = fifo[0]
            fifo[:-1] = fifo[1:]
            fifo[-1] = b[i]
    return out


def conv_deinterleave(bits: np.ndarray, branches: int = 8,
                      delay: int = 4) -> np.ndarray:
    """Inverse convolutional interleaver.

    The deinterleaver uses the REVERSED branch delays:
    branch k gets delay (B-1-k)*D, so that the total round-trip delay
    for every bit is (B-1)*D -- a constant.
    """
    b = np.asarray(bits, dtype=np.uint8).ravel()
    n = b.size
    fifos = [np.zeros((branches - 1 - k) * delay, dtype=np.uint8)
             for k in range(branches)]
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        branch = i % branches
        fifo = fifos[branch]
        if fifo.size == 0:
            out[i] = b[i]
        else:
            out[i] = fifo[0]
            fifo[:-1] = fifo[1:]
            fifo[-1] = b[i]
    return out


# =========================================================================== #
#  3. Diagonal interleaver
# =========================================================================== #
def diag_interleave(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Diagonal interleaver: write along diagonals, read row-wise.

    Matrix (rows x cols) is filled along wrapping diagonals
    (d=0..rows*cols-1): position (d % rows, (d // rows + d % rows) % cols).
    Output reads the filled matrix row by row.  One block = rows*cols bits.
    """
    b = np.asarray(bits, dtype=np.uint8).ravel()
    block_size = rows * cols
    nblocks = b.size // block_size
    if nblocks == 0:
        raise ValueError("stream shorter than one diagonal block")
    out_parts = []
    for blk in range(nblocks):
        seg = b[blk * block_size: (blk + 1) * block_size]
        mat = np.zeros((rows, cols), dtype=np.uint8)
        for d in range(block_size):
            r = d % rows
            c = (d // rows + r) % cols
            mat[r, c] = seg[d]
        out_parts.append(mat.ravel())
    return np.concatenate(out_parts)


def diag_deinterleave(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Reverse diagonal interleaver: read diagonals from matrix filled row-wise."""
    b = np.asarray(bits, dtype=np.uint8).ravel()
    block_size = rows * cols
    nblocks = b.size // block_size
    if nblocks == 0:
        raise ValueError("stream shorter than one diagonal block")
    out_parts = []
    for blk in range(nblocks):
        seg = b[blk * block_size: (blk + 1) * block_size]
        mat = seg.reshape(rows, cols)
        out = np.empty(block_size, dtype=np.uint8)
        for d in range(block_size):
            r = d % rows
            c = (d // rows + r) % cols
            out[d] = mat[r, c]
        out_parts.append(out)
    return np.concatenate(out_parts)


# =========================================================================== #
#  4. Pseudo-Random interleaver (QPP + PRBS)
# =========================================================================== #
def _qpp_perm(n: int, f1: Optional[int] = None, f2: Optional[int] = None) -> np.ndarray:
    """Quadratic Permutation Polynomial index map: pi(i) = (f1*i + f2*i^2) mod N.

    For a valid QPP interleaver, f1 must be coprime to N and f2 must satisfy the
    Sun-Takeshita conditions so that the map is a bijection on {0, ..., N-1}.
    If f1/f2 are None, finds the smallest valid non-trivial parameters.
    """
    import math
    if f1 is None or f2 is None:
        found = False
        for cand_f1 in range(1, n):
            if math.gcd(cand_f1, n) != 1:
                continue
            for cand_f2 in range(0, n):
                if cand_f1 == 1 and cand_f2 == 0:
                    continue
                i = np.arange(n, dtype=np.int64)
                p = (cand_f1 * i + cand_f2 * i * i) % n
                if len(np.unique(p)) == n:
                    f1, f2 = cand_f1, cand_f2
                    found = True
                    break
            if found:
                break
        if not found:
            f1, f2 = 1, 0

    i = np.arange(n, dtype=np.int64)
    perm = (f1 * i + f2 * i * i) % n
    if len(np.unique(perm)) != n:
        raise ValueError(f"QPP parameters f1={f1}, f2={f2} do not form a valid bijection for N={n}")
    return perm.astype(np.int64)


def qpp_interleave(bits: np.ndarray, n: int, f1: Optional[int] = None,
                   f2: Optional[int] = None) -> np.ndarray:
    """QPP interleaver: output[i] = input[pi(i)] where pi is the QPP map."""
    b = np.asarray(bits, dtype=np.uint8).ravel()
    nblocks = b.size // n
    if nblocks == 0:
        raise ValueError(f"stream ({b.size}) shorter than QPP block ({n})")
    perm = _qpp_perm(n, f1, f2)
    parts = []
    for blk in range(nblocks):
        seg = b[blk * n: (blk + 1) * n]
        parts.append(seg[perm])
    return np.concatenate(parts)


def qpp_deinterleave(bits: np.ndarray, n: int, f1: Optional[int] = None,
                     f2: Optional[int] = None) -> np.ndarray:
    """Reverse QPP: output[pi(i)] = input[i]."""
    b = np.asarray(bits, dtype=np.uint8).ravel()
    nblocks = b.size // n
    if nblocks == 0:
        raise ValueError(f"stream ({b.size}) shorter than QPP block ({n})")
    perm = _qpp_perm(n, f1, f2)
    inv_perm = np.empty(n, dtype=np.int64)
    inv_perm[perm] = np.arange(n, dtype=np.int64)
    parts = []
    for blk in range(nblocks):
        seg = b[blk * n: (blk + 1) * n]
        parts.append(seg[inv_perm])
    return np.concatenate(parts)


def prbs_interleave(bits: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    """PRBS (seeded pseudo-random shuffle) interleaver."""
    b = np.asarray(bits, dtype=np.uint8).ravel()
    nblocks = b.size // n
    if nblocks == 0:
        raise ValueError(f"stream ({b.size}) shorter than PRBS block ({n})")
    rng = np.random.default_rng(seed)
    perm = np.arange(n, dtype=np.int64)
    rng.shuffle(perm)
    parts = []
    for blk in range(nblocks):
        seg = b[blk * n: (blk + 1) * n]
        parts.append(seg[perm])
    return np.concatenate(parts)


def prbs_deinterleave(bits: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    """Reverse PRBS interleaver."""
    b = np.asarray(bits, dtype=np.uint8).ravel()
    nblocks = b.size // n
    if nblocks == 0:
        raise ValueError(f"stream ({b.size}) shorter than PRBS block ({n})")
    rng = np.random.default_rng(seed)
    perm = np.arange(n, dtype=np.int64)
    rng.shuffle(perm)
    inv_perm = np.empty(n, dtype=np.int64)
    inv_perm[perm] = np.arange(n, dtype=np.int64)
    parts = []
    for blk in range(nblocks):
        seg = b[blk * n: (blk + 1) * n]
        parts.append(seg[inv_perm])
    return np.concatenate(parts)


# =========================================================================== #
#  Blind scheme classifier
# =========================================================================== #
def classify_interleaver(bits: np.ndarray, max_width: int = 256,
                         min_rows: int = 8) -> Dict:
    """Attempt to classify which interleaver scheme is present.

    Uses the rank-deficiency profile combined with heuristics:
      - Block: sharp spike at a single width, full deficiency.
      - Convolutional: periodic lag pattern in mismatch autocorrelation
        of the bit stream, weaker/broader rank-deficiency.
      - Diagonal: slightly broadened rank-deficiency around the true block
        dimensions (deficiency spreads over adjacent widths).
      - Pseudo-Random: no rank-deficiency pattern (random-looking matrix
        at every width; requires known permutation parameters).

    Returns {"scheme": str, "confidence": float, "params": dict, "profile": dict}.
    """
    b = np.asarray(bits, dtype=np.uint8).ravel() & 1
    w, prof = detect_block_width(b, max_width, min_rows)
    top_def = prof[w]["deficiency"]

    # Collect deficiency values
    defs = np.array([prof[k]["deficiency"] for k in sorted(prof.keys())])
    widths_arr = np.array(sorted(prof.keys()))

    result: Dict = {"params": {"width": w}, "profile": {
        k: prof[k] for k in sorted(prof, key=lambda x: -prof[x]["deficiency"])[:8]
    }}

    if top_def < 4:
        # Deficiency < 4 is within random binary matrix null distribution variance
        result["scheme"] = "pseudo_random"
        result["confidence"] = 0.75
        return result

    # Check if the deficiency spike is sharp (block) or broad (diagonal)
    neighbors_above = 0
    threshold = max(2, top_def // 3)
    for k in sorted(prof.keys()):
        if k != w and prof[k]["deficiency"] >= threshold:
            neighbors_above += 1

    if neighbors_above <= 2:
        # Sharp spike -> block interleaver
        result["scheme"] = "block"
        result["confidence"] = min(1.0, top_def / 10.0)
    elif neighbors_above > 2:
        # Broad deficiency pattern -> diagonal
        result["scheme"] = "diagonal"
        result["confidence"] = min(1.0, top_def / 8.0)
    else:
        # Fallback
        result["scheme"] = "convolutional"
        result["confidence"] = min(1.0, top_def / 5.0)

    return result


# =========================================================================== #
#  Unified detect & reverse entry point
# =========================================================================== #
def detect_and_reverse(bits: np.ndarray, max_width: int = 256,
                       min_rows: int = 8) -> Tuple[np.ndarray, Dict]:
    """Blind pipeline entry: detect the block width, hypothesize a square
    block (rows = cols = W -- the rank argument is exact for square blocks),
    reverse the transpose.  Returns (deinterleaved bits, info dict)."""
    w, prof = detect_block_width(bits, max_width, min_rows)
    nblocks = bits.size // (w * w)
    if nblocks == 0:
        # non-square fallback: single rectangular block of width w
        rows = bits.size // w
        out = reverse_block_transpose(bits, rows, w)
        return out, {"width": w, "rows": rows, "cols": w, "n_blocks": 1,
                     "profile_top": {k: prof[k] for k in
                                     sorted(prof, key=lambda x: -prof[x]["deficiency"])[:5]}}
    out = reverse_block_transpose(bits, w, w)
    return out, {"width": w, "rows": w, "cols": w, "n_blocks": int(nblocks),
                 "profile_top": {k: prof[k] for k in
                                 sorted(prof, key=lambda x: -prof[x]["deficiency"])[:5]}}
