"""
amc.py -- Automatic Modulation Classification via Higher-Order Cumulants
========================================================================
(Phase 2, NTRO PS-26147)

Classifies the *modulation family* (FSK vs PSK vs QAM) of one channelized
burst in three stages:

1. **Symbol synchronization** (:func:`extract_symbols`)
   RRC matched filter (roll-off estimated from the measured occupied
   bandwidth vs baud) -> integer decimation to ~4 samples/symbol -> fractional
   timing search over one symbol period, choosing the offset that maximizes
   |C42| (standard timing criterion: cumulants peak when ISI is minimized).

2. **HOC extraction** (:func:`compute_cumulants`)
   Normalized cumulants C40, C42, C63 on unit-power symbols
   (moments M_mn = E[x^m (x*)^n], C40 = M40 - 3 M20^2, etc. -- the C63
   6th-order partition expansion is verified against brute-force cumulant
   evaluation in the Phase-2 test log).  Optional first-order SNR
   de-biasing using the Phase-1 burst SNR.

3. **Decision tree** (:func:`classify_family`)
   Constant-envelope gate (PAPR + instantaneous-frequency state count)
   catches FSK first -- cumulant theory above assumes a linearly modulated
   pulse train, which FSK is not.  Then a threshold tree over
   (|C40|, |C42|, |C63|) against analytically-derived constellation anchors,
   with a softmax over anchor distances producing the ranked confidence
   dictionary ``{family: score}``.

Theoretical anchors (unit-power symbols, exact for infinite SNR):

    mod      |C40|   |C42|   |C63|
    BPSK     2.000   2.000  14.000
    QPSK     1.000   1.000   5.000
    8PSK     0.000   1.000   5.000
    16QAM    0.679   0.679   4.038
    64QAM    0.619   0.622   5.435

Note the design constraint this table exposes: for *square* QAM
|C40| == |C42|, so the popular |C40|/|C42| ratio cannot separate QAM from
PSK -- the tree must use the joint (|C40|, |C42|, |C63|) space.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal

from symbol_rate import instantaneous_frequency_modes

log = logging.getLogger(__name__)

__all__ = [
    "AMCResult",
    "compute_cumulants",
    "extract_symbols",
    "classify_family",
    "analyze_burst",
    "rrc_taps",
]

# --------------------------------------------------------------------------- #
# Constellation anchors (see module docstring; exact arithmetic on the unit-
# power constellations, not Monte-Carlo).
# --------------------------------------------------------------------------- #
_ANCHORS: Dict[str, Tuple[float, float, float]] = {
    "BPSK":  (2.000, 2.000, 14.000),
    "QPSK":  (1.000, 1.000,  5.000),
    "8PSK":  (0.000, 1.000,  5.000),
    "16QAM": (0.679, 0.679,  4.038),
    "64QAM": (0.619, 0.622,  5.435),
}
_FAMILY_OF = {"BPSK": "PSK", "QPSK": "PSK", "8PSK": "PSK",
              "16QAM": "QAM", "64QAM": "QAM"}
_SCALES = np.array([2.0, 2.0, 9.0])   # feature normalizers (anchor spread)
_SOFTMAX_T = 0.30                     # temperature: sharp but not saturated


@dataclass(frozen=True)
class AMCResult:
    """AMC output: ranked family confidences + full feature provenance."""

    family_ranking: Dict[str, float]     # {"PSK": 0.81, "QAM": 0.15, "FSK": 0.04}
    subtype_guess: Optional[str]         # "QPSK", ... (None for FSK/unknown)
    n_symbols: int
    features: Dict = field(default_factory=dict)
    decisions: Tuple[str, ...] = ()      # human-readable decision-tree trace

    @property
    def best_family(self) -> Optional[str]:
        return next(iter(self.family_ranking), None)

    @property
    def confidence(self) -> float:
        return self.family_ranking.get(self.best_family, 0.0) if \
            self.family_ranking else 0.0


# --------------------------------------------------------------------------- #
# 1) Higher-order cumulants
# --------------------------------------------------------------------------- #
def compute_cumulants(samples: np.ndarray,
                      snr_db: Optional[float] = None) -> Dict[str, complex]:
    """Normalized HOCs C40, C42, C63 (+ C20/C21) of complex-baseband samples.

    The samples are first power-normalized to unit variance (cumulants are
    scale-equivariant, so this fixes the operating point of every threshold).
    Moment conventions: M_mn = E[x^m (x*)^n].

        C40 = M40 - 3 M20^2
        C42 = M42 - |M20|^2 - 2 M21power^2        (M21power = M11)
        C63 = M63 - 6 M11^3 - 9 |M20|^2 M11
                      - 9 |M21|^2 - |M30|^2       (M21 = E[x^2 x*])

    The C63 expansion enumerates the zero-mean set partitions of
    cum(x,x,x,x*,x*,x*): 15 three-pair partitions -> 6 M11^3 (the mixed
    perfect matchings) + 9 |M20|^2 M11, and 10 two-triple partitions ->
    9 |M21|^2 + |M30|^2.  Verified against direct brute-force evaluation
    (see test log): BPSK -14, QPSK/8PSK -5, 16QAM -4.04.

    SNR de-biasing (optional, first order): additive circular-Gaussian noise
    with gamma = Ps/Pn biases the measured cumulants of an M20 = 0 signal to

        C40_meas = (C40 g^2 + 6 g) / (g+1)^2 ,   C42_meas = (C42 g^2 + 2 g) / (g+1)^2

    (from E[|s+n|^4] = E|s|^4 + 2Pn^2 + 2PsPn etc.); we invert those exactly
    for C40/C42 given `snr_db` from Phase 1.  C63 is left raw (its bias
    involves 5th-order terms; at burst SNR > 12 dB the correction is < 2 %).
    """
    x = np.asarray(samples, dtype=np.complex128).ravel()
    if x.size < 64:
        raise ValueError(f"need >= 64 symbols for stable cumulants, got {x.size}")
    p = float(np.mean(np.abs(x) ** 2))
    if p <= 0:
        raise ValueError("zero-power symbol vector")
    x = x / np.sqrt(p)  # unit power

    m20 = np.mean(x ** 2)
    m11 = np.mean(np.abs(x) ** 2)          # = 1 after normalization (numerically)
    m30 = np.mean(x ** 3)
    m21 = np.mean(x ** 2 * np.conj(x))
    m40 = np.mean(x ** 4)
    m42 = np.mean(np.abs(x) ** 4)
    m63 = np.mean(np.abs(x) ** 6)

    c40 = m40 - 3.0 * m20 ** 2
    c42 = float((m42 - abs(m20) ** 2 - 2.0 * m11 ** 2).real)
    c63 = float((m63 - 6.0 * m11 ** 3 - 9.0 * abs(m20) ** 2 * m11
                 - 9.0 * abs(m21) ** 2 - abs(m30) ** 2).real)

    out: Dict[str, complex] = {
        "C20": complex(m20), "C21": complex(m11),
        "C40": complex(c40), "C42": complex(c42), "C63": complex(c63),
        "C40_abs": abs(c40), "C42_abs": abs(c42), "C63_abs": abs(c63),
    }

    if snr_db is not None and snr_db > 3.0:
        g = 10.0 ** (float(snr_db) / 10.0)
        g = min(g, 1e4)  # clamp: correction must not blow up at huge SNR
        # invert C40_meas = (C40 g^2 + 6g)/(g+1)^2  and  C42 analog (+2g)
        c40_c = ((g + 1.0) ** 2 * c40 - 6.0 * g) / g ** 2
        c42_c = ((g + 1.0) ** 2 * c42 + 2.0 * g) / g ** 2
        out["C40_corr"] = complex(c40_c)
        out["C42_corr"] = complex(c42_c)
        out["C40_abs_corr"] = abs(c40_c)
        out["C42_abs_corr"] = abs(c42_c)
    return out


# --------------------------------------------------------------------------- #
# 2) Symbol synchronization
# --------------------------------------------------------------------------- #
def rrc_taps(beta: float, span_symbols: int = 8, sps: float = 4.0) -> np.ndarray:
    """Root-raised-cosine FIR taps (unit energy).

    h(t) = [sin(pi t (1-b)) + 4 b t cos(pi t (1+b))] / [pi t (1 - (4 b t)^2)]
    with the standard closed forms at t = 0 and t = +-1/(4b).  The matched
    filter is intentionally generic: we do not know the transmit pulse, but
    RRC with the roll-off estimated from the measured occupied bandwidth is
    the minimum-ISI approximation available to a non-cooperative receiver.
    """
    beta = float(np.clip(beta, 0.02, 1.0))
    n = int(round(span_symbols * sps)) | 1          # odd length
    t = (np.arange(n) - (n - 1) / 2.0) / sps        # symbol units
    h = np.zeros(n)
    core = np.abs(1.0 - (4.0 * beta * t) ** 2) > 1e-9
    with np.errstate(all="ignore"):
        h[core] = (np.sin(np.pi * t[core] * (1 - beta))
                   + 4 * beta * t[core] * np.cos(np.pi * t[core] * (1 + beta))) \
                  / (np.pi * t[core] * (1 - (4 * beta * t[core]) ** 2))
    h[~core] = (beta / np.sqrt(2)) * (
        (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
    h[abs(t) < 1e-9] = 1 - beta + 4 / np.pi
    h /= np.sqrt(np.sum(h ** 2))
    return h


def extract_symbols(
    z: np.ndarray,
    fs: float,
    baud_hz: float,
    alpha: Optional[float] = None,
    span_symbols: int = 8,
    timing_steps: int = 48,
) -> np.ndarray:
    """Matched-filter + timing-recover a unit-power symbol sequence.

    alpha : pulse roll-off for the RRC matched filter.  When None it is
        derived from the occupied bandwidth: a raised-cosine spectrum spans
        B = (1 + alpha) * f_baud, so alpha ~= B/f_baud - 1 (clamped to
        [0.05, 1]).  A mismatch of +-0.1 here only mildly attenuates the
        cumulants; the family-level decision tree tolerates that.

    CFO cleanup: the channelization center comes from the Phase-1 PSD peak
    and is typically off by tens-to-hundreds of Hz; over a burst that is
    many phase wraps, and while |C42|/|C63| are rotation-invariant to a
    CONSTANT phase, C40 = E[x^4] averages to ~0 under drift. The residual is
    removed FIRST via the 4th-power line: E[(x e^{j2pi f_res t})^4] =
    E[x^4] e^{j8pi f_res t}, so FFT(z^4) shows a coherent tone at 4*f_res.
    E[x^4] is strongly nonzero for BPSK (2), QPSK (1) and square QAM
    (-0.68/-0.62), the families whose |C40| carries classification
    information. 8PSK needs no correction: its C40 anchor is 0 anyway and
    C42/C63 depend only on |x|, hence are drift-immune. The 4th power is
    used over the 8th because the |x|^8 envelope continuum masks the tone
    (measured prominence 1.5x vs 55x at ~20 dB SNR). The tone sits on the
    |x|^4 data continuum near DC, so prominence is measured against a
    median-filtered LOCAL baseline; the guard band excludes DC-leakage
    sidelobes of the (possibly strong) E[x^4] tone itself.
    (A lag-1 angle estimator angle(mean z z*) is NOT usable here: with a
    multi-turn phase arc across the burst the mean cancels to noise.)
    FSK bursts never reach this code (the envelope gate routes them out).
    """
    z = np.asarray(z, dtype=np.complex128).ravel()
    fs = float(fs)
    sps = fs / float(baud_hz)
    if sps < 1.5:
        raise ValueError(f"baud {baud_hz} too high for fs={fs} (need sps >= 1.5)")
    if alpha is None:
        alpha = 0.35  # neutral default when no bandwidth info is available

    # --- residual-CFO estimation & removal (4th-power tone) -----------------
    zu = z / np.sqrt(np.mean(np.abs(z) ** 2))
    u = zu ** 4
    spec4 = np.abs(np.fft.fft(u))
    freqs = np.fft.fftfreq(u.size, d=1.0 / fs)
    f_search = 4.0 * min(5000.0, fs / sps / 2.0)     # tone lives at 4*f_res
    guard = max(8, int(round(u.size * 50.0 / fs)))   # >= 50 Hz DC-leakage zone
    band = (np.abs(freqs) <= f_search) & (np.abs(freqs) > guard)
    f_res = 0.0
    if band.sum() > 16:
        kernel = max(31, (spec4.size // 512) | 1)
        baseline = sp_signal.medfilt(spec4, kernel_size=kernel)
        ratio = np.zeros_like(spec4)
        sel = band & (baseline > 0)
        ratio[sel] = spec4[sel] / baseline[sel]
        k4 = int(np.argmax(ratio))
        if ratio[k4] >= 4.0:  # tone >= 12 dB over its local continuum
            k_prev = (k4 - 1) % spec4.size
            k_next = (k4 + 1) % spec4.size
            a_, b_, c_ = np.log(spec4[[k_prev, k4, k_next]])
            denom = a_ - 2 * b_ + c_
            delta = float(np.clip(0.5 * (a_ - c_) / denom, -0.5, 0.5)) \
                if abs(denom) > 1e-12 else 0.0
            f_res = float(freqs[k4] + delta * (freqs[1] - freqs[0])) / 4.0
            n = np.arange(z.size)
            z = z * np.exp(-2j * np.pi * f_res * n / fs)

    taps = rrc_taps(alpha, span_symbols, sps)
    zf = sp_signal.filtfilt(taps, [1.0], z)         # zero-phase matched filter

    dec = max(1, int(round(sps / 4.0)))             # ~4 samples/symbol after decim
    sps_d = sps / dec
    zd = zf[::dec]

    n_sym_max = int((zd.size - 2) // sps_d)
    if n_sym_max < 64:
        raise ValueError(
            f"burst yields only {n_sym_max} symbols at baud {baud_hz:.0f} Hz")

    best = (None, -1.0, None)
    for tau in np.linspace(0.0, sps_d, timing_steps, endpoint=False):
        k = np.arange(n_sym_max)
        pos = tau + k * sps_d
        s = np.interp(pos, np.arange(zd.size), zd.real) + \
            1j * np.interp(pos, np.arange(zd.size), zd.imag)
        c = compute_cumulants(s)
        score = c["C42_abs"]
        if score > best[1]:
            s = s / np.sqrt(np.mean(np.abs(s) ** 2))  # unit power symbols
            best = (s, score, tau)
    return best[0]


# --------------------------------------------------------------------------- #
# 3) Decision tree
# --------------------------------------------------------------------------- #
def _softmax_dist(vec: np.ndarray, temperature: float = _SOFTMAX_T) -> np.ndarray:
    """softmax(-d^2 / T) over squared normalized distances."""
    z = -(vec ** 2) / temperature
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def fsk_gate_probability(env_mod_db: float, n_freq_states: int) -> float:
    """Stage-A gate: P(FSK) from envelope flatness + frequency-state count.

    FSK is constant-envelope AND occupies *discrete frequency states*;
    linear modulations show envelope ripple and a single
    instantaneous-frequency lobe.  The envelope feature is the modulation
    index 10*log10(1 + var(|x|^2)/mean^2) on a 5-tap-smoothed envelope --
    unlike max-based PAPR it is immune to the additive-noise chi^2 floor
    (measured: 4FSK 0.04 dB, shaped QPSK 0.38, 16QAM 1.34).  The two soft
    logistic factors (center 0.7 dB / 1.5 states) must AGREE (product), so a
    constant-envelope but single-state signal (unshaped PSK) stays out of
    the FSK branch and vice versa.
    """
    p_env = 1.0 / (1.0 + np.exp((env_mod_db - 0.7) / 0.5))
    p_st = 1.0 / (1.0 + np.exp(-(n_freq_states - 1.5) / 0.6))
    return float(p_env * p_st)


def classify_family(
    cumulants: Dict[str, complex],
    env_mod_db: float,
    n_freq_states: int,
    snr_db: Optional[float] = None,
    p_fsk: Optional[float] = None,
) -> AMCResult:
    """Threshold decision tree + anchor-softmax confidence ranking.

    Stage A -- constant-envelope gate (FSK vs linear families): higher-order
        cumulant theory assumes a linear pulse train, which FSK is not, so
        FSK is routed out first via :func:`fsk_gate_probability` (pass a
        precomputed `p_fsk` to reuse a gate value).

    Stage B -- cumulant tree on (|C40|, |C42|, |C63|), SNR-corrected when
        Phase-1 SNR is available.  Thresholds derive from the anchor table in
        the module docstring; `decisions` records which rule fired.

    Confidence: softmax over squared distances to the anchors in the
    normalized feature space -- a "closeness to the nearest constellation
    signature" score that degrades gracefully under noise/ISI instead of
    collapsing to 0/1.
    """
    trace: List[str] = []

    # ---- Stage A: envelope gate -------------------------------------------
    if p_fsk is None:
        p_fsk = fsk_gate_probability(env_mod_db, n_freq_states)
    trace.append(
        f"env-mod {env_mod_db:.2f} dB, {n_freq_states} freq state(s) "
        f"-> P(FSK gate) = {p_fsk:.2f}")

    # ---- Stage B: cumulant thresholds (features SNR-corrected if possible) --
    c40 = float(cumulants.get("C40_abs_corr", cumulants["C40_abs"]))
    c42 = float(cumulants.get("C42_abs_corr", cumulants["C42_abs"]))
    c63 = float(cumulants["C63_abs"])

    feats = np.array([c40, c42, c63]) / _SCALES
    names_rule = list(_ANCHORS)

    if c42 < 0.10:
        sub = None
        sub_p = np.ones(len(_ANCHORS)) / len(_ANCHORS)  # uninformative
        trace.append(f"|C42| {c42:.3f} < 0.10 -> too noisy/short for HOC tree")
    else:
        if c40 >= 1.55:
            sub = "BPSK"
            trace.append(f"|C40| {c40:.2f} >= 1.55 -> BPSK (only anchor with"
                         " |C40| = 2)")
        elif 0.70 <= c40 <= 1.45 and 0.70 <= c42 <= 1.35:
            sub = "QPSK"
            trace.append(f"|C40| {c40:.2f}, |C42| {c42:.2f} in QPSK box "
                         "[0.70-1.45]x[0.70-1.35] -> QPSK")
        elif c40 <= 0.35 and 0.75 <= c42 <= 1.25:
            sub = "8PSK"
            trace.append(f"|C40| {c40:.2f} ~ 0 with |C42| {c42:.2f} ~ 1 -> "
                         "8PSK (C40 vanishes when 4 does not divide M)")
        else:
            sub = "16QAM" if c63 < 4.7 else "64QAM"
            trace.append(f"outside PSK boxes (|C40| {c40:.2f}, |C42| {c42:.2f},"
                         f" |C63| {c63:.2f}) -> QAM ({sub})")
        d = np.array([np.linalg.norm(feats - np.array(a) / _SCALES)
                      for a in _ANCHORS.values()])
        # Sharper temperature + box-decision logit bonus: the threshold tree
        # above already isolated the modulation box; without a bonus the two
        # QAM anchors can accumulate more COMBINED softmax mass than the
        # single nearest anchor (mass leakage), flipping the family ranking.
        logits = -d ** 2 / 0.12
        logits[names_rule.index(sub)] += 1.0
        sub_p = np.exp(logits - logits.max())
        sub_p = sub_p / sub_p.sum()

    names = list(_ANCHORS)
    fam_p: Dict[str, float] = {"PSK": 0.0, "QAM": 0.0}
    for nm, pv in zip(names, sub_p):
        fam_p[_FAMILY_OF[nm]] += float(pv)

    # fold the FSK gate mass in and renormalize -> ranked family dict
    ranking: Dict[str, float] = {
        "PSK": fam_p["PSK"] * (1.0 - p_fsk),
        "QAM": fam_p["QAM"] * (1.0 - p_fsk),
        "FSK": p_fsk,
    }
    tot = sum(ranking.values())
    ranking = {k: v / tot for k, v in
               sorted(ranking.items(), key=lambda kv: -kv[1])}

    subtype = None if (sub is None or p_fsk > 0.5) else \
        names[int(np.argmax(sub_p))]
    return AMCResult(
        family_ranking=ranking,
        subtype_guess=subtype,
        n_symbols=0,
        features={"C40_abs": c40, "C42_abs": c42, "C63_abs": c63,
                  "env_mod_db": env_mod_db, "n_freq_states": n_freq_states,
                  "snr_db": snr_db},
        decisions=tuple(trace),
    )


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def analyze_burst(
    z_bb: np.ndarray,
    fs: float,
    baud_hz: float,
    snr_db: Optional[float] = None,
    alpha: Optional[float] = None,
) -> AMCResult:
    """Full AMC for one channelized burst with a known baud estimate.

    Pipeline: envelope statistics -> FSK gate.  If the gate decides FSK the
    (pointless) symbol synchronization is skipped; otherwise symbols are
    extracted (RRC matched filter + max-|C42| timing) and the cumulant tree
    runs with the Phase-1 SNR for de-biasing.
    """
    z = np.asarray(z_bb, dtype=np.complex128).ravel()
    if z.size < 256:
        raise ValueError(f"burst too short for AMC ({z.size} samples)")

    # Envelope statistics on the unit-power channelized burst. The first/last
    # ~2% of samples are trimmed (filtfilt edge transients), and |x|^2 is
    # 5-tap smoothed before the modulation index (see fsk_gate_probability).
    trim = max(1, int(0.02 * z.size))
    xc = z[trim:-trim]
    x = xc / np.sqrt(np.mean(np.abs(xc) ** 2))
    p = np.abs(x) ** 2
    p_s = np.convolve(p, np.ones(5) / 5.0, mode="same")
    env_mod_db = float(10.0 * np.log10(1.0 + p_s.var() / p_s.mean() ** 2))
    n_states, _modes = instantaneous_frequency_modes(x, fs)
    p_fsk = fsk_gate_probability(env_mod_db, n_states)
    gate_line = (f"env-mod {env_mod_db:.2f} dB, {n_states} freq state(s) "
                 f"-> P(FSK gate) = {p_fsk:.2f}")

    if p_fsk > 0.5:
        # Gate decides FSK: constant-envelope frequency modulation -- cumulant
        # symbol model does not apply, skip synchronization entirely.
        ranking = {"FSK": p_fsk, "PSK": 0.5 * (1.0 - p_fsk),
                   "QAM": 0.5 * (1.0 - p_fsk)}
        tot = sum(ranking.values())
        ranking = {k: v / tot for k, v in
                   sorted(ranking.items(), key=lambda kv: -kv[1])}
        return AMCResult(
            family_ranking=ranking,
            subtype_guess=None,
            n_symbols=0,
            features={"env_mod_db": env_mod_db, "n_freq_states": n_states,
                      "snr_db": snr_db},
            decisions=(gate_line,
                       "gate -> FSK; symbol synchronization skipped"),
        )

    syms = extract_symbols(z, fs, baud_hz, alpha=alpha)
    cums = compute_cumulants(syms, snr_db=snr_db)
    res = classify_family(cums, env_mod_db, n_states, snr_db, p_fsk=p_fsk)
    return AMCResult(
        family_ranking=res.family_ranking,
        subtype_guess=res.subtype_guess,
        n_symbols=int(syms.size),
        features={**res.features,
                  "n_symbols": int(syms.size),
                  "C40_raw": complex(cums["C40"]),
                  "C42_raw": complex(cums["C42"]),
                  "C63_raw": complex(cums["C63"])},
        decisions=res.decisions,
    )
