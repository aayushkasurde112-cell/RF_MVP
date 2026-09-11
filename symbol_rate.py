"""
symbol_rate.py -- Symbol-rate (baud) estimation (Phase 2, NTRO PS-26147)
========================================================================

Estimates the symbol rate of a *single* complex-baseband burst (one
:class:`spectral.Detection` after channelization) using a battery of
cyclostationary "spectral-line" methods, scored on a common scale
(line SNR in dB) so the best survivor wins:

1. ``envelope_spectrum``   FFT of |x|^2, |x| and x^2.  For linearly modulated
                           signals with excess bandwidth (roll-off alpha > 0)
                           the (squared) envelope is periodic at the symbol
                           rate -> a discrete tone at f_baud.  Works on QAM
                           and PSK alike.
2. ``phase_transition``    |angle(x[n] x*[n-1])| spikes at every symbol
                           boundary (phase jumps / frequency-state switches),
                           so its FFT carries a comb with fundamental f_baud.
                           Catches FSK and zero-roll-off PSK where the
                           envelope is flat.
3. ``delay_multiply``      x(t) * conj(x(t-tau)) -- the classic Gardner /
                           cyclic-autocorrelation tone at cyclic frequency
                           alpha = f_baud.  Survives alpha = 0 (rectangular
                           pulses); tau is swept over a geometric grid, with
                           an FSK-aware candidate tau = 1/(2*d_tone) derived
                           from the instantaneous-frequency mode spacing
                           (then every tone difference fi-fj maps to +/-1,
                           making the product data-independent).
4. ``zero_crossing``       Hysteretic zero-crossing count of the
                           mean-removed transform (envelope^2 or the
                           delay product).  Cheap, robust on very short
                           bursts where FFT resolution is marginal; used as
                           the last-resort fallback.

Harmonic folding guards against locking onto 2*f_baud: if the spectrum shows
a comparably strong line at f/2, the estimate is folded down.

Entry points
------------
``estimate_symbol_rate(samples, fs, ...) -> float``      (contract: float;
NaN when no credible baud line exists, e.g. a CW carrier)
``estimate_symbol_rate_report(...) -> SymbolRateEstimate``  (method, score,
confidence, candidates -- what the pipeline and logs consume)

All functions accept the normalized complex64 arrays produced by Phase 1
(``ingestion.load_*`` -> ``spectral.channelize_burst``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Reuse Phase-1 machinery: parabolic peak refinement + spectrum line lookup.
from spectral import estimate_carrier_frequency

log = logging.getLogger(__name__)

__all__ = [
    "SymbolRateEstimate",
    "estimate_symbol_rate",
    "estimate_symbol_rate_report",
    "instantaneous_frequency_modes",
]

#: minimum line SNR (dB) for a candidate to be believed at all
MIN_LINE_SNR_DB = 8.0
#: penalty (dB) applied to less-robust methods when ranking ties
_METHOD_PENALTY_DB = {"zero_crossing": 2.0, "phase_transition": 1.0}


@dataclass(frozen=True)
class SymbolRateEstimate:
    """Result of the baud search (report form; see ``estimate_symbol_rate``)."""

    baud_hz: float              # refined estimate (NaN if not found)
    method: str                 # winning method ("none" if nothing credible)
    line_snr_db: float          # tone height above the local PSD floor
    confidence: float           # 0..1 logistic mapping of line SNR
    candidates: Tuple[Tuple[str, float, float], ...] = ()  # (method, Hz, dB)
    detail: Dict = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return self.method != "none" and bool(np.isfinite(self.baud_hz)) \
            and self.baud_hz > 0


# --------------------------------------------------------------------------- #
# Spectrum-line helpers
# --------------------------------------------------------------------------- #
def _band_slice(freqs: np.ndarray, f_min: float, f_max: float) -> Tuple[int, int]:
    """Contiguous [start, stop) index window of the (ascending) band."""
    return int(np.searchsorted(freqs, f_min)), int(np.searchsorted(freqs, f_max))


def _line_search(
    freqs: np.ndarray, psd_db: np.ndarray, f_min: float, f_max: float
) -> Tuple[float, float]:
    """Peak search + parabolic refine inside [f_min, f_max].

    Returns (f_peak_hz, line_snr_db) where the score is the peak height above
    the *median* PSD of the band -- the same robust-floor logic as Phase 1's
    energy detector, applied to the transformed-signal spectrum.
    """
    s, e = _band_slice(freqs, f_min, f_max)
    if e - s < 8:
        return float("nan"), -np.inf
    seg = psd_db[s:e]
    coarse, refined = estimate_carrier_frequency(freqs, psd_db, (s, e))
    # Score against a LOCAL median floor around the peak, not the band-wide
    # median: the transforms carry a strong low-frequency bulk (envelope /
    # autocorrelation tails), which would otherwise masquerade as a huge
    # "line" when compared to a distant, quiet floor. For a genuine narrow
    # tone the local median IS the noise floor; for a broad hill the local
    # median rises with it, correctly yielding a low score.
    k0 = s + int(np.argmax(seg))
    w = max(32, (e - s) // 20)
    lo, hi = max(s, k0 - w), min(e, k0 + w + 1)
    floor_local = float(np.median(psd_db[lo:hi]))
    return float(refined), float(psd_db[k0] - floor_local)


def _psd_at(freqs: np.ndarray, psd_db: np.ndarray, f: float) -> float:
    """Interpolated PSD value at frequency f (NaN-safe lookup)."""
    return float(np.interp(f, freqs, psd_db,
                           left=-np.inf, right=-np.inf))


def _positive_spectrum(w: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided (f >= 0) power spectrum in dB of a real OR complex signal.

    numpy >= 2.0 rfft() rejects complex input, so all transforms (including
    the complex x^2 and delay products) go through the full complex FFT and
    keep the non-negative-frequency half, which is already in ascending
    order on the fftfreq axis.
    """
    w = np.asarray(w, dtype=np.complex128)
    spec = np.abs(np.fft.fft(w)) ** 2
    freqs = np.fft.fftfreq(w.size, d=1.0 / fs)
    keep = freqs >= 0
    psd_db = 10.0 * np.log10(np.maximum(spec[keep], np.finfo(float).tiny))
    return freqs[keep].copy(), psd_db


def _fold_harmonic(
    freqs: np.ndarray, psd_db: np.ndarray, f: float, f_min: float
) -> float:
    """If a comparably strong line exists at f/2 (or f/3), prefer it.

    The nonlinear transforms produce tones at k*f_baud (k = 1, 2, ...); a pure
    argmax can lock onto the 2nd harmonic.  Folding is conservative: the
    sub-harmonic must be within 5 dB of the harmonic *and* above the floor --
    otherwise noise at f/2 would drag genuine estimates down.
    """
    for div in (2, 3):
        f_sub = f / div
        if f_sub <= f_min:
            continue
        if _psd_at(freqs, psd_db, f_sub) >= _psd_at(freqs, psd_db, f) - 5.0:
            return f_sub
    return f


# --------------------------------------------------------------------------- #
# Instantaneous-frequency modes (FSK tone spacing + FSK gate input)
# --------------------------------------------------------------------------- #
def instantaneous_frequency_modes(
    z: np.ndarray, fs: float, n_bins: int = 129
) -> Tuple[int, np.ndarray]:
    """Count + locate discrete frequency states of a constant-envelope burst.

    Histograms the instantaneous frequency f(t) = d/dt unwrap(angle z) / 2pi,
    smooths it, and counts prominent modes.  4FSK -> 4 modes ~ equally
    spaced; PSK/QAM -> a single central lobe (residual carrier).

    Returns (n_modes, mode_frequencies_hz).
    """
    phi = np.unwrap(np.angle(np.asarray(z, dtype=np.complex128)))
    f_inst = np.diff(phi) / (2.0 * np.pi) * float(fs)
    med = float(np.median(f_inst))
    mad = float(np.median(np.abs(f_inst - med))) * 1.4826
    span = max(6.0 * mad, fs * 1e-3)
    lo, hi = med - span, med + span
    hist, edges = np.histogram(f_inst, bins=n_bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    # Light Gaussian smoothing so histogram raggedness doesn't split one
    # state into several peaks.
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    hist_s = gaussian_filter1d(hist, sigma=2.0)
    peaks, _ = find_peaks(
        hist_s, prominence=max(0.15 * hist_s.max(), 1e-9),
        distance=max(2, n_bins // 16),
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    modes = np.sort(centers[peaks])
    return len(modes), modes


# --------------------------------------------------------------------------- #
# Method 1: envelope / squared-signal spectrum
# --------------------------------------------------------------------------- #
def _envelope_candidates(
    z: np.ndarray, fs: float, f_min: float, f_max: float
) -> Tuple[List[Tuple[float, float, str]], Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Baud candidates from |x|^2, |x| and x^2 transforms.

    Why three transforms: |x|^2 is the statistically strongest line for shaped
    QAM/PSK (alpha > 0); |x| sometimes wins when amplitude ripple dominates;
    x^2 removes the data phase of BPSK entirely (a_k^2 = 1) giving a very
    strong line -- but collapses for QPSK/QAM, so it is just one more voter.
    """
    out: List[Tuple[float, float, str]] = []
    best_spec: Optional[Tuple[np.ndarray, np.ndarray]] = None
    transforms = (
        ("env_sq", np.abs(z) ** 2),
        ("env_abs", np.abs(z)),
        ("square", z ** 2),
    )
    for tag, w in transforms:
        w = w - np.mean(w)                      # kill DC (transform's k=0 term)
        freqs, psd_db = _positive_spectrum(w, fs)
        f, snr = _line_search(freqs, psd_db, f_min, f_max)
        if np.isfinite(f):
            f = _fold_harmonic(freqs, psd_db, f, f_min)
            out.append((f, snr, f"envelope_spectrum[{tag}]"))
            if best_spec is None or snr > best_spec[0]:
                best_spec = (snr, (freqs, psd_db))
    return out, (best_spec[1] if best_spec else None)


# --------------------------------------------------------------------------- #
# Method 2: phase-transition spectrum (FSK / zero-roll-off friendly)
# --------------------------------------------------------------------------- #
def _phase_transition_candidate(
    z: np.ndarray, fs: float, f_min: float, f_max: float
) -> Tuple[float, float]:
    """Line at f_baud in the symbol-boundary switching spectrum.

    s[n] = |angle(x[n] conj(x[n-1]))| is ~constant within a symbol (residual
    carrier) and spikes at every symbol boundary (PSK phase jump, FSK
    frequency-state switch).  A train of spikes at the symbol rate has a
    deterministic Fourier component at f_baud -- even when the envelope is
    perfectly flat, which is exactly where method 1 goes blind.
    """
    dphi = np.abs(np.angle(np.asarray(z[1:], dtype=np.complex128)
                           * np.conj(z[:-1])))
    s = dphi - np.median(dphi)
    freqs, psd_db = _positive_spectrum(s, fs)
    f, snr = _line_search(freqs, psd_db, f_min, f_max)
    if np.isfinite(f):
        f = _fold_harmonic(freqs, psd_db, f, f_min)
    return f, snr


# --------------------------------------------------------------------------- #
# Method 3: delay-and-multiply (cyclic autocorrelation tone)
# --------------------------------------------------------------------------- #
def _delay_multiply_candidates(
    z: np.ndarray,
    fs: float,
    f_min: float,
    f_max: float,
    n_modes: int = 0,
    modes: Optional[np.ndarray] = None,
    max_taus: int = 14,
) -> Tuple[List[Tuple[float, float, str]], Dict]:
    """Cyclic-autocorrelation line search over a lag grid.

    For d(t) = x(t) conj(x(t-tau)) the cyclic statistic at cycle frequency
    alpha = f_baud is nonzero for ANY pulse shape -- including alpha = 0
    (rectangular), where the envelope spectrum is silent (|x|^2 is constant
    within a symbol).  The tone amplitude peaks near tau = T/2; since T is
    the unknown, sweep tau geometrically and additionally try the FSK-aware
    lag tau = 1/(2*d_tone) (computed from the instantaneous-frequency mode
    spacing): at that lag every tone difference maps to +/-1 and the product
    becomes data-independent inside symbols.
    """
    x = np.asarray(z, dtype=np.complex128)
    n = x.size
    tau_min = max(2, int(fs / (2.0 * f_max)))
    tau_max = min(n // 4, max(tau_min + 4, int(fs / (2.0 * f_min))))
    taus = sorted(set(np.unique(
        np.geomspace(tau_min, max(tau_min + 1, tau_max), max_taus).astype(int)
    ).tolist()))
    if n_modes >= 2 and modes is not None and len(modes) >= 2:
        # FSK-aware lag: spacing of the instantaneous-frequency modes.
        d_tone = float(np.median(np.diff(modes)))
        tau_fsk = int(round(0.5 * fs / max(d_tone, 1.0)))
        if tau_min <= tau_fsk <= tau_max:
            taus.append(tau_fsk)
            taus = sorted(set(taus))

    out: List[Tuple[float, float, str]] = []
    for tau in taus:
        d = x[tau:] * np.conj(x[:-tau])
        d = d - np.mean(d)
        freqs, psd_db = _positive_spectrum(d, fs)
        f, snr = _line_search(freqs, psd_db, f_min, f_max)
        if np.isfinite(f):
            f = _fold_harmonic(freqs, psd_db, f, f_min)
            out.append((f, snr, f"delay_multiply[tau={tau}]"))
    return out, {"n_taus_tried": len(taus)}


# --------------------------------------------------------------------------- #
# Method 4: hysteretic zero-crossing (last-resort fallback)
# --------------------------------------------------------------------------- #
def _zero_crossing_candidate(
    s: np.ndarray, fs: float, freqs: np.ndarray, psd_db: np.ndarray
) -> Tuple[float, float]:
    """Fundamental frequency of a real transform via hysteretic ZC counting.

    Counting rising edges through a +-h hysteresis band (h = 0.5 sigma) makes
    the count immune to noise chatter around zero.  Scored by looking the
    resulting frequency up in the companion PSD so it competes on the same
    dB scale as the FFT methods.
    """
    s = np.asarray(s, dtype=np.float64)
    s = s - np.median(s)
    h = 0.5 * float(np.std(s))
    if h <= 0:
        return float("nan"), -np.inf
    above = np.flatnonzero(s > +h)
    below = np.flatnonzero(s < -h)
    if above.size < 2 or below.size < 2:
        return float("nan"), -np.inf
    # Rising event = sample above +h whose count of prior below-dips
    # exceeds that of the previous above-sample (enforces a full excursion).
    dips_before = np.searchsorted(below, above, side="right")
    prev = np.concatenate([[-1], dips_before[:-1]])
    events = above[dips_before > prev]
    if events.size < 2:
        return float("nan"), -np.inf
    f = events.size * fs / s.size
    if not (freqs[0] <= f <= freqs[-1]):
        return float("nan"), -np.inf
    # score vs a LOCAL median floor (same rationale as _line_search): the
    # transforms carry a low-frequency bulk that poisons a global median.
    k0 = int(np.searchsorted(freqs, f))
    w = max(32, freqs.size // 40)
    lo, hi = max(0, k0 - w), min(freqs.size, k0 + w + 1)
    return float(f), float(_psd_at(freqs, psd_db, f)
                          - float(np.median(psd_db[lo:hi])))


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def estimate_symbol_rate_report(
    samples: np.ndarray,
    sample_rate: float,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
    min_line_snr_db: float = MIN_LINE_SNR_DB,
) -> SymbolRateEstimate:
    """Full battery baud estimate for one channelized burst.

    Parameters
    ----------
    samples : complex baseband of the burst (from spectral.channelize_burst).
    sample_rate : Hz of `samples`.
    f_min, f_max : plausible baud band; defaults [fs/500, fs/4].  The upper
        limit respects the Nyquist pulse constraint (baud <= fs/2, and shaped
        signals need excess BW), the lower avoids the DC/carrier-recovery
        region of the transform spectra.
    min_line_snr_db : floor under which "no line" is declared (CW carriers,
        unmodulated tones) -> baud NaN, method "none".
    """
    z = np.asarray(samples, dtype=np.complex128).ravel()
    fs = float(sample_rate)
    if z.size < 256:
        return SymbolRateEstimate(float("nan"), "none", -np.inf, 0.0,
                                  detail={"reason": "burst too short"})
    if fs <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    f_min = float(f_min) if f_min else fs / 500.0
    f_max = float(f_max) if f_max else fs / 4.0
    if not (0 < f_min < f_max <= fs / 2):
        raise ValueError(f"invalid baud band [{f_min}, {f_max}] for fs={fs}")

    n_modes, modes = instantaneous_frequency_modes(z, fs)

    cand: List[Tuple[float, float, str]] = []
    detail: Dict = {"freq_modes": int(n_modes)}

    # --- method 1: envelope family -----------------------------------------
    env, env_spec = _envelope_candidates(z, fs, f_min, f_max)
    cand += env

    # --- method 2: phase-transition spectrum --------------------------------
    f_pt, snr_pt = _phase_transition_candidate(z, fs, f_min, f_max)
    if np.isfinite(f_pt):
        cand.append((f_pt, snr_pt, "phase_transition"))

    # --- method 3: delay-and-multiply / cyclic autocorrelation --------------
    dm, dm_detail = _delay_multiply_candidates(
        z, fs, f_min, f_max, n_modes=n_modes, modes=modes)
    detail.update(dm_detail)
    cand += dm

    # --- method 4: zero-crossing on the two most promising real transforms --
    if env_spec is not None:
        best_transform = (np.abs(z) ** 2 - np.mean(np.abs(z) ** 2))
        f_zc, snr_zc = _zero_crossing_candidate(
            best_transform, fs, env_spec[0], env_spec[1])
        if np.isfinite(f_zc) and f_min <= f_zc <= f_max:
            cand.append((f_zc, snr_zc - _METHOD_PENALTY_DB["zero_crossing"],
                         "zero_crossing"))

    if not cand:
        return SymbolRateEstimate(float("nan"), "none", -np.inf, 0.0,
                                  detail=detail)
    detail["n_candidates"] = len(cand)

    # --- pick the strongest line (with method-robustness penalties) --------
    scored = sorted(cand, key=lambda c: c[1], reverse=True)
    f_best, snr_best, method_best = scored[0]

    if snr_best < min_line_snr_db:
        log.info("no credible baud line (best %.1f dB < %.1f dB)",
                 snr_best, min_line_snr_db)
        return SymbolRateEstimate(float("nan"), "none", float(snr_best), 0.0,
                                  candidates=tuple((m, round(f, 2), round(s, 1))
                                                   for f, s, m in scored[:5]),
                                  detail=detail)

    conf = 1.0 / (1.0 + math.exp(-(snr_best - 14.0) / 4.0))
    return SymbolRateEstimate(
        baud_hz=float(f_best),
        method=method_best.split("[")[0],
        line_snr_db=float(snr_best),
        confidence=float(min(max(conf, 0.0), 1.0)),
        candidates=tuple((m.split("[")[0], round(f, 2), round(s, 1))
                         for f, s, m in scored[:5]),
        detail=detail,
    )


def estimate_symbol_rate(
    samples: np.ndarray,
    sample_rate: float,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
    min_line_snr_db: float = MIN_LINE_SNR_DB,
) -> float:
    """Contract-required float API: estimated baud rate in Hz.

    Returns ``float('nan')`` when no credible symbol-rate line is found
    (pure carrier, noise-only burst) -- callers should check with
    ``np.isfinite``.  See :func:`estimate_symbol_rate_report` for method,
    score and candidates.
    """
    rep = estimate_symbol_rate_report(samples, sample_rate, f_min, f_max,
                                      min_line_snr_db)
    return float(rep.baud_hz)
