"""
spectral.py -- Signal detection & spectral characterization (Phase 1)
=====================================================================

Chain (all complex baseband, float64 internally):

    Welch PSD  ->  robust noise floor (median / MAD)  ->  CFAR-style energy
    threshold  ->  contiguous-region detection (wrap-aware)  ->  per-region
    carrier estimate (FFT peak + 3-point parabolic refine)  ->  x-dB-down
    bandwidth.

Every function is pure (arrays in -> results out) so the same code runs
offline on a loaded :class:`~ingestion.IQData` or per-chunk inside a
streaming detector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy import signal as sp_signal

log = logging.getLogger(__name__)

__all__ = [
    "NoiseFloor",
    "Detection",
    "SpectralReport",
    "welch_psd",
    "estimate_noise_floor",
    "find_signal_regions",
    "estimate_carrier_frequency",
    "estimate_bandwidth_xdb",
    "characterize_spectrum",
]


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NoiseFloor:
    """Robust noise-floor statistics of the PSD (dB/Hz domain)."""

    median_db: float    # estimated noise floor (PSD median)
    sigma_db: float     # 1.4826 * MAD -- Gaussian-consistent spread
    threshold_db: float  # median + k_sigma * sigma  (energy-detection rule)
    k_sigma: float


@dataclass(frozen=True)
class Detection:
    """One detected signal emission, in Hz relative to baseband center."""

    carrier_hz: float        # refined peak frequency (parabolic-interpolated)
    center_hz: float         # midpoint of the x-dB-down edges
    bandwidth_hz: float      # x-dB-down width (or threshold-region fallback)
    f_low_hz: float          # lower -x dB edge
    f_high_hz: float         # upper -x dB edge
    peak_power_db: float     # PSD at the peak [dB/Hz]
    snr_db: float            # peak power above the robust noise floor
    bandwidth_method: str = "x-dB-down"
    peak_bin_hz: float = 0.0  # coarse FFT-bin quantized estimate (pre-refine)


@dataclass(frozen=True)
class SpectralReport:
    """Full spectral characterization of one capture."""

    sample_rate_hz: float
    nfft: int
    noise: NoiseFloor
    detections: Tuple[Detection, ...]
    freqs_hz: np.ndarray  # ascending, [-fs/2, fs/2)
    psd_db: np.ndarray    # Welch PSD, dB/Hz, aligned with freqs_hz

    @property
    def threshold_db(self) -> float:
        return self.noise.threshold_db

    def summary_lines(self) -> List[str]:
        """Human-readable summary (used by main.py / REST reporting)."""
        out = [
            f"noise floor {self.noise.median_db:7.1f} dB/Hz   "
            f"sigma_MAD {self.noise.sigma_db:5.2f} dB   "
            f"threshold {self.noise.threshold_db:7.1f} dB/Hz "
            f"(median + {self.noise.k_sigma:g} sigma)",
        ]
        if not self.detections:
            out.append("  no signals above threshold")
        for i, d in enumerate(self.detections, 1):
            out.append(
                f"  detection #{i}: carrier {d.carrier_hz / 1e3:+10.2f} kHz  "
                f"center {d.center_hz / 1e3:+10.2f} kHz  "
                f"bw[{d.bandwidth_method}] {abs(d.bandwidth_hz) / 1e3:8.2f} kHz  "
                f"peak {d.peak_power_db:7.1f} dB/Hz  SNR {d.snr_db:5.1f} dB"
            )
        return out


# --------------------------------------------------------------------------- #
# 1) Welch PSD
# --------------------------------------------------------------------------- #
def welch_psd(
    samples: np.ndarray,
    sample_rate: float,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = "hann",
    detrend: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Two-sided Welch PSD of a complex-baseband capture.

    Parameters & DSP rationale
    --------------------------
    * Complex input -> ``return_onesided=False``; the PSD spans
      [-fs/2, fs/2) and negative frequencies are *meaningful* for IQ data.
    * Hann window + 50 % overlap: the Hann mainlobe/sidelobe trade
      (-31.5 dB first sidelobe) suppresses spectral leakage between
      emissions, and 50 % overlap with Hann keeps the averaged Welch
      estimator variance low (effectively ~2x the segment count).
    * ``detrend=False`` keeps the DC bin honest: ingestion already removed
      LO leakage, and we still want to *see* a residual spur if it returns.
    * ``scaling='density'`` -> V^2/Hz; returned as 10*log10 -> dB/Hz.
    * ``nperseg=None`` -> power of two giving >= ~8 segment averages
      (capped at 4096) -- good variance reduction without over-smoothing
      narrowband features.

    Returns
    -------
    (freqs_hz, psd_db) : ascending frequency axis in [-fs/2, fs/2), dB/Hz PSD.
    """
    z = np.asarray(samples)
    if z.ndim != 1 or z.size < 32:
        raise ValueError(f"need a 1-D capture of >= 32 samples, got {np.shape(z)}")
    fs = float(sample_rate)
    if fs <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    z = z.astype(np.complex128 if np.iscomplexobj(z) else np.float64, copy=False)
    if nperseg is None:
        nperseg = int(2 ** np.clip(np.floor(np.log2(max(z.size // 8, 64))), 7, 12))
    nperseg = int(min(nperseg, z.size))
    noverlap = nperseg // 2 if noverlap is None else int(noverlap)

    f, pxx = sp_signal.welch(
        z,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        scaling="density",
        detrend=detrend,
    )
    # scipy emits two-sided output in FFT order (f[0]=0, ..., f[N/2]=-fs/2);
    # fftshift BOTH arrays together -> ascending axis in [-fs/2, fs/2) with
    # psd_db[i] aligned to freqs_hz[i].
    f = np.fft.fftshift(f)
    pxx = np.fft.fftshift(pxx)
    psd_db = 10.0 * np.log10(np.maximum(pxx, np.finfo(np.float64).tiny))
    return f.astype(np.float64), psd_db


# --------------------------------------------------------------------------- #
# 2) Robust noise floor + energy-detection threshold (median / MAD)
# --------------------------------------------------------------------------- #
def estimate_noise_floor(psd_db: np.ndarray, k_sigma: float = 6.0,
                         sigma_cap_db: Optional[float] = None) -> NoiseFloor:
    """Median/MAD noise-floor estimate and CFAR-style detection threshold.

    WHY median/MAD instead of a fixed dB offset:
      * Per-bin periodogram power of Gaussian noise is chi^2-distributed, and
        after Welch averaging the dB-valued floor fluctuates roughly Gaussian
        *around* its median. A strong emission, however, raises only the few
        bins it occupies. The **median** is a breakdown-robust estimator: it
        stays pinned to the noise level even when up to ~50 % of bins carry
        signal power, whereas the mean would be biased upward by exactly the
        signals we want to detect.
      * **MAD** (median absolute deviation) inherits that robustness and is
        converted to a Gaussian-equivalent sigma via the consistency constant
        1/Phi^-1(3/4) ~= 1.4826.
      * ``threshold = median + k * sigma_MAD`` is therefore a
        **constant-false-alarm-rate (CFAR) energy detector**: under the
        Gaussian approximation, k = 6 gives a per-bin P_fa ~ 1e-9, and the
        rule self-calibrates against gain/temperature/bandwidth changes that
        would silently break any fixed-dB threshold.

    Assumptions & failure mode: the model needs ONE dominant noise
    population. Analytic (Hilbert-derived) signals violate it -- their
    negative-frequency half is structurally dead, making the PSD bimodal and
    inflating sigma_MAD; callers should scope those to the live half band
    (see :func:`characterize_spectrum`'s ``analysis_band``). We warn loudly
    when sigma_MAD looks pathological (> 15 dB; averaged thermal noise in dB
    domain spreads ~1-3 dB even with only a few segment averages).

    A degenerate PSD (MAD -> 0, e.g. synthetic noiseless captures) falls back
    to a nominal 0.5 dB spread so the detector never locks wide open.
    """
    p = np.asarray(psd_db, dtype=np.float64)
    if p.size == 0:
        raise ValueError("empty PSD")
    med = float(np.median(p))
    mad = float(np.median(np.abs(p - med)))
    sigma = 1.4826 * mad

    # --- dominant-population (mode) refinement -------------------------------
    # A global median/MAD assumes the noise is the *majority* population. For
    # heavy-tailed PSDs (sinc^2 skirts of unshaped bursts, FM sidelobes) the
    # skirt mass inflates MAD and the threshold lands ABOVE the signal peak.
    # Fix: locate the histogram mode of the PSD (the dominant population =
    # noise for typical burst duty cycles) and re-estimate floor+MAD using
    # only bins within +-6 dB of that mode.
    if p.size >= 256:
        hist, edges = np.histogram(p, bins=128)
        hist = np.convolve(hist, np.ones(3), mode="same")  # smooth ragged bins
        k_mode = int(np.argmax(hist))
        mode = float(0.5 * (edges[k_mode] + edges[k_mode + 1]))
        near = np.abs(p - mode) <= 6.0
        # apply only when the mode population sits clearly BELOW the global
        # median (i.e. noise is a minority -- the heavy-tail case) ...
        if near.sum() >= 64 and (med - mode) > 3.0:
            floor = float(np.median(p[near]))
            sigma = 1.4826 * float(np.median(np.abs(p[near] - floor)))
            med = floor
            log.debug("mode-based noise floor %.1f dB (global median %.1f)",
                      floor, med)

    if sigma > 15.0:
        log.warning(
            "sigma_MAD = %.1f dB -- PSD looks bimodal (signal on <50%% of bins, "
            "or an analytic signal with a dead half-band). Noise statistics may "
            "be unreliable; consider restricting analysis_band.", sigma,
        )
    # --- Welch-statistics sigma cap ------------------------------------------
    # For pure noise, the dB-spread of an averaged Welch PSD is bounded:
    # 10*log10(chi2_A/A) has std ~= 5.57/sqrt(A) dB for A effective segment
    # averages.  A measured sigma far above that bound means CONTAMINATION
    # (signal skirts), not noise variance -- trusting it would push the CFAR
    # threshold above real signal peaks (e.g. sinc^2 spectra of unshaped
    # bursts).  Capping sigma at the physically-achievable noise spread keeps
    # the detector sensitive while staying CFAR-safe: true noise bins still
    # sit within ~6*sigma_true of the floor (P_fa ~ 1e-9/bin).
    if sigma_cap_db is not None and sigma > sigma_cap_db:
        log.debug("sigma_MAD %.2f dB capped at Welch bound %.2f dB",
                  sigma, sigma_cap_db)
        sigma = float(sigma_cap_db)

    if sigma < 1e-3:
        log.debug("MAD underflow (%.2e dB) -- using nominal 0.5 dB sigma", mad)
        sigma = 0.5
    return NoiseFloor(
        median_db=med,
        sigma_db=sigma,
        threshold_db=med + k_sigma * sigma,
        k_sigma=float(k_sigma),
    )


# --------------------------------------------------------------------------- #
# 3) Region detection (wrap-aware)
# --------------------------------------------------------------------------- #
def _true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Half-open [start, stop) runs of True values in a boolean mask."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0 or not m.any():
        return []
    d = np.diff(m.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    stops = list(np.flatnonzero(d == -1) + 1)
    if m[0]:
        starts.insert(0, 0)
    if m[-1]:
        stops.append(m.size)
    return list(zip(starts, stops))


def find_signal_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous above-threshold regions of the two-sided PSD.

    The spectrum of complex baseband is *circular*: an emission sitting on DC
    (or straddling +-fs/2) wraps from the last bin back to the first. Such
    leading/trailing runs are stitched into a single region whose ``start``
    may be negative -- index the PSD with ``% N`` when consuming it.
    """
    runs = _true_runs(mask)
    n = int(np.size(mask))
    if len(runs) >= 2 and runs[0][0] == 0 and runs[-1][1] == n:
        _, e_first = runs[0]
        s_last, _ = runs[-1]
        merged = (s_last - n, e_first)  # e.g. (-3, 5): bins -3..-1 and 0..4
        runs = runs[1:-1] + [merged]
        runs.sort(key=lambda r: r[0])
    return runs


# --------------------------------------------------------------------------- #
# 4) Carrier frequency: FFT peak search + parabolic refinement
# --------------------------------------------------------------------------- #
def estimate_carrier_frequency(
    freqs_hz: np.ndarray,
    psd_db: np.ndarray,
    region: Optional[Tuple[int, int]] = None,
) -> Tuple[float, float]:
    """Peak-search the carrier inside a PSD region.

    Coarse estimate = argmax of the PSD over the region. A pure FFT peak is
    quantized to the bin spacing df = fs/NFFT, which is far too coarse for a
    parameter-extraction deliverable -- so we refine with a **3-point
    parabolic interpolation in the log-power domain**: the mainlobe of a
    windowed sinusoid is log-parabolic to first order, and the vertex of the
    parabola through (k-1, k, k+1) locates the true peak to ~1/10 bin for
    typical windows::

        delta = 0.5 * (P[k-1] - P[k+1]) / (P[k-1] - 2 P[k] + P[k+1])

    Returns
    -------
    (coarse_hz, refined_hz)
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    p = np.asarray(psd_db, dtype=np.float64)
    n = p.size
    if region is None:
        region = (0, n)
    s, e = region
    idx = np.arange(s, e) % n  # supports wrap-around regions (negative start)
    k = int(idx[int(np.argmax(p[idx]))])
    coarse = float(f[k])
    if k <= 0 or k >= n - 1:  # no left/right neighbour to interpolate with
        return coarse, coarse
    a, b, c = float(p[k - 1]), float(p[k]), float(p[k + 1])
    denom = a - 2.0 * b + c
    if abs(denom) < 1e-12:
        return coarse, coarse
    delta = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
    return coarse, coarse + delta * float(f[1] - f[0])


# --------------------------------------------------------------------------- #
# 5) Bandwidth: x-dB-down walk with interpolated crossings
# --------------------------------------------------------------------------- #
def estimate_bandwidth_xdb(
    freqs_hz: np.ndarray,
    psd_db: np.ndarray,
    peak_bin: int,
    x_db: float = 10.0,
    smooth_bins: int = 3,
    max_span_bins: Optional[int] = None,
) -> Tuple[float, float, float]:
    """x-dB-down bandwidth around a PSD peak.

    Walk outward from the peak until the PSD falls below ``peak - x_db``,
    then **linearly interpolate** the exact crossing between the last bin
    above and first bin below the shoulder -- bin-quantized walk-ups would
    otherwise quantize the bandwidth to multiples of df.

    The PSD is lightly median-filtered (default 3 bins) first: a single noise
    notch inside the occupied band would otherwise trigger a premature
    crossing, while a long moving average would inflate the width. Median
    filtering removes isolated notches without broadening edges.

    Returns
    -------
    (f_low_hz, f_high_hz, bandwidth_hz)
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    n = f.size
    p = np.asarray(psd_db, dtype=np.float64)
    if smooth_bins and smooth_bins > 1:
        p = sp_signal.medfilt(p, kernel_size=int(smooth_bins) | 1)

    # Re-locate the peak after smoothing (it can shift by a bin or two).
    k = int(peak_bin) % n
    lo_w = max(k - 2, 0)
    k = lo_w + int(np.argmax(p[lo_w : min(k + 3, n)]))

    df = float(f[1] - f[0])
    target = float(p[k]) - float(x_db)
    max_span = max_span_bins if max_span_bins is not None else n // 2

    # --- walk left ---
    i = k
    while i > 0 and (k - i) < max_span and p[i - 1] > target:
        i -= 1
    if i == 0:
        f_lo = float(f[0])
        log.debug("x-dB walk hit lower spectrum edge; bandwidth may be clipped")
    else:
        frac = (p[i] - target) / max(float(p[i] - p[i - 1]), 1e-12)
        f_lo = float(f[i] - np.clip(frac, 0.0, 1.0) * df)

    # --- walk right ---
    j = k
    while j < n - 1 and (j - k) < max_span and p[j + 1] > target:
        j += 1
    if j == n - 1:
        f_hi = float(f[-1])
        log.debug("x-dB walk hit upper spectrum edge; bandwidth may be clipped")
    else:
        frac = (p[j] - target) / max(float(p[j] - p[j + 1]), 1e-12)
        f_hi = float(f[j] + np.clip(frac, 0.0, 1.0) * df)

    return f_lo, f_hi, f_hi - f_lo


# --------------------------------------------------------------------------- #
# 6) Orchestrator
# --------------------------------------------------------------------------- #
def characterize_spectrum(
    samples: np.ndarray,
    sample_rate: float,
    *,
    k_sigma: float = 6.0,
    x_db: float = 10.0,
    min_region_bins: int = 3,
    close_bins: int = 2,
    analysis_band: Optional[Tuple[float, float]] = None,
    nperseg: Optional[int] = None,
) -> SpectralReport:
    """One-shot detection + parameter extraction over the whole capture.

    Parameters
    ----------
    k_sigma : CFAR multiplier for the median/MAD threshold (see
              :func:`estimate_noise_floor`).
    x_db : shoulder level for the bandwidth measurement (3 dB ~ half-power
           for a tone; 10 dB a robust default for modulated emissions;
           regulatory masks often quote 26 dB).
    min_region_bins : regions narrower than this are treated as threshold
                      outliers (single-bin noise pops) and discarded.
    close_bins : binary-closing radius (in bins) applied to the threshold
                 mask before region growing -- bridges 1-2 bin noise notches
                 inside an occupied band so one emission is not split into
                 several detections. 0 disables.
    analysis_band : (f_lo, f_hi) Hz restriction of the analysis, in baseband
                    coordinates. Use it when part of the spectrum is
                    *structurally* dead -- e.g. ``(0, fs/2)`` for an analytic
                    signal from a mono-WAV Hilbert transform, whose negative
                    half carries no noise either -- so the noise-floor
                    statistics see a single population. A wrapping band
                    (f_lo > f_hi) selects the two band edges.
    """
    f, p = welch_psd(samples, sample_rate, nperseg=nperseg)
    # effective number of Welch segment averages -> theoretical noise spread
    nper = p.size and (2 * len(samples)) // p.size  # ~nperseg
    n_avg = max(1, (len(samples) - nper // 2) // max(nper // 2, 1))
    sigma_cap = max(1.5, 5.57 / np.sqrt(max(0.6 * n_avg, 1.0)))
    if analysis_band is not None:
        b_lo, b_hi = float(analysis_band[0]), float(analysis_band[1])
        sel = ((f >= b_lo) & (f <= b_hi)) if b_lo <= b_hi else \
              ((f >= b_lo) | (f <= b_hi))
        if sel.sum() >= 64:
            f, p = f[sel], p[sel]
        else:
            log.warning("analysis_band %s keeps < 64 bins -- ignoring it",
                        analysis_band)
    nf = estimate_noise_floor(p, k_sigma=k_sigma, sigma_cap_db=sigma_cap)
    mask = p > nf.threshold_db
    if close_bins > 0:
        # Morphological closing (dilate then erode): merges regions separated
        # by <= 2*close_bins without growing the outer edges of the mask.
        mask = ndimage.binary_closing(mask, structure=np.ones(2 * close_bins + 1))

    detections: List[Detection] = []
    n = p.size
    for s, e in find_signal_regions(mask):
        if (e - s) < min_region_bins:
            continue  # lone noise spike, not an emission
        coarse, refined = estimate_carrier_frequency(f, p, (s, e))
        idx = np.arange(s, e) % n
        k = int(idx[int(np.argmax(p[idx]))])
        snr = float(p[k] - nf.median_db)
        if snr > x_db:
            f_lo, f_hi, bw = estimate_bandwidth_xdb(f, p, k, x_db=x_db)
            method = f"{x_db:g}dB-down"
        else:
            # SNR below the shoulder level: the x-dB walk would run to the
            # horizon, so fall back to the threshold-crossing edges (which is
            # what an energy detector reports anyway). Handle wrapped regions
            # by un-rolling the frequency axis.
            f_lo = float(f[s % n])
            f_hi = float(f[(e - 1) % n])
            if f_hi < f_lo:
                f_hi += sample_rate
            bw = f_hi - f_lo
            method = "threshold-region"
        detections.append(
            Detection(
                carrier_hz=refined,
                center_hz=0.5 * (f_lo + f_hi),
                bandwidth_hz=bw,
                f_low_hz=f_lo,
                f_high_hz=f_hi,
                peak_power_db=float(p[k]),
                snr_db=snr,
                bandwidth_method=method,
                peak_bin_hz=coarse,
            )
        )

    if not detections:
        log.info("no emissions above threshold (%.1f dB/Hz)", nf.threshold_db)
    return SpectralReport(
        sample_rate_hz=float(sample_rate),
        nfft=n,
        noise=nf,
        detections=tuple(detections),
        freqs_hz=f,
        psd_db=p,
    )


# --------------------------------------------------------------------------- #
# 7) Burst channelization (Phase-2 bridge)
# --------------------------------------------------------------------------- #
def channelize_burst(
    samples: np.ndarray,
    sample_rate: float,
    center_hz: float,
    bandwidth_hz: float,
    *,
    transition_frac: float = 0.25,
    out_rate_factor: float = 6.0,
) -> Tuple[np.ndarray, float]:
    """Extract one detected emission as clean complex baseband at DC.

    Phase-2 bridge: turns a :class:`Detection` (center + bandwidth from the
    Phase-1 spectral report) into the input expected by
    ``symbol_rate.estimate_symbol_rate`` and ``amc.analyze_burst``.

    Steps:
      1. complex mixing: z_bb[n] = x[n] * exp(-j 2pi fc n / fs) -- moves the
         detection center to 0 Hz (classic digital down-conversion);
      2. Kaiser-windowed linear-phase FIR low-pass with passband +-B/2 and
         transition zone B/2*(1+tf), applied with ``filtfilt`` (zero-phase,
         so burst edges and envelope statistics are not group-delay biased);
      3. integer decimation to fs_out ~= out_rate_factor * B (enough
         oversampling for matched filtering; keeps Phase-2 FFTs cheap).

    Returns (z_bb complex64, fs_out float).
    """
    z = np.asarray(samples, dtype=np.complex128).ravel()
    fs = float(sample_rate)
    bw = float(abs(bandwidth_hz))
    if not 0 < bw < fs:
        raise ValueError(f"bandwidth {bw} Hz invalid for fs {fs} Hz")
    n = z.size
    if n < 64:
        raise ValueError(f"capture too short to channelize ({n} samples)")

    t = np.arange(n) / fs
    z = z * np.exp(-2j * np.pi * float(center_hz) * t)

    tf = float(transition_frac)
    # The Phase-1 bandwidth is a -10-dB-down measure -- NARROWER than the true
    # occupied spectrum. Truncating the skirts there injects ISI into every
    # downstream higher-order statistic, so give the passband 30% slack (the
    # extra noise admitted is harmless: AMC de-biases with the burst SNR).
    bw_eff = bw * 1.3
    # NOTE: cutoff/transition are passed in Hz via firwin(fs=...) -- firwin's
    # normalized axis is 0..1 = 0..NYQUIST, a classic units trap (a /fs-
    # normalized value silently halves the passband).
    cut_hz = 0.5 * bw_eff * (1.0 + 0.5 * tf)     # FIR -6 dB point (midpoint)
    trans_hz = 0.5 * bw_eff * tf                 # transition width in Hz
    ntaps = int(np.clip(int(np.ceil(4.0 * fs / max(trans_hz, 1e-3))) | 1,
                        31, 255))
    taps = sp_signal.firwin(ntaps, cutoff=cut_hz, fs=fs,
                            window=("kaiser", 7.0))
    z = sp_signal.filtfilt(taps, [1.0], z)

    dec = max(1, int(fs // max(out_rate_factor * bw, 1.0)))
    return z[::dec].astype(np.complex64), fs / dec

def estimate_burst_snr(z_bb: np.ndarray, fs: float, bw_hz: float) -> float:
    """In-band SNR (dB) of a channelized burst -- the number the AMC cumulant
    de-bias actually needs.

    Phase-1's Detection.snr_db is a *spectral-peak* metric (peak PSD minus
    floor); for a wideband emission it systematically UNDERSTATES the true
    SNR by ~10*log10(fs/B). Here we integrate the PSD: signal power = sum of
    in-band PSD exceeding the robust noise floor, noise power = floor times
    the in-band width. That ratio is the symbol-band SNR the cumulant bias
    formulas assume (white noise through the receiver BW).
    """
    f, p_db = welch_psd(z_bb, fs)
    p_lin = 10.0 ** (p_db / 10.0)
    bw = float(abs(bw_hz))
    # Noise floor = median PSD in the inter-band annulus 0.55B..0.70B: above
    # the emission's occupied band but still INSIDE the channelizer passband
    # (which spans ~0.72B at -6 dB), so those bins are pure noise -- no
    # signal leakage, no filter skirt (the skirt only takes over beyond the
    # passband edge, where levels plunge 60+ dB and would poison any
    # percentile taken over a wider window).
    ann = (np.abs(f) >= 0.55 * bw) & (np.abs(f) <= 0.70 * bw)
    if ann.sum() >= 8:
        floor = float(np.median(p_lin[ann]))
    else:
        floor = float(np.percentile(p_lin, 20))

    df = float(f[1] - f[0])
    inband = np.abs(f) <= 0.55 * bw
    sig = float(np.sum(np.maximum(p_lin[inband] - floor, 0.0))) * df
    noise = floor * (0.55 * bw) * 2.0
    if sig <= 0 or noise <= 0:
        return 0.0
    return 10.0 * float(np.log10(sig / noise))
