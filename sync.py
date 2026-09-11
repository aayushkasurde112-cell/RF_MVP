"""
sync.py -- Synchronization chain (Phase 3, NTRO PS-26147)
=========================================================

Turns a channelized burst (``spectral.channelize_burst``) + Phase-2 estimates
(baud, modulation family/subtype) into a clean **1-sample-per-symbol**
decision-ready symbol array.  Two branches, routed by the AMC result:

Linear branch (PSK / QAM)
-------------------------
1. **RRC matched filter** at the native channelizer rate (roll-off carried
   over from Phase 2), then linear resampling to exactly **2 samples/symbol**
   (the Gardner operating point).
2. **Coarse CFO**: M-th-power method.  x^M removes the data modulation
   (M = 2 BPSK, 4 QPSK/16QAM/64QAM, 8 8PSK), E[x^M] != 0 leaves a residual-
   carrier tone at M*f_err; FFT peak (prominence vs a median-filtered local
   baseline, parabolic-refined) -> f_err = f_peak / M -> mixdown.  8PSK is
   corrected with M = 8 like any other (E[x^8] = 1 for constant-envelope).
3. **Fine timing**: Gardner TED at 2 samples/symbol,
       e(k) = Re{ mid*(k) conj-independent ... }
   in complex form  e(k) = Re{ conj(y(k-T/2)) * (y(kT) - y(kT-T)) }  driving
   a 2nd-order loop (proportional + accumulator) whose fractional interval
   steps a **4-tap Farrow interpolator** (Catmull-Rom cubic kernel in
   polynomial/Farrow form).  Output strobes are the 1 sps symbols.
4. **Carrier phase**: decision-directed Costas loop (2nd-order DPLL): slice
   the phase-rotated symbol to the nearest constellation point of the
   classified subtype and rotate out the remaining static/CW phase error.
   Lock is to the constellation's symmetry class (e.g. 90 deg for QPSK) --
   irrelevant for detection/EVM; differential coding resolution is Phase-4.

FSK branch (non-coherent)
-------------------------
**Squared delay-and-multiply discriminator**:
   w(t) = z^2(t) * conj(z^2(t - tau))  =>  arg w = 4*pi*f_inst*tau,
which is *data-phase independent* (the symbol phase cancels in the squaring),
so f_inst(t) = arg(w) * fs / (4 pi tau) demodulates CPFSK without any carrier
recovery -- the Costas loop is bypassed entirely.  Symbol timing is recovered
by an offline fractional-offset search maximizing the between-class variance
(Otsu-style R^2) of the tone-sliced samples; symbols are the nearest-tone
indices mapped onto a unit-power complex ring.

Every stage is array-in/array-out with a diagnostics dict, mirroring the
Phase-1/2 style so the FastAPI layer can expose per-stage health later.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal

from amc import rrc_taps
from symbol_rate import instantaneous_frequency_modes

log = logging.getLogger(__name__)

__all__ = [
    "SyncResult",
    "synchronize_linear",
    "synchronize_fsk",
    "coarse_cfo_mpower",
    "farrow_interp",
    "resample_to_sps",
    "GardnerTimingRecovery",
    "CostasLoop",
]

# unit-power ideal constellations (must match amc._ANCHORS families)
def _constellation(subtype: Optional[str]) -> Tuple[np.ndarray, str]:
    """(unit-power ideal points, m_order for the M-th power CFO method)."""
    if subtype == "BPSK":
        pts = np.array([-1.0, 1.0], dtype=np.complex128)
        return pts, 2
    if subtype == "8PSK":
        pts = np.exp(2j * np.pi * np.arange(8) / 8 + 1j * np.pi / 8)
        return pts, 8
    if subtype in ("16QAM", "64QAM"):
        n = 4 if subtype == "16QAM" else 8
        a = 2.0 * np.arange(n) - (n - 1)          # odd levels -(n-1)..(n-1) step 2
        g = (a[:, None] + 1j * a[None, :]).ravel()
        return g / np.sqrt(np.mean(np.abs(g) ** 2)), 4
    # QPSK / unknown linear -> QPSK slicer, 4th power
    pts = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2.0)
    return pts, 4


# --------------------------------------------------------------------------- #
# Farrow interpolator (4-tap Catmull-Rom cubic, polynomial/Farrow structure)
# --------------------------------------------------------------------------- #
def farrow_interp(x: np.ndarray, pos: float) -> complex:
    """Interpolate x at the *fractional sample index* `pos`.

    Cubic convolution (Catmull-Rom) kernel in Farrow form: the output is a
    fixed 4-tap FIR whose coefficients are polynomials in the fractional
    interval mu -- exactly the structure a hardware NCO would drive.  Base
    points x[n-1..n+2] with n = floor(pos):

        y(mu) = 1/2 [ 2 x0 + (-x-1 + x1) mu
                      + (2x-1 - 5x0 + 2x1 - x2) mu^2
                      + (-x-1 + 3x0 - 3x1 + x2) mu^3 ]

    (Kernel reproduces x0 at mu=0 and x1 at mu=1 -- the mu^2 term
    carries 4*x1, NOT 2*x1, otherwise the segment interpolates 0 at mu=1.)
    """
    n = int(np.floor(pos))
    mu = pos - n
    if n < 1 or n + 2 >= x.size:
        raise IndexError(f"farrow_interp: position {pos} outside [{1}, {x.size - 3}]")
    xm1, x0, x1, x2 = x[n - 1], x[n], x[n + 1], x[n + 2]
    return 0.5 * (
        2.0 * x0
        + (x1 - xm1) * mu
        + (2.0 * xm1 - 5.0 * x0 + 4.0 * x1 - x2) * mu * mu
        + (3.0 * x0 - xm1 - 3.0 * x1 + x2) * mu * mu * mu
    )


def resample_to_sps(z: np.ndarray, fs: float, baud_hz: float,
                    target_sps: float = 2.0) -> Tuple[np.ndarray, float]:
    """Band-limited-interpolation resample to exactly `target_sps` samples/
    symbol.  Linear interpolation between native samples is adequate AFTER
    matched filtering (the RRC output is heavily oversampled at the
    channelizer rate, so the residual spectrum near fs/2 is noise-only)."""
    step = float(fs) / (float(target_sps) * float(baud_hz))  # native samples per output sample
    n_out = int((z.size - 1) / step) - 2
    pos = np.arange(n_out) * step
    zi = np.interp(pos, np.arange(z.size), z.real) + \
        1j * np.interp(pos, np.arange(z.size), z.imag)
    return zi.astype(np.complex128), float(baud_hz) / target_sps


# --------------------------------------------------------------------------- #
# 1) Coarse CFO: M-th power method
# --------------------------------------------------------------------------- #
def coarse_cfo_mpower(z: np.ndarray, fs: float, m_order: int,
                      search_hz: float = 5_000.0,
                      min_prominence: float = 4.0) -> Tuple[np.ndarray, float, float]:
    """Remove a coarse carrier-frequency offset via the M-th power tone.

    z^M strips the data phase for M-PSK/square-QAM modulations; its mean is
    E[x^M] e^{j 2 pi M f_err t}, i.e. a discrete tone at M*f_err on top of a
    (M-th-power) noise/data continuum.  The tone is found as the max
    prominence bin of FFT(z^M) against a median-filtered local baseline
    (a global floor would sit under the continuum hill at DC), refined
    parabolically, and the input is de-rotated by f_peak/M.

    Returns (z_corrected, cfo_hz, prominence_ratio).
    """
    zz = np.asarray(z, dtype=np.complex128)
    p = float(np.mean(np.abs(zz) ** 2))
    if p <= 0:
        return zz, 0.0, 0.0
    u = (zz / np.sqrt(p)) ** int(m_order)
    spec = np.abs(np.fft.fft(u))
    freqs = np.fft.fftfreq(u.size, d=1.0 / fs)

    guard = max(8, int(round(u.size * 30.0 / fs)))     # DC-leakage exclusion
    band = (np.abs(freqs) <= float(m_order) * search_hz) & (np.abs(freqs) > guard)
    if band.sum() < 16:
        return zz, 0.0, 0.0
    kernel = max(31, (spec.size // 512) | 1)
    baseline = sp_signal.medfilt(spec, kernel_size=kernel)
    ratio = np.zeros_like(spec)
    sel = band & (baseline > 0)
    ratio[sel] = spec[sel] / baseline[sel]
    k = int(np.argmax(ratio))
    prom = float(ratio[k])
    if prom < min_prominence:
        log.debug("coarse CFO: no tone above prominence %.1f (best %.2f)",
                  min_prominence, prom)
        return zz, 0.0, prom
    k_prev = (k - 1) % spec.size
    k_next = (k + 1) % spec.size
    a_, b_, c_ = np.log(spec[[k_prev, k, k_next]])
    den = a_ - 2 * b_ + c_
    delta = float(np.clip(0.5 * (a_ - c_) / den, -0.5, 0.5)) if abs(den) > 1e-12 else 0.0
    f_tone = float(freqs[k] + delta * (freqs[1] - freqs[0]))
    cfo = f_tone / float(m_order)
    n = np.arange(zz.size)
    zz = zz * np.exp(-2j * np.pi * cfo * n / fs)
    return zz, cfo, prom


def envelope_timing_phase(zm: np.ndarray, fs: float, baud_hz: float,
                          min_prominence: float = 3.0) -> Tuple[float, float]:
    """Absolute symbol-timing phase from the envelope-spectrum line (native rate).

    |z(t)|^2 of a shaped linear signal oscillates at f_baud with maxima at the
    symbol instants:  |z|^2 ~= DC + A cos(2 pi f_baud (t - t_max)).  The FFT
    bin at f_baud therefore carries the timing in its PHASE:
    t_max = -angle(X) / (2 pi f_baud).  Measured at the NATIVE rate because at
    exactly 2 samples/symbol f_baud aliases onto fs/2 (Nyquist) and the phase
    readout degenerates.  Returns (strobe phase in [0, 2) samples at 2 sps,
    tone prominence vs its local baseline); prominence below threshold means
    "no envelope timing info" (e.g. zero roll-off) and callers fall back.
    """
    w = np.abs(np.asarray(zm, dtype=np.complex128)) ** 2
    w = w - w.mean()
    X = np.fft.fft(w)
    freqs = np.fft.fftfreq(w.size, d=1.0 / fs)
    band = (np.abs(freqs) >= 0.8 * baud_hz) & (np.abs(freqs) <= 1.2 * baud_hz)
    if band.sum() < 4:
        return 0.0, 0.0
    mag = np.abs(X)
    kernel = max(31, (mag.size // 512) | 1)
    baseline = sp_signal.medfilt(mag, kernel_size=kernel)
    ratio = np.where(band & (baseline > 0), mag / np.maximum(baseline, 1e-30), 0.0)
    k = int(np.argmax(ratio))
    prom = float(ratio[k])
    if prom < min_prominence:
        return 0.0, prom
    # sign convention: X(f) = A/2 e^{+j theta} with cos(2 pi f (t - t_max))
    # <=> theta = -2 pi f t_max  ->  t_max = -angle(X) / (2 pi f)
    t_max = -float(np.angle(X[k])) / (2.0 * np.pi * baud_hz)
    phase_2sps = (t_max * 2.0 * baud_hz) % 2.0
    return float(phase_2sps), prom


def c42_timing_scan(z2: np.ndarray, power: float, n_sym: int = 64,
                    steps: int = 16) -> float:
    """Fallback timing scan for bursts WITHOUT an envelope tone
    (zero roll-off): strobe the first `n_sym` symbols at `steps` fractional
    phases and keep the phase maximizing |C42| (max-cumulant == min-ISI, the
    same criterion as amc.extract_symbols).  Short window on purpose: long
    windows smear the metric under carrier/timing drift.
    """
    best, best_c = 0.0, -1.0
    for ph in np.linspace(0.0, 2.0, steps, endpoint=False):
        try:
            st = np.array([farrow_interp(z2, 2.0 + ph + 2.0 * k)
                           for k in range(n_sym)])
        except IndexError:
            continue
        c = compute_c42(st / np.sqrt(np.mean(np.abs(st) ** 2)))
        if c > best_c:
            best_c, best = c, float(ph)
    log.debug("c42 timing scan -> phase %.3f (|C42| %.2f)", best, best_c)
    return best


def compute_c42(unit_power_symbols: np.ndarray) -> float:
    """|C42| helper for the timing scan (normalized absolute kurtosis)."""
    x = np.asarray(unit_power_symbols, dtype=np.complex128)
    m20 = np.mean(x ** 2)
    m42 = np.mean(np.abs(x) ** 4)
    return float(abs(m42 - abs(m20) ** 2 - 2.0))


# --------------------------------------------------------------------------- #
# 2) Fine timing: Gardner TED + Farrow NCO loop (2 samples/symbol)
# --------------------------------------------------------------------------- #
def measure_ted_gain(z2: np.ndarray, power: float, phase: float,
                     delta: float = 0.1, n_sym: int = 1200) -> float:
    """Empirical Gardner detector gain Kd = d mean(e) / d(strobe offset).

    Finite difference of the mean TED error between fixed-strobe runs at
    `phase +- delta`.  Feed this into :meth:`GardnerTimingRecovery.run` so the
    designed loop bandwidth is actually realized on the measured waveform.
    """
    def mean_e(off: float) -> float:
        prev = None
        es = []
        for k in range(64, 64 + n_sym):
            try:
                st = farrow_interp(z2, 2.0 + off + 2.0 * k)
                mid = farrow_interp(z2, 1.0 + off + 2.0 * k)
            except IndexError:
                break
            e = 0.0 if prev is None else float(
                np.real(np.conj(mid) * (st - prev)) / max(power, 1e-12))
            prev = st
            es.append(e)
        return float(np.mean(es)) if es else 0.0

    return (mean_e(phase + delta) - mean_e(phase - delta)) / (2.0 * delta)


class GardnerTimingRecovery:
    """Second-order Gardner timing loop operating at 2 samples/symbol.

    TED (complex Gardner): with y(kT) the decision-phase strobe and
    y(kT - T/2) the midpoint sample,
        e(k) = Re{ conj(mid_k) * (strobe_k - strobe_{k-1}) } / P
    which is zero when the strobes sit at the optimum (max-eye) instants and
    sign-correct on either side; /P (signal power) normalizes the detector
    gain across burst amplitudes.

    Loop (per symbol, 2nd order / PI controller):
        acc  += Ki * e            # frequency (drift) integrator
        pos  += sps + acc + Kp * e
    with `pos` the fractional sample index of the next strobe, driving
    :func:`farrow_interp`.  Gains follow the standard DPLL design with
    damping zeta = 0.707 and normalized natural frequency wn (= loop
    bandwidth in rad/symbol):
        d = 1 + 2 z wn + wn^2 ;  Kp = 4 z wn / d ;  Ki = 4 wn^2 / d
    """

    def __init__(self, sps: float = 2.0, loop_bw: float = 0.02,
                 damping: float = 0.707, max_drift: float = 0.05,
                 e_clip: float = 0.35):
        self.sps = float(sps)
        wn = float(loop_bw)
        z = float(damping)
        d = 1.0 + 2.0 * z * wn + wn * wn
        self.kp = 4.0 * z * wn / d
        self.ki = 4.0 * wn * wn / d
        self.max_drift = float(max_drift)
        self.e_clip = float(e_clip)

    def run(self, x: np.ndarray, power: float,
            init_phase: float = 0.0, init_drift: float = 0.0,
            drift_clamp: Optional[float] = None,
            ted_gain: float = 1.0) -> Tuple[np.ndarray, Dict]:
        """Recover strobes from a 2 sps signal; returns (symbols, diag).

        `init_phase` (samples, mod 2) seeds the strobe position -- pass
        :func:`envelope_timing_phase`'s output so the loop starts inside the
        basin of the TRUE Gardner null.  Without it the loop can settle on
        the T/2 quasi-stable null (zero TED output, maximally closed eye),
        which no amount of loop bandwidth fixes.
        """
        x = np.asarray(x, dtype=np.complex128)
        pos = 2.0 + float(init_phase) % 2.0     # skip filter ramp-in, seed phase
        acc = float(init_drift)                 # acquired drift (samples/symbol)
        drift_clamp = self.max_drift if drift_clamp is None else float(drift_clamp)
        # Normalize the loop by the MEASURED detector gain: the designed
        # bandwidth only holds if Kp/Ki are divided by the actual S-curve
        # slope (measured TED gain). Assuming Kd = 1 when the true gain is
        # ~0.04 leaves the loop with ~4% of the intended restoring force --
        # it then random-walks instead of tracking (measured on real bursts).
        kd = float(ted_gain) if abs(float(ted_gain)) > 1e-3 else 1.0
        kp = self.kp / kd
        ki = self.ki / kd
        e_prev_strobe = None
        out: list = []
        traj: list = []
        for _ in range(int((x.size - 4) // self.sps) - 1):
            try:
                strobe = farrow_interp(x, pos)
                mid = farrow_interp(x, pos - 0.5 * self.sps)
            except IndexError:
                break
            if e_prev_strobe is not None:
                # raw TED error -> DESIGN units (divide by measured gain) ->
                # clip in design units (a raw-unit clip would saturate the
                # normalized error constantly and turn the loop nonlinear).
                e = float(np.real(np.conj(mid) * (strobe - e_prev_strobe)))
                e = e / max(power, 1e-12) / kd
                e = float(np.clip(e, -self.e_clip, self.e_clip))  # hangup guard
            else:
                e = 0.0
            e_prev_strobe = strobe
            acc += self.ki * e
            acc = float(np.clip(acc, init_drift - drift_clamp,
                                init_drift + drift_clamp))
            pos += self.sps + acc + self.kp * e
            out.append(strobe)
            traj.append(pos % self.sps)
        sym = np.asarray(out, dtype=np.complex128)
        drift_ppm = 1e6 * acc / self.sps
        # converged loop state, for open-loop replay (see synchronize_linear):
        # the closed loop SOLVES the timing; replaying its solution without
        # feedback noise gives clean deterministic strobes.
        self.pos_final = float(pos)
        self.acc_final = float(acc)
        self.n_strobes = int(sym.size)
        return sym, {
            "timing_drift_ppm": round(drift_ppm, 2),
            "timing_phase_final": round(float(traj[-1]), 4) if traj else None,
            "n_timing_symbols": int(sym.size),
        }


# --------------------------------------------------------------------------- #
# 3) Carrier phase: decision-directed Costas loop (1 sample/symbol)
# --------------------------------------------------------------------------- #
class CostasLoop:
    """2nd-order decision-directed carrier-phase loop.

    Phase detector (generic, works for PSK *and* QAM): slice the rotated
    symbol to the nearest ideal constellation point d, then
        e(k) = angle( z_k * conj(d_k) )
    which equals the residual phase error whenever the slicer is correct --
    the decision-directed generalization of the classic Costas product PD.
    Loop structure identical to the Gardner PI above (theta in radians,
    `acc` in rad/symbol).  Lock converges to the constellation's rotational
    symmetry class (e.g. +90 deg for QPSK, 180 deg for BPSK): harmless for
    detection/EVM; differential decoding resolves it in Phase 4.
    """

    def __init__(self, ideal_points: np.ndarray, loop_bw: float = 0.05,
                 damping: float = 0.707):
        self.pts = np.asarray(ideal_points, dtype=np.complex128)
        wn = float(loop_bw)
        z = float(damping)
        d = 1.0 + 2.0 * z * wn + wn * wn
        self.kp = 4.0 * z * wn / d
        self.ki = 4.0 * wn * wn / d

    def _slice(self, z: np.ndarray) -> np.ndarray:
        d = np.abs(self.pts[None, :] - z[:, None])   # (N, M)
        return self.pts[np.argmin(d, axis=1)]

    def run(self, sym: np.ndarray) -> Tuple[np.ndarray, Dict]:
        sym = np.asarray(sym, dtype=np.complex128)
        theta = 0.0
        acc = 0.0
        out = np.empty_like(sym)
        for i, s in enumerate(sym):
            z = s * np.exp(-1j * theta)
            d = self.pts[int(np.argmin(np.abs(self.pts - z)))]
            e = float(np.angle(z * np.conj(d)))
            e = float(np.clip(e, -np.pi / 2, np.pi / 2))  # hangup protection
            acc += self.ki * e
            acc = float(np.clip(acc, -0.5, 0.5))
            theta += self.kp * e + acc
            theta = float(np.angle(np.exp(1j * theta)))   # wrap
            out[i] = z
        resid = float(np.angle(np.mean(out * np.conj(self._slice(out)))))
        return out, {"residual_phase_deg": round(float(np.degrees(resid)), 3)}


# --------------------------------------------------------------------------- #
# Linear-branch orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class SyncResult:
    """Synchronization output: 1 sps decision-ready symbols + diagnostics."""

    symbols: np.ndarray                  # complex, unit power, 1 sample/symbol
    family: str                          # "PSK" | "QAM" | "FSK"
    subtype: Optional[str]               # "QPSK", ... (None for FSK)
    evm_pct: Optional[float]             # None for FSK (freq mod)
    n_symbols: int
    diag: Dict = field(default_factory=dict)


def synchronize_linear(z_bb: np.ndarray, fs: float, baud_hz: float,
                       alpha: float, subtype: Optional[str],
                       acquire_skip: int = 32) -> SyncResult:
    """Full linear-family chain: matched filter -> coarse CFO -> 2 sps ->
    Gardner timing -> Costas phase -> 1 sps unit-power symbols."""
    z = np.asarray(z_bb, dtype=np.complex128).ravel()
    pts, m_order = _constellation(subtype)

    # (0) RRC matched filter at the native rate -- eye opening before any loop
    sps_native = float(fs) / float(baud_hz)
    taps = rrc_taps(alpha, 8, sps_native)
    zm = sp_signal.filtfilt(taps, [1.0], z)

    # (1) coarse CFO (M-th power tone)
    zm, cfo_hz, cfo_prom = coarse_cfo_mpower(zm, fs, m_order)

    # (2) resample to the Gardner operating point: 2 samples/symbol
    z2, fs2 = resample_to_sps(zm, fs, baud_hz, target_sps=2.0)

    # (3) fine timing: envelope-phase init -> Gardner @ 2 sps + Farrow.
    # The Gardner TED has quasi-stable nulls at T/2 offsets; seeding from the
    # envelope-spectrum phase (or a |C42| scan when the envelope is flat,
    # i.e. zero roll-off) starts the loop in the true null's basin.
    pwr = float(np.mean(np.abs(z2) ** 2))
    phase0, env_prom = envelope_timing_phase(zm, fs, baud_hz)
    if env_prom < 3.0:
        phase0 = c42_timing_scan(z2, pwr)

    # --- open-loop 2D acquisition (phase x drift) ---------------------------
    # The Gardner TED has quasi-stable nulls (T/2) AND, at burst SNR, the
    # closed loop alone can self-excite (TED noise x loop gain). Standard
    # offline remedy: acquire (phase, drift) with a small slicer-EVM grid --
    # which also covers BOTH Gardner nulls -- then let a LOW-AUTHORITY
    # Gardner loop (small bw, clipped error, drift clamp) track residuals.
    def _evm_fixed(ph: float, dr: float, n0: int, n1: int) -> float:
        k = np.arange(n0, min(n1, int((z2.size - 8) / 2.0)))
        # base 2.0 == the Gardner loop's seed frame (pos = 2 + init_phase):
        # a mismatch here is an exact T/2 offset, i.e. the quasi-stable null.
        st = np.array([farrow_interp(z2, 2.0 + ph + 2.0 * (k + dr * k / 2.0))
                       for k in k])
        s = st / np.sqrt(np.mean(np.abs(st) ** 2))
        ideal = pts[np.argmin(np.abs(pts[None, :] - s[:, None]), axis=1)]
        return 100.0 * float(np.sqrt(np.mean(np.abs(s - ideal) ** 2)))

    n_cap = int((z2.size - 8) / 2.0)
    n0 = min(200, max(0, n_cap // 10))
    n1 = min(n_cap, n0 + 1500)
    best = (1e9, phase0, 0.0)
    for ph in np.linspace(0.0, 2.0, 16, endpoint=False):
        for dr in np.linspace(-2e-3, 2e-3, 9):
            ev = _evm_fixed(ph, dr, n0, n1)
            if ev < best[0]:
                best = (ev, ph, dr)
    ev_grid, ph_acq, dr_acq = best
    # stage 2: fine drift refine (+-250 ppm around the coarse winner) so the
    # residual clock offset is <= ~30 ppm (<= ~0.5 symbol slip per 16k burst)
    for dr in np.linspace(dr_acq - 250e-6, dr_acq + 250e-6, 9):
        ev = _evm_fixed(ph_acq, dr, n0, n1)
        if ev < best[0]:
            best = (ev, ph_acq, dr)
    ev_grid, ph_acq, dr_acq = best

    # --- Gardner closed-loop tracking (low authority) ------------------------
    kd = measure_ted_gain(z2, pwr, ph_acq)
    # Tracking loop with authority matched to the residuals the grid leaves
    # (sub-% clock offsets): at burst SNR the raw TED noise exceeds the
    # S-curve slope, so a wide loop random-walks into the T/2 null -- narrow
    # bandwidth + small drift clamp make the Gardner loop a genuine
    # fine-tracker on top of the open-loop acquisition.
    tracker = GardnerTimingRecovery(sps=2.0, loop_bw=5e-4, e_clip=0.5)
    symbols, tdiag = tracker.run(z2, pwr, init_phase=ph_acq,
                                 init_drift=dr_acq / 2.0, drift_clamp=2e-4,
                                 ted_gain=kd)
    tdiag["ted_gain_measured"] = round(kd, 4)
    if symbols.size <= acquire_skip + 16:
        raise ValueError(
            f"timing recovery produced too few symbols ({symbols.size})")

    # --- deliver the better eye: closed-loop vs open-loop -------------------
    # The Gardner loop tracks real residual drift, but at burst SNR its
    # position can random-walk; the acquired open-loop constants cannot.
    # Selection is by measured EVM on the SAME post-acquisition span.
    k_hi = min(symbols.size + acquire_skip, int((z2.size - 8) / 2.0) - 2)
    k_ol = np.arange(acquire_skip, k_hi, dtype=float)
    sym_open = np.array([farrow_interp(
        z2, 2.0 + ph_acq + (2.0 + dr_acq) * t) for t in k_ol])
    ev_loop = 100.0 * float(np.sqrt(np.mean(
        np.abs(symbols[acquire_skip:] /
               np.sqrt(np.mean(np.abs(symbols[acquire_skip:]) ** 2))
               - pts[np.argmin(np.abs(pts[None, :] -
                   (symbols[acquire_skip:] /
                    np.sqrt(np.mean(np.abs(symbols[acquire_skip:]) ** 2)))[:, None]),
                   axis=1)]) ** 2)))
    s_open = sym_open / np.sqrt(np.mean(np.abs(sym_open) ** 2))
    id_open = pts[np.argmin(np.abs(pts[None, :] - s_open[:, None]), axis=1)]
    ev_open = 100.0 * float(np.sqrt(np.mean(np.abs(s_open - id_open) ** 2)))
    if ev_open < ev_loop:
        symbols = sym_open[: symbols.size]
        tdiag["timing_source"] = "open_loop"
        tdiag["evm_pre_costas"] = round(ev_open, 2)
    else:
        tdiag["timing_source"] = "gardner_track"
        tdiag["evm_pre_costas"] = round(ev_loop, 2)
    tdiag["env_timing_prominence"] = round(env_prom, 1)
    tdiag["timing_phase_init"] = round(phase0, 3)
    tdiag["acquired_phase"] = round(ph_acq, 3)
    tdiag["acquired_drift_ppm"] = round(dr_acq * 1e6, 1)
    tdiag["grid_evm_pct"] = round(ev_grid, 2)

    # (4) decision-directed carrier phase recovery (Costas)
    symbols, cdiag = CostasLoop(pts, loop_bw=0.05).run(symbols)
    symbols = symbols[acquire_skip:]
    symbols = symbols / np.sqrt(np.mean(np.abs(symbols) ** 2))
    ideal = pts[np.argmin(np.abs(pts[None, :] - symbols[:, None]), axis=1)]
    evm = 100.0 * float(np.sqrt(np.mean(np.abs(symbols - ideal) ** 2)))

    return SyncResult(
        symbols=symbols.astype(np.complex64),
        family="PSK" if subtype in ("BPSK", "QPSK", "8PSK") else "QAM",
        subtype=subtype,
        evm_pct=round(evm, 2),
        n_symbols=int(symbols.size),
        diag={"cfo_corrected_hz": round(cfo_hz, 2),
              "cfo_tone_prominence": round(cfo_prom, 1),
              "m_order": m_order, "alpha_used": round(alpha, 3),
              "sps_native": round(sps_native, 3), **tdiag, **cdiag},
    )


# --------------------------------------------------------------------------- #
# FSK branch (non-coherent): squared delay-and-multiply discriminator
# --------------------------------------------------------------------------- #
def fsk_discriminator(z: np.ndarray, fs: float, baud_hz: float,
                      delay: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """w(t) = z^2(t) conj(z^2(t - tau)) -> f_inst = arg(w) fs / (4 pi tau).

    The squaring makes the product's phase data-independent for CPFSK
    (symbol phase cancels), so this is a *non-coherent* FM demodulator --
    no Costas loop on this branch.  tau = T/2 maximizes the deviation gain
    while keeping the two multiplied samples inside one symbol for baud
    rates up to fs/4.  Returns (f_inst, delay_samples).
    """
    z = np.asarray(z, dtype=np.complex128)
    d = max(1, int(delay if delay is not None
                   else round(0.5 * fs / float(baud_hz))))
    w = (z[d:] * np.conj(z[:-d])) ** 2
    f_inst = np.angle(w) * fs / (4.0 * np.pi * d)
    return f_inst, d


def synchronize_fsk(z_bb: np.ndarray, fs: float, baud_hz: float,
                    n_tones: Optional[int] = None,
                    tone_spacing_hz: Optional[float] = None) -> SyncResult:
    """Non-coherent FSK chain: discriminator -> timing search -> tone slicing.

    Timing: fractional-offset search over one symbol period; at each offset
    the discriminator output is sampled at symbol centers and sliced to the
    nearest tone; the offset maximizing the between-class variance ratio
    R^2 = 1 - SS_within / SS_total (Otsu criterion) wins -- equivalent to
    maximum eye opening for a frequency-modulated signal.

    Symbols: tone index k -> exp(j 2 pi k / M) on the unit circle (unit
    power), so downstream consumers see a complex 1 sps symbol array like
    the linear branch.
    """
    z = np.asarray(z_bb, dtype=np.complex128).ravel()

    # Tone span FIRST (raw-phase derivative has no delay ambiguity): the
    # delay-and-multiply discriminator wraps at |f| > fs/(4d), so its delay
    # must be chosen from the measured tone span, not a fixed T/2.
    n_states, modes = instantaneous_frequency_modes(z, fs)
    m = int(n_tones or max(2, n_states))
    f_span_half = 0.5 * (modes[-1] - modes[0]) * (m / max(modes.size, 1)) \
        if modes.size >= 2 else 0.25 * baud_hz
    d = int(np.clip(fs / (4.0 * max(f_span_half * 1.2, 1.0)), 1,
                    max(1.0, fs / baud_hz)))
    f_inst, delay = fsk_discriminator(z, fs, baud_hz, delay=d)

    # tone grid: from AMC state count + measured instantaneous-freq modes.
    # For equi-spaced MFSK the spacing is best taken from the EXTREME modes
    # (span/(M-1)) -- a median of consecutive diffs is corrupted by spurious
    # histogram peaks and half-detected edge states.
    if tone_spacing_hz is None:
        if m >= 2 and modes.size >= 2:
            spacing = float((modes[-1] - modes[0]) / max(modes.size - 1, 1)) \
                if modes.size >= m else \
                float(np.median(np.diff(modes))) if modes.size >= 2 else \
                float(baud_hz) / 2.0
        else:
            spacing = float(baud_hz) / 2.0
    else:
        spacing = float(tone_spacing_hz)
    if modes.size >= 2:
        f_center = 0.5 * (modes[-1] + modes[0])
    else:
        f_center = 0.0
    tones = f_center + spacing * (np.arange(m) - (m - 1) / 2.0)

    # --- symbol-timing search (Otsu R^2 over fractional offsets) ------------
    best = (None, -1.0, None)
    n_sym = int((f_inst.size - 2) // (fs / baud_hz)) - 1
    for tau in np.linspace(0.0, fs / baud_hz, 64, endpoint=False):
        k = np.arange(n_sym)
        s = np.interp(tau + k * fs / baud_hz, np.arange(f_inst.size), f_inst)
        cls = np.argmin(np.abs(s[:, None] - tones[None, :]), axis=1)
        ss_tot = float(np.sum((s - s.mean()) ** 2))
        ss_in = 0.0
        for j in range(m):
            sel = s[cls == j]
            if sel.size:
                ss_in += float(np.sum((sel - sel.mean()) ** 2))
        r2 = 1.0 - ss_in / ss_tot if ss_tot > 0 else -1.0
        if r2 > best[1]:
            best = (tau, r2, cls)
    tau_best, r2, cls = best
    if cls is None:
        raise ValueError("FSK timing search failed (empty burst?)")
    # re-sample at the winning offset (values the slice decision used)
    k = np.arange(n_sym)
    s_sym = np.interp(tau_best + k * fs / baud_hz,
                      np.arange(f_inst.size), f_inst)

    # hard meta-symbol output: tone index -> unit-power complex ring point
    ring = np.exp(2j * np.pi * np.arange(m) / m)
    symbols = ring[cls].astype(np.complex64)
    hist = np.bincount(cls, minlength=m).tolist()
    freq_rms = float(np.sqrt(np.mean([
        np.mean((s_sym[cls == j] - tones[j]) ** 2)
        for j in range(m) if np.sum(cls == j) > 0])))
    return SyncResult(
        symbols=symbols,
        family="FSK",
        subtype=None,
        evm_pct=None,
        n_symbols=int(symbols.size),
        diag={"n_tones": m,
              "tone_spacing_hz": round(spacing, 1),
              "tone_freqs_hz": [round(float(t), 1) for t in tones],
              "tone_occupancy": hist,
              "fsk_timing_r2": round(float(r2), 3),
              "fsk_timing_offset_frac": round(float(tau_best) * baud_hz / fs, 3),
              "discriminator_delay_s": round(delay / fs, 6),
              "freq_slice_rms_hz": round(freq_rms, 1)},
    )
