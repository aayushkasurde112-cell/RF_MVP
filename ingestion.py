"""
ingestion.py -- Data ingestion & normalization (Phase 1, NTRO PS-26147)
=======================================================================

Python orchestration layer for the Automated RF Signal Parameter Extraction
MVP. Every public loader returns an :class:`IQData` container holding a
*complex64*, unit-norm baseband array. All downstream DSP (``spectral.py``)
is written against this contract, so a future streaming front-end (SDR
driver callback / ring buffer) can reuse the exact same math per chunk.

Supported inputs
----------------
``*.iq`` / ``*.raw``  Headerless interleaved IQ. Datatype auto-guessed from
                      file size (modulo sanity check + a content probe),
                      fallback ladder ``cf32_le -> cs16 -> cu8``; an explicit
                      ``datatype`` always wins (in production you know the
                      radio config).
``*.wav``             RIFF/WAVE via ``scipy.io.wavfile``. Mono -> analytic
                      signal via the FFT Hilbert transform. Stereo -> (L, R)
                      mapped to (I, Q).

All failures are raised as :class:`IngestionError` so a FastAPI layer can
map them to a single HTTP 4xx/5xx handler.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy import signal as sp_signal
from scipy.io import wavfile

log = logging.getLogger(__name__)

__all__ = [
    "IngestionError",
    "IQData",
    "load_iq",
    "load_wav",
    "guess_datatype",
    "guess_datatype_auto",
    "remove_dc_offset",
]

# Bytes-per-complex-sample table for raw formats.  Multi-byte formats are
# little-endian by SigMF/GNU-Radio convention; explicit `_be` tags supported.
#   name -> (numpy element dtype, bytes per complex sample)
_RAW_FORMATS: Dict[str, Tuple[str, int]] = {
    "cf32_le": ("<f4", 8),  # complex float32  (GNU Radio "complex", SDR# float)
    "cf32": ("<f4", 8),     # alias
    "cf32_be": (">f4", 8),
    "cs16_le": ("<i2", 4),  # complex int16    (RTL-SDR "cs16", HackRF scope)
    "cs16": ("<i2", 4),     # alias (SigMF default endianness is LE)
    "cs16_be": (">i2", 4),
    "cu8": ("u1", 2),       # complex uint8    (classic RTL-SDR / rtl_sdr)
    "ci8": ("i1", 2),       # complex int8     (HackRF int8, some puck boards)
}

#: PS-26147 fallback ladder -- try the widest (highest-fidelity) type first.
_LADDER: Tuple[str, ...] = ("cf32_le", "cs16", "cu8")

#: Sanity ceiling for "plausible float32 IQ": normalized captures live in
#: roughly +-1.0; anything >> 16 or non-finite means the bytes are *not* f32 IQ.
_CF32_PLAUSIBLE_CEIL = 16.0


class IngestionError(RuntimeError):
    """Raised when a capture file cannot be read, parsed or normalized."""


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #
@dataclass
class IQData:
    """Normalized complex-baseband capture plus provenance metadata.

    Attributes
    ----------
    samples : np.ndarray       complex64, |I|,|Q| ~ within [-1, 1], 1-D.
    sample_rate : float | None Hz. Raw `.iq` files carry no header, so the
                               caller supplies it (or sets it afterwards).
    source_path : Path | None  Original file, used by the SigMF writer.
    datatype : str             Ingest-format tag, e.g. ``cf32_le``,
                               ``wav_mono_hilbert``, ``wav_stereo_iq``.
    center_freq_hz : float | None  RF tune frequency, if known (goes into the
                               SigMF capture record as ``core:frequency``).
    notes : dict               Ingest provenance (datatype guess, DC removed,
                               WAV encoding, ...) mirrored into SigMF metadata.
    """

    samples: np.ndarray
    sample_rate: Optional[float] = None
    source_path: Optional[Path] = None
    datatype: str = "cf32_le"
    center_freq_hz: Optional[float] = None
    notes: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.samples = np.ascontiguousarray(self.samples, dtype=np.complex64).ravel()
        if self.samples.size == 0:
            raise IngestionError("ingested capture contains zero samples")
        if self.sample_rate is not None and self.sample_rate <= 0:
            raise IngestionError(f"sample_rate must be > 0, got {self.sample_rate}")
        if self.source_path is not None:
            self.source_path = Path(self.source_path)

    def __len__(self) -> int:
        return int(self.samples.size)


# --------------------------------------------------------------------------- #
# Datatype guessing (raw IQ)
# --------------------------------------------------------------------------- #
def guess_datatype(size_bytes: int) -> str:
    """Pure modulo-sanity datatype guess (PS-26147 fallback ladder).

    Walks the ladder ``cf32_le (8 B/cplx) -> cs16 (4) -> cu8 (2)`` and returns
    the first format whose complex-sample width divides the file size.

    .. note::
        This check is *weak by construction*: any file whose size is a
        multiple of 8 bytes is also a legal multiple of 4 and 2 bytes, so the
        ladder simply prefers the widest type. See :func:`guess_datatype_auto`
        for the content-probe refinement actually used by :func:`load_iq`.
    """
    if size_bytes <= 0:
        raise IngestionError("cannot guess the datatype of an empty file")
    for name in _LADDER:
        if size_bytes % _RAW_FORMATS[name][1] == 0:
            return name
    raise IngestionError(
        f"file size {size_bytes} B is not a multiple of any supported "
        f"complex-sample width (2/4/8 B)"
    )


def _cf32_plausible(head_f32: np.ndarray) -> bool:
    """Heuristic: do these bytes look like normalized float32 IQ?

    Integer IQ (cs16/cu8) reinterpreted as float32 produces a zoo of denormals,
    NaNs and absurd magnitudes -- a cheap and highly discriminating probe.
    """
    if head_f32.size < 16:
        return True  # too little evidence; give the ladder-top the benefit
    finite = np.isfinite(head_f32)
    if finite.mean() < 0.99:
        return False
    mag = np.abs(head_f32[finite])
    return bool(np.max(mag) < 1e6 and np.median(mag) < _CF32_PLAUSIBLE_CEIL)


def guess_datatype_auto(path: Union[str, Path], probe_bytes: int = 1 << 16) -> str:
    """Modulo sanity check + content probe.

    1. Keep only ladder formats whose byte width divides the file size.
    2. If the modulo winner is ``cf32_le`` (i.e. size % 8 == 0, where *every*
       format is byte-legal), decode the first `probe_bytes` as float32 IQ and
       run the plausibility probe; on failure, descend to ``cs16``.

    Caveat: a genuine ``cu8`` file whose size is a multiple of 8 bytes decodes
    to in-range (but garbage) cs16 samples -- cs16-vs-cu8 is *information
    theoretically* ambiguous from content alone. Pass ``datatype="cu8"``
    explicitly; operators know their radio configuration.
    """
    p = Path(path)
    size = p.stat().st_size
    ladder = [dt for dt in _LADDER if size % _RAW_FORMATS[dt][1] == 0]
    if not ladder:
        raise IngestionError(
            f"{p.name}: size {size} B matches no supported sample width"
        )
    top = ladder[0]
    if top != "cf32_le" or size < 8:
        return top  # unique modulo winner (size % 8 != 0)

    with open(p, "rb") as fh:
        head = fh.read(min(probe_bytes, size))
    head = head[: (len(head) // 8) * 8]  # whole complex samples
    floats = np.frombuffer(head, dtype="<f4")
    if _cf32_plausible(floats):
        return "cf32_le"
    log.info(
        "%s: cf32 probe failed (bytes are not float32 IQ) -- descending ladder to cs16",
        p.name,
    )
    return ladder[1] if len(ladder) > 1 else top


# --------------------------------------------------------------------------- #
# Raw IQ loader
# --------------------------------------------------------------------------- #
def _decode_raw(raw: np.ndarray, datatype: str) -> np.ndarray:
    """Map integer/float element data to unit-norm complex64 (I + jQ)."""
    kind = datatype.split("_", 1)[0]
    if kind == "cf32":
        # Float captures are assumed already volt-normalized; `astype` also
        # resolves big-endian byte order to native.
        x = raw.astype(np.float32)
    elif kind == "cs16":
        # int16 full scale 32768 -> [-1, 1).  Divide (not bit-shift) so the
        # mapping is exact for -32768 as well.
        x = raw.astype(np.float32) / 32768.0
    elif kind == "cu8":
        # Offset-binary uint8: centre on 127.5 so 0 and 255 map symmetrically
        # to -1/+1 (GNU Radio convention).
        x = (raw.astype(np.float32) - 127.5) / 127.5
    elif kind == "ci8":
        x = raw.astype(np.float32) / 128.0
    else:
        raise IngestionError(f"unsupported raw datatype '{datatype}'")

    if x.size % 2:  # dangling scalar => truncated/corrupt write
        log.warning("%s: odd element count, dropping trailing scalar", datatype)
        x = x[:-1]
    # De-interleave: even indices = I, odd = Q.
    return x[0::2] + 1j * x[1::2]


def _rescale_if_hot(z: np.ndarray) -> np.ndarray:
    """Guard rails for float captures whose peak exceeds the +-1 contract."""
    peak = float(np.max(np.abs(z))) if z.size else 0.0
    if peak > 1.0:
        log.warning("float capture peak %.2f > 1.0 -- peak-normalizing", peak)
        z = (z / peak).astype(np.complex64)
    return z


def load_iq(
    path: Union[str, Path],
    sample_rate: Optional[float] = None,
    *,
    datatype: Optional[str] = None,
    center_freq_hz: Optional[float] = None,
    dc_block: bool = True,
    dc_window: Optional[int] = None,
) -> IQData:
    """Ingest a headerless interleaved-IQ file into an :class:`IQData`.

    Parameters
    ----------
    path : raw capture path (``.iq``, ``.raw``, any extension really).
    sample_rate : Hz. Raw files carry no header -- supply it when known;
                  it can also be set on the returned container afterwards.
    datatype : explicit format (``cf32_le`` | ``cs16`` | ``cu8`` | ...).
               Overrides the auto-guess. **Recommended in production**, since
               the guess is only a sanity heuristic.
    dc_block : apply :func:`remove_dc_offset` after normalization.
    dc_window : moving-average window in samples (None -> whole-record mean).
    """
    p = Path(path)
    if not p.exists():
        raise IngestionError(f"IQ file not found: {p}")
    if not p.is_file():
        raise IngestionError(f"IQ path is not a regular file: {p}")
    size = p.stat().st_size
    if size == 0:
        raise IngestionError(f"IQ file is empty: {p}")
    if size > (1 << 30):
        log.warning(
            "%s: %.1f GiB capture will be loaded fully into RAM "
            "(MVP loader is offline, not chunked)",
            p.name,
            size / 2**30,
        )

    guessed = datatype is None
    dt = (datatype if datatype else guess_datatype_auto(p, size)).lower()
    if dt not in _RAW_FORMATS:
        raise IngestionError(
            f"unsupported raw datatype '{datatype}' "
            f"(choose from {sorted(set(_RAW_FORMATS))})"
        )
    elem_dt, bytes_per_cplx = _RAW_FORMATS[dt]
    n_cplx = size // bytes_per_cplx
    if size % bytes_per_cplx:
        log.warning(
            "%s: %d trailing bytes do not fit '%s' -- truncating",
            p.name,
            size % bytes_per_cplx,
            dt,
        )
    if guessed:
        log.info("%s: datatype auto-guessed as '%s'", p.name, dt)

    try:
        raw = np.fromfile(p, dtype=elem_dt, count=2 * n_cplx)
    except OSError as exc:
        raise IngestionError(f"failed to read {p}: {exc}") from exc

    z = _decode_raw(raw, dt)
    if dt.startswith("cf32"):
        z = _rescale_if_hot(z)
    z = z.astype(np.complex64, copy=False)

    notes: Dict = {
        "datatype_guessed": guessed,
        "ingest_bytes": int(size),
        "n_complex_samples": int(z.size),
    }
    if dc_block:
        z, dc_info = remove_dc_offset(z, window=dc_window)
        notes.update(dc_info)

    return IQData(
        samples=z,
        sample_rate=sample_rate,
        source_path=p,
        datatype=dt,
        center_freq_hz=center_freq_hz,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# WAV loader
# --------------------------------------------------------------------------- #
def _pcm_to_float(x: np.ndarray) -> np.ndarray:
    """Normalize common WAV PCM encodings to floats in [-1, 1]."""
    dt = x.dtype
    if np.issubdtype(dt, np.floating):
        return x.astype(np.float32)
    if dt == np.uint8:  # 8-bit PCM is offset-binary in WAV
        return (x.astype(np.float32) - 128.0) / 128.0
    if dt == np.int8:
        return x.astype(np.float32) / 128.0
    if dt == np.int16:
        return x.astype(np.float32) / 32768.0
    if dt == np.int32:
        return x.astype(np.float32) / 2147483648.0
    raise IngestionError(f"unsupported WAV sample encoding: {dt}")


def load_wav(
    path: Union[str, Path],
    *,
    center_freq_hz: Optional[float] = None,
    dc_block: bool = True,
    dc_window: Optional[int] = None,
) -> IQData:
    """Ingest a RIFF/WAVE file into an :class:`IQData`.

    * **mono** -> treated as a *real* recording: converted to its analytic
      signal ``z = x + j*H{x}`` via the FFT Hilbert transform, which cancels
      the negative-frequency image. This makes a mono WAV a legitimate
      complex-baseband source for spectral analysis.
    * **stereo** -> soundcard-style SDR dumps: L channel = I, R channel = Q.
    """
    p = Path(path)
    if not p.is_file():
        raise IngestionError(f"WAV file not found: {p}")
    try:
        fs, data = wavfile.read(str(p))
    except FileNotFoundError as exc:
        raise IngestionError(f"WAV file not found: {p}") from exc
    except (ValueError, OSError) as exc:
        raise IngestionError(f"failed to parse WAV {p}: {exc}") from exc
    if fs <= 0:
        raise IngestionError(f"{p.name}: invalid sample rate {fs} in WAV header")

    data = np.asarray(data)
    if data.ndim == 2 and data.shape[1] == 1:  # WAVE_FORMAT_EXTENSIBLE mono
        data = data[:, 0]

    notes: Dict = {"wav_dtype": data.dtype.name}
    if data.ndim == 1:
        x = _pcm_to_float(data)
        if x.size < 16:
            raise IngestionError(f"{p.name}: clip too short for Hilbert analysis")
        if x.size > (1 << 27):
            log.warning(
                "%s: %.1f Msamples -- FFT Hilbert will need ~%.1f GiB; "
                "swap in an FIR Hilbert transformer for streaming",
                p.name,
                x.size / 1e6,
                x.size * 16 / 2**30,
            )
        # Analytic signal via FFT round-trip: Z(f) = 2*X(f) for f>0, 0 for f<0.
        # O(N log N) time, 16 B/sample peak RAM -- fine offline; the streaming
        # build should use an odd-tap FIR Hilbert filter applied per chunk.
        z = sp_signal.hilbert(x).astype(np.complex64)
        notes["wav_channels"] = 1
        dt_tag = "wav_mono_hilbert"
    elif data.ndim == 2 and data.shape[1] >= 2:
        if data.shape[1] > 2:
            log.warning(
                "%s: %d channels -- using ch0/ch1 as I/Q", p.name, data.shape[1]
            )
        i = _pcm_to_float(np.ascontiguousarray(data[:, 0]))
        q = _pcm_to_float(np.ascontiguousarray(data[:, 1]))
        z = (i + 1j * q).astype(np.complex64)
        notes["wav_channels"] = int(data.shape[1])
        dt_tag = "wav_stereo_iq"
    else:
        raise IngestionError(f"{p.name}: unexpected WAV data shape {data.shape}")

    if dc_block:
        z, dc_info = remove_dc_offset(z, window=dc_window)
        notes.update(dc_info)

    return IQData(
        samples=z,
        sample_rate=float(fs),
        source_path=p,
        datatype=dt_tag,
        center_freq_hz=center_freq_hz,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# DC blocking (LO-leakage removal)
# --------------------------------------------------------------------------- #
def remove_dc_offset(
    samples: np.ndarray,
    window: Optional[int] = None,
    method: str = "moving_average",
) -> Tuple[np.ndarray, Dict]:
    """Remove the DC spur caused by local-oscillator (LO) leakage.

    Direct-conversion receivers leak their own LO into baseband, which shows
    up as a strong stationary spur at exactly 0 Hz -- i.e. a non-zero mean of
    I and Q. Left in place it (a) wastes ~6 dB of ADC/dynamic range and
    (b) fakes a detection at baseband.

    Methods
    -------
    ``moving_average`` (default, per PS-26147):
        Estimate the slowly-varying DC term with a centered boxcar moving
        average and subtract it::

            d[n] = (1/W) * sum_{k=0..W-1} x[n - (W-1)/2 + k],   y = x - d

        Implemented with a cumulative sum -> O(N) independent of W. The DC
        estimate at each point is the mean of W neighbours, so slow drifts
        (e.g. thermal LO wander) are tracked, not just the global offset.
        `window=None` degenerates to a single whole-record mean -- the
        maximum-SNR DC estimate for short, stationary captures.
        Edges use reflect-padding so the estimate stays unbiased.

    ``recursive`` (streaming-friendly alternative):
        One-pole DC canceller ``y[n] = x[n] - x[n-1] + a*y[n-1]`` with
        ``a = exp(-1/W)`` (W = time-constant in samples). O(1) state, i.e.
        exactly what the future streaming front-end runs per chunk; provided
        here so offline and online paths share one definition of "DC removed".

    Returns
    -------
    (z_clean, info) : complex64 array + provenance dict for SigMF notes.
    """
    z = np.asarray(samples)
    if z.size == 0:
        return z.astype(np.complex64), {"dc_method": "none"}

    if method == "recursive":
        w = float(window) if window else 1024.0
        a = math.exp(-1.0 / max(1.0, w))
        # DC is the exact zero of H(z) = (1 - z^-1) / (1 - a z^-1).
        y = sp_signal.lfilter([1.0, -1.0], [1.0, -a], z)
        info = {"dc_method": "recursive", "dc_alpha": round(a, 9)}
        return y.astype(np.complex64), info

    if method != "moving_average":
        raise ValueError(f"unknown DC-block method '{method}'")

    dc_i = float(np.mean(z.real))
    dc_q = float(np.mean(z.imag))
    if window is None or window >= z.size:
        # Whole-record mean: minimum-variance DC estimate for stationary data.
        y = z - (dc_i + 1j * dc_q)
        info = {"dc_method": "record_mean", "dc_removed_iq": (dc_i, dc_q)}
        return y.astype(np.complex64), info

    # Centered sliding mean via cumulative sum. Odd W keeps the estimate
    # aligned with no group delay; complex128 accumulation avoids error
    # growth over long captures.
    w = max(3, int(window) | 1)
    pad = (w - 1) // 2
    xp = np.pad(z, (pad, pad), mode="reflect")
    c = np.cumsum(np.asarray(xp, dtype=np.complex128))
    ma = (c[w:] - c[:-w]) / float(w)  # len = n  (verified: n + 2*pad - w + 1)
    y = z - ma.astype(np.complex64)
    info = {"dc_method": "moving_average", "dc_window": w}
    return y, info
