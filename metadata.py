"""
metadata.py -- SigMF sidecar generation (Phase 1, NTRO PS-26147)
================================================================

Wraps every ingested capture in a SigMF v1 ``<name>.sigmf-meta`` record
placed next to the data file, so Phase-2 tooling (and any external SigMF
consumer, e.g. IQEngine / sigmf-cli) can discover sample rate, datatype,
tune frequency and detected-emission annotations.

* Uses the ``sigmf`` package when available (native validation + sha512 of
  the data file computed automatically).
* Falls back to writing the *exact same* JSON schema by hand when the
  package is missing -- the output stays spec-compliant either way.

Custom (non-core) keys follow the vendor-prefix convention: ``ntro:*``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ingestion import IQData

try:  # sigmf >= 1.x exposes the schema keys at module level
    import sigmf as _sigmf_mod
    from sigmf import SigMFFile

    HAVE_SIGMF = True
except Exception:  # ImportError or broken install -- degrade gracefully
    _sigmf_mod = None
    SigMFFile = None
    HAVE_SIGMF = False

log = logging.getLogger(__name__)

__all__ = [
    "HAVE_SIGMF",
    "sigmf_datatype_for",
    "build_annotation",
    "write_sigmf_meta",
    "read_sigmf_meta",
]

# --- SigMF core keys (stable across spec versions; used verbatim by the
# --- fallback writer and resolved through the package when present) --------
_SIGMF_VERSION = "1.0.0"


def _key(name: str, fallback: str) -> str:
    """Resolve a schema key through the sigmf package, else use the literal."""
    if _sigmf_mod is not None and hasattr(_sigmf_mod, name):
        return str(getattr(_sigmf_mod, name))
    return fallback


K_DATATYPE = _key("DATATYPE_KEY", "core:datatype")
K_SAMPLE_RATE = _key("SAMPLE_RATE_KEY", "core:sample_rate")
K_NUM_SAMPLES = _key("NUM_SAMPLES_KEY", "core:num_samples")
K_VERSION = _key("VERSION_KEY", "core:version")
K_AUTHOR = _key("AUTHOR_KEY", "core:author")
K_DESCRIPTION = _key("DESCRIPTION_KEY", "core:description")
K_DATETIME = _key("DATETIME_KEY", "core:datetime")
K_FREQUENCY = _key("FREQUENCY_KEY", "core:frequency")
K_SHA512 = _key("SHA512_KEY", "core:sha512")
K_LABEL = _key("LABEL_KEY", "core:label")
K_FLO = _key("FREQ_LOWER_EDGE_KEY", "core:freq_lower_edge")
K_FHI = _key("FREQ_UPPER_EDGE_KEY", "core:freq_upper_edge")
K_START = _key("START_INDEX_KEY", "core:sample_start")
K_EXTENSIONS = _key("EXTENSIONS_KEY", "core:extensions")
K_LENGTH = _key("LENGTH_INDEX_KEY", "core:sample_length")

# WAV PCM encodings -> SigMF datatype strings (for sidecars pointing at WAVs).
# SigMF grammar: (c|r)(f32|f64|i32|i16|u32|u16|i8|u8)(_le|_be)? -- i.e. complex
# signed int16 is "ci16_le" in SigMF even though SDR tooling calls it "cs16".
_WAV2SIGMF = {
    "int16": "ci16_le",
    "uint8": "cu8",
    "int8": "ci8",
    "int32": "ci32_le",
    "float32": "cf32_le",
    "float64": "cf64_le",
}


def sigmf_datatype_for(iq: IQData) -> str:
    """Map an ingest-format tag to a legal SigMF datatype string.

    Ingest tags follow SDR convention (``cs16`` = complex signed 16-bit);
    SigMF spells the same thing ``ci16_le``, so the mapping is done here,
    at the metadata boundary.
    """
    dt = (iq.datatype or "cf32_le").lower()
    known = {"cf32_le", "cf32_be", "ci16_le", "ci16_be", "ci32_le", "ci32_be",
             "cu8", "ci8", "cf64_le", "cf64_be"}
    if dt in known:
        return dt
    alias = {"cf32": "cf32_le", "cs16": "ci16_le", "cs32": "ci32_le",
             "cs16_le": "ci16_le", "cs16_be": "ci16_be"}
    if dt in alias:
        return alias[dt]
    # WAV sources: describe the *file's* PCM encoding; the processing chain
    # (Hilbert / L-R->IQ) is recorded in core:description.
    base = _WAV2SIGMF.get(str(iq.notes.get("wav_dtype", "")), "cf32_le")
    if dt == "wav_mono_hilbert":
        # Real mono recording -> SigMF "r" (real) type: the dataset sample
        # count then equals the WAV frame count, keeping annotations aligned
        # with the file contents.
        return "r" + base.split("c", 1)[1]
    return base


# --------------------------------------------------------------------------- #
# Annotations (detections -> SigMF)
# --------------------------------------------------------------------------- #
def build_annotation(
    detection,
    n_samples: int,
    center_freq_hz: Optional[float] = None,
    label: str = "detected_emission",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert a :class:`spectral.Detection` into a SigMF annotation dict.

    SigMF annotation frequency edges are RF-absolute. When the capture's tune
    frequency is unknown (raw IQ with no `center_freq_hz`), edges are emitted
    as baseband offsets referenced to 0 Hz and the label says so -- downstream
    tools can shift them once the RF frequency is provisioned.

    `extra` merges Phase-2 measurement keys (e.g. ``ntro:baud_hz``,
    ``ntro:modulation_family``) into the annotation; vendor keys win over
    defaults so downstream stages can override derived quantities.
    """
    base = float(center_freq_hz) if center_freq_hz is not None else 0.0
    if center_freq_hz is None:
        label += " (baseband offset)"
    flo = base + float(getattr(detection, "f_low_hz"))
    fhi = base + float(getattr(detection, "f_high_hz"))
    ann = {
        K_START: 0,
        K_LENGTH: int(n_samples),  # annotate the whole capture window
        K_LABEL: label,
        K_FLO: flo,
        K_FHI: fhi,
        "ntro:carrier_hz": base + float(getattr(detection, "carrier_hz")),
        "ntro:bandwidth_hz": float(getattr(detection, "bandwidth_hz")),
        "ntro:snr_db": float(getattr(detection, "snr_db")),
        "ntro:bandwidth_method": str(getattr(detection, "bandwidth_method")),
    }
    if extra:
        ann.update({k: _jsonable(v) for k, v in extra.items()})
    return ann


# --------------------------------------------------------------------------- #
# Sidecar writer
# --------------------------------------------------------------------------- #
def _iso_utc_now() -> str:
    """SigMF datetime: ISO-8601 UTC with 'Z' suffix, e.g. 2026-09-11T09:41:00.123456Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_global_info(
    iq: IQData, author: Optional[str], description: Optional[str]
) -> Dict[str, Any]:
    g: Dict[str, Any] = {
        K_VERSION: _SIGMF_VERSION,
        K_DATATYPE: sigmf_datatype_for(iq),
        K_NUM_SAMPLES: len(iq),
        K_DESCRIPTION: description
        or f"PS-26147 Phase-1 ingest of {iq.source_path.name} ({iq.datatype})",
        # Declare the project-local vendor extension so ntro:* keys validate
        # (SigMF requires extensions to be declared in core:extensions).
        K_EXTENSIONS: [{"name": "ntro", "version": "0.1.0", "optional": True}],
    }
    if iq.sample_rate is not None:
        g[K_SAMPLE_RATE] = float(iq.sample_rate)
    if author:
        g[K_AUTHOR] = author
    # provenance mirror (ntro:* = project-local vendor prefix)
    for note in ("dc_method", "dc_window", "dc_removed_iq"):
        if note in iq.notes:
            g[f"ntro:{note}"] = _jsonable(iq.notes[note])
    if iq.notes.get("datatype_guessed"):
        g["ntro:datatype_auto_guessed"] = True
    return g


def _jsonable(obj: Any) -> Any:
    """numpy -> plain-python so json.dump never chokes on provenance values."""
    if isinstance(obj, (tuple, list)):
        return [_jsonable(o) for o in obj]
    if hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    return obj


def write_sigmf_meta(
    iq: IQData,
    annotations: Sequence[Dict[str, Any]] = (),
    *,
    author: Optional[str] = None,
    description: Optional[str] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """Write a ``.sigmf-meta`` sidecar for an ingested capture.

    The sidecar is placed next to the original data file (SigMF naming:
    ``<stem>.sigmf-meta`` beside ``<stem>.<ext>``) and points at it as the
    dataset. Annotations (typically one per :class:`spectral.Detection`)
    carry the measured frequency edges + SNR into the metadata record.

    Returns the path of the written metadata file.
    """
    if iq.source_path is None:
        raise ValueError("IQData has no source file -- cannot place a sidecar")
    data_path = Path(iq.source_path)
    meta_path = Path(out_path) if out_path else data_path.with_suffix(".sigmf-meta")

    ginfo = _build_global_info(iq, author, description)
    capture: Dict[str, Any] = {K_START: 0, K_DATETIME: _iso_utc_now()}
    if iq.center_freq_hz is not None:
        capture[K_FREQUENCY] = float(iq.center_freq_hz)

    if HAVE_SIGMF:
        meta = SigMFFile(data_file=str(data_path), global_info=ginfo)
        meta.add_capture(start_index=0, metadata=capture)
        for ann in annotations:
            extra = {k: v for k, v in ann.items() if k not in (K_START, K_LENGTH)}
            meta.add_annotation(
                start_index=int(ann.get(K_START, 0)),
                length=int(ann.get(K_LENGTH, len(iq))),
                metadata=extra,
            )
        try:
            meta.tofile(str(meta_path), overwrite=True)  # validates + hashes
        except TypeError:  # sigmf < 1.2 has no overwrite kwarg
            if meta_path.exists():
                meta_path.unlink()
            meta.tofile(str(meta_path))
        except Exception as exc:
            # Never let a metadata hiccup kill the ingest batch: fall back to
            # writing the package's own JSON dump directly.
            log.warning("SigMF tofile failed for %s (%s) -- writing JSON directly",
                        meta_path.name, exc)
            meta_path.write_text(meta.dumps(), encoding="utf-8")
    else:
        # Fallback: emit the exact SigMF JSON schema by hand. Consumers see an
        # identical document; only sha512 of the dataset is skipped.
        log.warning(
            "sigmf package not installed -- writing schema-compliant JSON "
            "directly (no dataset sha512)"
        )
        doc = {
            "global": ginfo,
            "captures": [capture],
            "annotations": [dict(a) for a in annotations],
        }
        meta_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    log.info("wrote %s", meta_path.name)
    return meta_path


def read_sigmf_meta(path: "str | Path") -> Dict[str, Any]:
    """Load a .sigmf-meta file into a plain dict (through sigmf if available)."""
    p = Path(path)
    if HAVE_SIGMF:
        from sigmf.sigmffile import fromfile

        return json.loads(fromfile(str(p)).dumps())
    return json.loads(p.read_text(encoding="utf-8"))
