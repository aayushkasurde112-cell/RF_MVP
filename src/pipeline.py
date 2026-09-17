"""Phase 6 -- end-to-end orchestration of the NTRO RF MVP DSP chain.

Layer-cake per detection (mirrors main.py but importable by the API):

    ingest -> Welch PSD -> burst detection -> channelize
           -> symbol rate -> SNR -> AMC family + subtype
           -> sync (Costas/Gardner linear, non-coherent FSK)
           -> Phase 4: hard slicing + EVM, rank-deficiency interleave probe
           -> Phase 5: bitstream frame analysis (period, entropy, ASM, CRC)

Every stage emits progress events through `progress(stage, pct, detail)` so
the WebSocket endpoint can stream live status.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ingestion  # noqa: E402  (Phase 1, flat at project root)
import metadata
import spectral
import symbol_rate
import amc
import sync
from src import demod, interleaver, fec, bitstream  # noqa: E402  (Phase 4/5)

ProgressCB = Callable[[str, float, Dict[str, Any]], None]


def _j(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {str(k): _j(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_j(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _j(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    return obj


def decode_framed_payload(bits: np.ndarray, tol: int = 2,
                          asm_bits: Optional[np.ndarray] = None
                          ) -> Dict[str, Any]:
    """Blind framed-payload extraction (Phase 4+5 chain, shared by the API
    probe and main.py's Phase-7 verification):

      1. locate ASM markers (bipolar correlation, `tol` bit errors)
      2. payload chunks = bit spans between marker END and next marker START
         (adjacent trailing+leading markers naturally yield empty spans)
      3. per chunk: rank-deficiency interleaver-width detect -> de-transpose
      4. per 64-bit row: Viterbi (r=1/2, K=7) -> first 26 info bits (G0')
      5. pack to bytes in transmission order

    Returns {asm_offsets, chunks, block_widths, payload_bytes, ...}.
    """
    if asm_bits is None:
        asm_bits = np.array([(0x1ACFFC1D >> (31 - i)) & 1
                             for i in range(32)], dtype=np.uint8)
    offs = bitstream.find_sync(bits, asm_bits, tol=tol)
    out: Dict[str, Any] = {"asm_offsets": offs, "chunks": [],
                           "block_widths": [], "payload_bytes": b"",
                           "crc_hits": []}
    if len(offs) < 2:
        return out
    gaps = [(offs[i] + 32, offs[i + 1])
            for i in range(len(offs) - 1)
            if offs[i + 1] - offs[i] - 32 >= 64]   # adjacent pair -> skipped
    info_parts = []
    for lo, hi in gaps:
        chunk = bits[lo:hi]
        try:
            w, _prof = interleaver.detect_block_width(chunk, max_width=256)
            mat = interleaver.reverse_block_transpose(
                chunk[: (chunk.size // w) * w], w, w)
            rows = mat.reshape(-1, w)
            k_info = w // 2 - (7 - 1)              # r=1/2, K=7 tail
            row_bits = np.concatenate(
                [fec.viterbi_decode(r)[:k_info] for r in rows])
        except Exception as exc:
            out["chunks"].append({"lo": lo, "hi": hi, "error": str(exc)})
            continue
        out["chunks"].append({"lo": lo, "hi": hi, "bits": int(chunk.size),
                              "width": w})
        out["block_widths"].append(w)
        info_parts.append(row_bits)
    if info_parts:
        payload = bitstream.pack_bits_to_bytes(np.concatenate(info_parts))
        out["payload_bytes"] = payload
        # The CRC terminates each frame; frames may repeat (beacon), so the
        # sweep runs at candidate frame boundaries (full / half / quarter).
        # The smallest frame with a hit is the true frame length.
        cands = sorted({n for n in (len(payload), len(payload) // 2,
                                    len(payload) // 4)
                        if n >= 16})
        hits = []
        for n in cands:
            for a, e in bitstream.crc_sweep(payload[:n]):
                hits.append({"algo": a, "endianness": e, "frame_bytes": n})
        out["crc_hits"] = hits
    return out


def _blind_protocol_probe(symbols: np.ndarray, subtype: str) -> Dict[str, Any]:
    """Phase 4+5 bitstream layer: slice, probe interleaving, look for frames.

    symbols : 1 sps synchronized unit-power symbols (linear families only).
    """
    out: Dict[str, Any] = {}
    try:
        # Phase ambiguity resolution for linear modulations:
        # Costas loop converges within the constellation's rotational symmetry class
        # (2 states for BPSK, 4 states for QPSK). Test rotations against ASM.
        rot_candidates = [0]
        if subtype in ("BPSK", "PSK"):
            rot_candidates = [0, 1]
            rot_step = np.pi
        elif subtype in ("QPSK", "4PSK"):
            rot_candidates = [0, 1, 2, 3]
            rot_step = np.pi / 2
        else:
            rot_step = np.pi / 2

        asm_bits = np.array([(0x1ACFFC1D >> (31 - i)) & 1
                             for i in range(32)], dtype=np.uint8)

        best_bits, best_evm, best_diag = None, 1e9, {}
        best_hits = -1
        for k in rot_candidates:
            rot_syms = symbols * np.exp(-1j * k * rot_step)
            b, ev, dg = demod.slice_hard(rot_syms, subtype)
            hits = len(bitstream.find_sync(b, asm_bits, tol=2))
            if hits > best_hits:
                best_hits = hits
                best_bits, best_evm, best_diag = b, ev, dg

        bits, evm, diag = best_bits, best_evm, best_diag
        out["slicing"] = {"evm_pct": round(evm, 2), "n_bits": int(bits.size),
                          "modulation": diag["modulation"]}
    except Exception as exc:                                  # unknown subtype
        return {"error": f"slicing failed: {exc}"}

    # -- Phase 5 bitstream structure ---------------------------------------
    # allow periods up to half the stream (+ margin): a beacon that
    # repeats its frame twice has the repeat distance AT n/2
    d_max = int(min(max(bits.size // 2 + 64, 64), 65536))
    period, z = bitstream.frame_period(bits, d_min=32, d_max=d_max)
    out["frame_period"] = {"bits": period, "z_score": z}

    asm_offsets = bitstream.find_sync(bits, asm_bits, tol=2)
    out["asm_offsets"] = asm_offsets

    # entropy profile: where does high-entropy payload stop?
    try:
        edge, drop = bitstream.entropy_boundary(bits, window=64, step=4)
        out["entropy_boundary"] = {"edge_bit": edge, "drop": round(drop, 3)}
    except Exception as exc:
        out["entropy_boundary"] = {"error": str(exc)}

    # -- frame extraction + full payload decode ----------------------------
    decoded = decode_framed_payload(bits, asm_bits=asm_bits)
    out["frame_decode"] = {
        "n_chunks": len(decoded["chunks"]),
        "block_widths": decoded["block_widths"],
        "payload_bytes": int(len(decoded["payload_bytes"])),
        "crc_hits": decoded["crc_hits"],
        "payload_hex": decoded["payload_bytes"].hex() if decoded["payload_bytes"] else None,
    }
    if decoded["payload_bytes"]:
        head = decoded["payload_bytes"][:64]
        out["payload_preview_hex"] = head.hex()
        out["payload_preview_ascii"] = "".join(
            chr(c) if 32 <= c < 127 else "." for c in head)
    else:
        out["payload_preview_hex"] = None
        out["payload_preview_ascii"] = None
    return out


def _process_detection(iq, det, k: int, emit, emit_base: float) -> Dict[str, Any]:
    """Channelize + classify + synchronize + protocol-probe one region.

    Raises on implausible inputs; the caller records the failure per
    detection so one spurious noise region cannot fail a whole request.
    """
    entry: Dict[str, Any] = {
        "detection": k,
        "center_hz": round(float(det.center_hz), 1),
        "bw_hz": round(float(det.bandwidth_hz), 1),
    }

    emit("channelize", 0.20 + emit_base, {"detection": k})
    z_bb, fs_bb = spectral.channelize_burst(iq.samples, float(iq.sample_rate),
                                            det.center_hz, det.bandwidth_hz)

    sr = symbol_rate.estimate_symbol_rate_report(z_bb, fs_bb)
    baud = float(sr.baud_hz)
    # plausibility: need >= 2 samples/symbol AND a matched filter that fits
    # in the burst (guards against baud lines locked onto noise in tiny
    # regions -> downstream scipy filtfilt padlen blowups)
    sps_native = fs_bb / baud if baud > 0 else 0.0
    if not (2.0 <= sps_native <= max(z_bb.size, 1) / 100.0):
        raise ValueError(
            f"implausible baud {baud:.1f} Hz for {z_bb.size} channeled "
            f"samples (sps {sps_native:.1f})")
    snr_db = float(spectral.estimate_burst_snr(z_bb, fs_bb, det.bandwidth_hz))
    emit("modulation", 0.25 + emit_base,
         {"baud_hz": round(baud, 1), "snr_db": round(snr_db, 1)})

    # roll-off from measured occupied BW: B = (1+alpha) f_baud (clamped)
    alpha = float(np.clip(float(det.bandwidth_hz) / baud - 1.0, 0.05, 1.0))
    am = amc.analyze_burst(z_bb, fs_bb, baud, snr_db=snr_db, alpha=alpha)
    emit("sync", 0.35 + emit_base,
         {"family": am.best_family, "confidence": round(am.confidence, 3)})

    entry.update({
        "snr_db": round(snr_db, 1),
        "baud_hz": round(baud, 1),
        "amc": {"family": am.best_family,
                "confidence": round(float(am.confidence), 3),
                "subtype": am.subtype_guess,
                "ranking": _j(am.family_ranking)},
        "symbol_rate_method": sr.method,
    })

    # ---------------- Phase 3: synchronization ------------------------
    symbols = None
    if am.best_family == "FSK":
        sy = sync.synchronize_fsk(z_bb, fs_bb, baud)
        entry["sync"] = {
            "branch": "fsk", "n_symbols": int(sy.n_symbols),
            "n_tones": sy.diag.get("n_tones"),
            "tones_hz": sy.diag.get("tone_freqs_hz"),
            "occupancy": _j(sy.diag.get("tone_occupancy")),
            "timing_r2": sy.diag.get("fsk_timing_r2"),
        }
    else:
        sy = sync.synchronize_linear(z_bb, fs_bb, baud, alpha=alpha,
                                     subtype=am.subtype_guess)
        symbols = sy.symbols
        entry["sync"] = {"branch": "linear",
                         "n_symbols": int(sy.n_symbols),
                         "subtype": sy.subtype,
                         "evm_pct": (round(float(sy.evm_pct), 2)
                                     if sy.evm_pct is not None else None),
                         "cfo_removed_hz": sy.diag.get("cfo_corrected_hz"),
                         "timing": sy.diag.get("timing_source")}
        # constellation vectors for the frontend (downsampled)
        pts, _ = demod.constellation_table(sy.subtype or "QPSK")
        n_show = min(512, symbols.size)
        entry["constellation"] = {
            "ideal_i": [float(v.real) for v in pts],
            "ideal_q": [float(v.imag) for v in pts],
            "rx_i": _j(symbols[:n_show].real.astype(np.float32)),
            "rx_q": _j(symbols[:n_show].imag.astype(np.float32)),
            "modulation": sy.subtype or "QPSK",
        }

    # --------------- Phase 4/5: demod + bitstream probe ---------------
    if symbols is not None and symbols.size >= 2048:
        emit("protocol", 0.55 + emit_base, {"detection": k})
        entry["protocol"] = _blind_protocol_probe(symbols,
                                                  sy.subtype or "QPSK")
    return entry


def run_pipeline(path: str, fs: Optional[float] = None,
                 datatype: Optional[str] = None,
                 center_hz: Optional[float] = None,
                 progress: Optional[ProgressCB] = None) -> Dict[str, Any]:
    """Full Phase 1-5 chain over one capture. Returns a JSON-safe report."""
    t0 = time.perf_counter()

    def emit(stage: str, pct: float, detail: Optional[Dict[str, Any]] = None):
        if progress:
            progress(stage, pct, detail or {})

    emit("ingest", 0.02, {"path": str(path)})
    p = Path(path)

    # 1. Auto-discover fs and datatype from companion .sigmf-meta if not provided
    meta_candidates = [
        p.with_suffix(".sigmf-meta"),
        Path(str(p) + ".sigmf-meta"),
    ]
    for mpath in meta_candidates:
        if mpath.is_file():
            try:
                mdict = metadata.read_sigmf_meta(mpath)
                g = mdict.get("global", {})
                if fs is None and "core:sample_rate" in g:
                    fs = float(g["core:sample_rate"])
                if datatype is None and "core:datatype" in g:
                    datatype = str(g["core:datatype"])
                break
            except Exception:
                pass

    # 2. Ingest capture (WAV audio container vs raw IQ)
    if p.suffix.lower() == ".wav":
        iq = ingestion.load_wav(p)
        if fs is not None and fs > 0:
            iq.sample_rate = float(fs)
    else:
        # If fs is still unknown for raw captures, fallback to standard 1.0 MS/s default
        fallback_used = False
        if fs is None or fs <= 0:
            fs = 1_000_000.0  # 1 MS/s standard SDR default
            fallback_used = True

        iq = ingestion.load_iq(p, fs, datatype=datatype)
        if fallback_used:
            iq.notes["fs_defaulted"] = True

    if iq.sample_rate is None or iq.sample_rate <= 0:
        iq.sample_rate = 1_000_000.0
        iq.notes["fs_defaulted"] = True

    emit("ingest", 0.06, {"n_samples": int(iq.samples.size),
                          "fs": float(iq.sample_rate),
                          "datatype": iq.datatype})

    # ---------------- Phase 1: spectral characterization -------------------
    emit("spectral", 0.10, {})
    analysis_band = None
    if iq.datatype == "wav_mono_hilbert" and iq.sample_rate:
        analysis_band = (0.0, float(iq.sample_rate) / 2.0)
    rep = spectral.characterize_spectrum(iq.samples, float(iq.sample_rate),
                                        analysis_band=analysis_band)
    detections = sorted(rep.detections,
                        key=lambda d: -d.bandwidth_hz)  # widest first
    if center_hz is not None:   # Phase-1 lesson: wideband may split a burst
        best = [d for d in detections
                if abs(d.center_hz - center_hz) < 0.05 * float(iq.sample_rate)]
        detections = best or detections
    emit("spectral", 0.16, {"n_detections": len(detections),
                            "noise_floor_db": round(rep.noise.median_db, 1)})

    # ---------------- per-detection processing ------------------------------
    results = []
    for k, det in enumerate(detections[:4]):          # cap work per request
        try:
            results.append(_process_detection(iq, det, k, emit, 0.1 * k))
        except Exception as exc:   # one bad region must not fail the request
            results.append({"detection": k,
                            "error": f"{type(exc).__name__}: {exc}"})

    emit("sidecar", 0.92, {})
    sidecar_note = "sigmf sidecar written alongside the capture"
    try:
        meta_path = metadata.write_sigmf_meta(iq)
        sidecar_note = str(meta_path.name)
    except Exception as exc:
        sidecar_note = f"sidecar write skipped: {exc}"

    emit("done", 1.0, {})
    return _j({
        "file": str(path),
        "fs_hz": float(iq.sample_rate),
        "datatype": iq.datatype,
        "n_samples": int(iq.samples.size),
        "duration_s": round(iq.samples.size / float(iq.sample_rate), 4),
        "noise_floor_db": round(float(rep.noise.median_db), 1),
        "n_detections": len(detections),
        "detections": results,
        "sidecar": sidecar_note,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    })
