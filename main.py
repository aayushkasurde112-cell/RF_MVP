"""
main.py -- End-to-end Phase-1 + Phase-2 pipeline demo (NTRO PS-26147)
=====================================================================

Phase 1: ingest -> normalize -> DC-block -> Welch PSD -> median/MAD CFAR
detection -> carrier / bandwidth -> SigMF sidecars.

Phase 2: per detection -> channelize burst -> symbol-rate (baud) estimation
(nonlinear-transform spectral lines + fallbacks) -> HOC (C40/C42/C63)
automatic modulation classification -> family ranking, folded back into the
SigMF annotations as ntro:* keys.

1. Synthesizes dummy captures with KNOWN ground truth into ./testdata/:

     demo_cf32.iq     float32 IQ : CW @ +120 kHz, shaped QPSK block
                      @ -250 kHz (baud 125 k), AWGN sigma=0.05, injected DC
                      offset 0.06+0.04j (simulated LO leakage)
     demo_cs16.iq     same signal, int16 IQ quantization
     demo_cu8.iq      same signal, uint8 IQ quantization
     demo_mono.wav    real int16 WAV of the CW tone (Hilbert path)
     demo_stereo.wav  int16 L/R WAV carrying the full IQ scene
     demo_multiclass.iq   three bursts: BPSK (baud 100 k, @ +80 kHz, RRC 0.35),
                      16-QAM (baud 62.5 k, @ -180 kHz, RRC 0.25) and 4-FSK
                      (baud 40 k, tone spacing 25 k, @ +230 kHz)
     demo_zerorolloff.iq  rectangular-pulse (zero roll-off) QPSK @ -40 kHz,
                      baud 100 k -- exercises the delay-multiply / phase
                      -transition fallbacks where envelope lines vanish

2. Runs every file through the full chain and prints measured vs ground-truth
   parameters.

Usage
-----
    python main.py                          # synth + process the demo set
    python main.py --file mycap.iq --fs 2.4e6 [--datatype cs16]
    python main.py --file myrec.wav
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal
from scipy.io import wavfile

import ingestion
import metadata
import spectral
import symbol_rate
import amc
import sync

log = logging.getLogger("main")

TESTDIR = Path(__file__).resolve().parent / "testdata"

# Ground truth used to synthesize the demo scenes (baseband Hz, amplitude).
TONE_FREQ_HZ = 120_000.0        # CW carrier
TONE_AMP = 0.35
QPSK_CENTER_HZ = -250_000.0     # shaped QPSK emission
QPSK_AMP = 0.5
QPSK_SPS = 8                    # samples/symbol -> symbol rate = fs / 8
NOISE_SIGMA = 0.05
DC_OFFSET = 0.06 + 0.04j        # simulated LO leakage

FS = 1.0e6
N_SAMPLES = 1 << 17

# Phase-2 multiclass scene: (name, center_hz, amplitude, truth)
P2_BPSK = dict(name="BPSK", center=+80e3, amp=0.45, baud=100e3, alpha=0.35)
P2_QAM = dict(name="16QAM", center=-180e3, amp=0.50, baud=62.5e3, alpha=0.25)
P2_FSK = dict(name="4FSK", center=+230e3, amp=0.45, baud=40e3, tone_spacing=25e3)
P2_RECT = dict(name="QPSK(0)", center=-40e3, amp=0.50, baud=100e3)


def _rrc_pulse_train(rng: np.random.Generator, nsym: int, sps: int,
                     beta: float, points: np.ndarray) -> np.ndarray:
    """Unit-power RRC-shaped symbol stream (symbols drawn from `points`)."""
    ups = np.zeros(nsym * sps, dtype=np.complex128)
    ups[::sps] = rng.choice(points, nsym)
    taps = amc.rrc_taps(beta, 8, sps)
    return sp_signal.upfirdn(taps, ups)[: nsym * sps] * np.sqrt(sps)


def _awgn(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    return sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def _interleave(z: np.ndarray, dtype: np.dtype, scale: float, center: float = 0.0) -> np.ndarray:
    """Quantize complex samples and interleave I,Q for raw file writing."""
    lo, hi = (0, 255) if dtype == np.uint8 else (-32768, 32767)
    out = np.empty(2 * z.size, dtype=dtype)
    out[0::2] = np.clip(np.round(z.real * scale + center), lo, hi)
    out[1::2] = np.clip(np.round(z.imag * scale + center), lo, hi)
    return out


def synth_demo_captures(outdir: Path = TESTDIR, fs: float = FS,
                        n: int = N_SAMPLES) -> None:
    """Generate all dummy scenes above as .iq / .wav files (deterministic)."""
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(26147)
    t = np.arange(n) / fs

    # ---------------- Phase-1 scene: CW + shaped QPSK --------------------
    tone = TONE_AMP * np.exp(2j * np.pi * TONE_FREQ_HZ * t)

    # Pulse-shaped QPSK at -250 kHz (tests bandwidth + baud estimation).
    sym_pw = rng.standard_normal(n // QPSK_SPS)
    bits = rng.integers(0, 4, n // QPSK_SPS)
    sym = np.exp(1j * (np.pi / 4 + bits * np.pi / 2)) / np.sqrt(2.0)
    ups = np.zeros((n // QPSK_SPS) * QPSK_SPS, dtype=np.complex128)
    ups[::QPSK_SPS] = sym
    taps = sp_signal.firwin(9 * QPSK_SPS + 1, cutoff=1.0 / QPSK_SPS,
                            window="hamming")
    wide = QPSK_AMP * np.sqrt(QPSK_SPS) * sp_signal.lfilter(taps, 1.0, ups)
    wide = np.concatenate([wide[4 * QPSK_SPS:], np.zeros(4 * QPSK_SPS)])[:n]
    # Complex up-conversion: z(t) -> z(t)*e^{+j2pi f_c t} moves baseband to f_c.
    wide *= np.exp(2j * np.pi * QPSK_CENTER_HZ * t)

    noise = NOISE_SIGMA * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    z = (tone + wide + noise + DC_OFFSET).astype(np.complex64)

    z.tofile(outdir / "demo_cf32.iq")
    _interleave(z, np.int16, 32767.0).tofile(outdir / "demo_cs16.iq")
    _interleave(z, np.uint8, 127.5, center=127.5).tofile(outdir / "demo_cu8.iq")

    mono = (TONE_AMP * np.cos(2 * np.pi * TONE_FREQ_HZ * t)
            + NOISE_SIGMA * rng.standard_normal(n))
    wavfile.write(outdir / "demo_mono.wav", int(fs),
                  np.round(mono * 32767.0).astype(np.int16))

    st = np.empty((n, 2), dtype=np.int16)
    st[:, 0] = np.round(z.real * 32767.0)
    st[:, 1] = np.round(z.imag * 32767.0)
    wavfile.write(outdir / "demo_stereo.wav", int(fs), st)

    # ---------------- Phase-2 multiclass scene ---------------------------
    qpsk_pts = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2.0)
    a_lv = np.arange(-3, 5, 2)
    qam_pts = (a_lv[:, None] + 1j * a_lv[None, :]).ravel()
    qam_pts = qam_pts / np.sqrt(np.mean(np.abs(qam_pts) ** 2))

    scene = np.zeros(n, dtype=np.complex128)
    # BPSK, RRC 0.35, baud 100 kHz @ +80 kHz
    s = pad_to(_rrc_pulse_train(rng, n // 10, 10, P2_BPSK["alpha"],
                                np.array([1.0, -1.0])), n)
    scene += P2_BPSK["amp"] * s * np.exp(2j * np.pi * P2_BPSK["center"] * t)
    # 16-QAM, RRC 0.25, baud 62.5 kHz @ -180 kHz
    s = pad_to(_rrc_pulse_train(rng, n // 16, 16, P2_QAM["alpha"], qam_pts), n)
    scene += P2_QAM["amp"] * s * np.exp(2j * np.pi * P2_QAM["center"] * t)
    # 4-FSK, tone spacing 25 kHz, baud 40 kHz @ +230 kHz (CPFSK, const env)
    fsym = (rng.integers(0, 4, n // 25) - 1.5) * P2_FSK["tone_spacing"]
    ph = 2j * np.pi * np.cumsum(np.repeat(fsym, 25)) / fs
    s = np.exp(ph)
    scene += P2_FSK["amp"] * pad_to(s, n) * np.exp(2j * np.pi * P2_FSK["center"] * t)
    scene += _awgn(rng, n, NOISE_SIGMA)
    scene.astype(np.complex64).tofile(outdir / "demo_multiclass.iq")

    # ---------------- zero-roll-off scene (fallback battery) -------------
    sym_r = rng.choice(qpsk_pts, n // 10)
    s_rect = np.repeat(sym_r, 10)  # rectangular pulses: alpha = 0
    zr = (P2_RECT["amp"] * pad_to(s_rect, n)
          * np.exp(2j * np.pi * P2_RECT["center"] * t)
          + _awgn(rng, n, NOISE_SIGMA))
    zr.astype(np.complex64).tofile(outdir / "demo_zerorolloff.iq")

    log.info("synthesized 7 demo captures in %s", outdir)


def pad_to(s: np.ndarray, n: int) -> np.ndarray:
    return np.pad(s, (0, max(0, n - s.size)))[:n]


def run_phase2(iq: "ingestion.IQData", report: "spectral.SpectralReport") -> list:
    """Phase-2 per-detection stage: baud + AMC. Returns annotation extras."""
    fs = iq.sample_rate
    extras: list = []
    for i, d in enumerate(report.detections):
        if d.snr_db < 8.0 or d.bandwidth_hz < 2e3:
            print(f"  detection #{i + 1}: skipped (SNR {d.snr_db:.1f} dB / "
                  f"BW {d.bandwidth_hz / 1e3:.1f} kHz below extraction floor)")
            extras.append(None)
            continue
        z_bb, fs_bb = spectral.channelize_burst(
            iq.samples, fs, d.center_hz, d.bandwidth_hz)
        sr = symbol_rate.estimate_symbol_rate_report(z_bb, fs_bb)
        if not sr.found:
            print(f"  detection #{i + 1} @ {d.center_hz / 1e3:+.1f} kHz: "
                  "no baud line (CW carrier or noise)")
            extras.append(None)
            continue
        # true in-band SNR (Phase-1 snr_db is a peak metric, too low for a
        # wideband emission -- the cumulant de-bias needs the symbol-band SNR)
        snr_true = spectral.estimate_burst_snr(z_bb, fs_bb, d.bandwidth_hz)
        alpha = float(np.clip(d.bandwidth_hz / sr.baud_hz - 1.0, 0.05, 1.0))
        am = amc.analyze_burst(z_bb, fs_bb, sr.baud_hz, snr_db=snr_true,
                               alpha=alpha)

        ranking = "  ".join(f"{k} {v:.2f}" for k, v in am.family_ranking.items())
        sub = f"/{am.subtype_guess}" if am.subtype_guess else ""
        print(f"  detection #{i + 1} @ {d.center_hz / 1e3:+.1f} kHz "
              f"(BW {d.bandwidth_hz / 1e3:.1f} kHz, SNR {snr_true:.1f} dB):")
        print(f"    symbol rate : {sr.baud_hz / 1e3:8.2f} kHz  "
              f"[{sr.method}, line {sr.line_snr_db:.1f} dB, "
              f"conf {sr.confidence:.2f}]")
        fx = am.features
        print(f"    AMC         : {am.best_family}{sub} "
              f"(conf {am.confidence:.2f})  ranked: {ranking}")
        print(f"    cumulants   : |C40| {fx.get('C40_abs', 0):.2f}  "
              f"|C42| {fx.get('C42_abs', 0):.2f}  "
              f"|C63| {fx.get('C63_abs', 0):.2f}   "
              f"({fx.get('n_symbols', 0)} symbols, env-mod "
              f"{fx.get('env_mod_db', float('nan')):.2f} dB, "
              f"{fx.get('n_freq_states', 0)} freq state(s))")
        extras.append({
            "ntro:baud_hz": float(sr.baud_hz),
            "ntro:baud_method": sr.method,
            "ntro:baud_confidence": float(sr.confidence),
            "ntro:modulation_family": am.best_family,
            "ntro:modulation_subtype": am.subtype_guess,
            "ntro:amc_confidence": float(am.confidence),
            "ntro:c40_abs": float(fx.get("C40_abs", 0.0)),
            "ntro:c42_abs": float(fx.get("C42_abs", 0.0)),
            "ntro:c63_abs": float(fx.get("C63_abs", 0.0)),
        })

        # ---------------- Phase 3: synchronization chain -----------------
        # Route by the HOC family: FSK -> non-coherent delay-and-multiply
        # discriminator (bypasses the Costas loop); PSK/QAM -> coarse CFO
        # (M-th power) -> Gardner timing + Farrow -> Costas.
        try:
            if am.best_family == "FSK":
                sy = sync.synchronize_fsk(z_bb, fs_bb, sr.baud_hz)
            else:
                sy = sync.synchronize_linear(z_bb, fs_bb, sr.baud_hz,
                                             alpha=alpha,
                                             subtype=am.subtype_guess)
        except (ValueError, IndexError) as exc:
            print(f"    sync         : FAILED ({exc})")
            extras[-1]["ntro:sync_ok"] = False
            continue

        stem = iq.source_path.stem if iq.source_path else "capture"
        sym_file = TESTDIR / f"{stem}_det{i + 1}_{am.best_family.lower()}_syms.cf32"
        sy.symbols.astype(np.complex64).tofile(sym_file)
        if sy.evm_pct is not None:
            print(f"    sync         : {sy.n_symbols} symbols @ 1 sps, "
                  f"EVM {sy.evm_pct:.1f}%  [{sy.subtype} slicer, "
                  f"CFO {sy.diag.get('cfo_corrected_hz', 0):+.1f} Hz, "
                  f"timing {sy.diag.get('timing_source', 'fsk')}] -> {sym_file.name}")
            extras[-1]["ntro:evm_pct"] = float(sy.evm_pct)
            extras[-1]["ntro:sync_ok"] = True
        else:
            print(f"    sync         : {sy.n_symbols} symbols @ 1 sps "
                  f"(non-coherent FSK, {sy.diag['n_tones']} tones, "
                  f"timing R^2 {sy.diag['fsk_timing_r2']:.2f}) -> {sym_file.name}")
            extras[-1]["ntro:sync_ok"] = True
        extras[-1]["ntro:symbols_file"] = sym_file.name
        print("    first 8 synchronized symbols (I, Q):")
        for v in sy.symbols[:8]:
            print(f"      ({v.real:+.4f}, {v.imag:+.4f})")
    return extras


def process_capture(path: Path, fs: float | None,
                    datatype: str | None) -> spectral.SpectralReport:
    """Full chain for one file: ingest -> characterize -> baud -> AMC -> SigMF."""
    if path.suffix.lower() == ".wav":
        iq = ingestion.load_wav(path)
    else:
        iq = ingestion.load_iq(path, sample_rate=fs, datatype=datatype)

    # An analytic signal (mono WAV -> Hilbert) has NO content -- not even
    # noise -- on negative frequencies: its two-sided PSD is structurally
    # bimodal, which would poison the median/MAD noise floor. Scope the
    # analysis to the live half-band.
    analysis_band = None
    if iq.datatype == "wav_mono_hilbert" and iq.sample_rate:
        analysis_band = (0.0, float(iq.sample_rate) / 2.0)
    report = spectral.characterize_spectrum(
        iq.samples, iq.sample_rate, analysis_band=analysis_band)

    fs_mhz = (iq.sample_rate or 0) / 1e6
    print(f"\n== {path.name} ==")
    print(f"  ingested : {len(iq)} cplx samples  @ {fs_mhz:.3f} MS/s  "
          f"[{iq.datatype}" + (", auto-guessed]" if iq.notes.get("datatype_guessed") else "]"))
    if "dc_removed_iq" in iq.notes:
        di, dq = iq.notes["dc_removed_iq"]
        print(f"  DC block : removed I={di:+.4f}, Q={dq:+.4f} (LO leakage)")
    if analysis_band:
        print(f"  analysis band : [{analysis_band[0] / 1e3:.0f}, "
              f"{analysis_band[1] / 1e3:.0f}] kHz (analytic signal: dead half-band excluded)")
    for line in report.summary_lines():
        print(f"  {line}")

    extras = run_phase2(iq, report)

    anns = [
        metadata.build_annotation(d, n_samples=len(iq),
                                  center_freq_hz=iq.center_freq_hz,
                                  extra=extras[i])
        for i, d in enumerate(report.detections)
    ]
    metadata.write_sigmf_meta(iq, annotations=anns)
    return report


def run_phase7_verification(fs: float = 1.0e6) -> int:
    """Phase 7 -- synthetic reference chain + closed-loop self-check.

    1. src.synthetic_gen.generate() emits testdata/synthetic_bpsk.iq:
       BPSK (RRC 0.35, sps 8) carrying [ASM|conv-coded+interleaved block]x2
       groups + AWGN + CFO + timing offset, and writes the ground truth.
    2. The capture is then processed *blind* through the Phase 1-6 chain
       (ingest -> spectral -> baud -> AMC -> sync -> slicer -> ASM search ->
       rank-deficiency interleave detect -> per-row Viterbi -> CRC sweep).
    3. Every measured parameter is compared against the truth; any mismatch
       fails the check so the pipeline can self-correct.
    """
    from src import pipeline, synthetic_gen, demod   # Phase 4-7 modules

    print("\n" + "=" * 78)
    print("PHASE 7 -- synthetic reference chain (ground truth vs blind decode)")
    print("=" * 78)
    gt = synthetic_gen.generate(TESTDIR / "synthetic_bpsk.iq", fs=fs)
    print(f"  generated : {gt['file']}  ({gt['samples']} samples, "
          f"{gt['total_bits']} bits, group {gt['group_bits']} bits)")
    print(f"  truth     : BPSK baud {gt['baud_hz']:.0f} Hz, CFO "
          f"{gt['cfo_hz']:+.0f} Hz, SNR {gt['snr_db_target']:.0f} dB, "
          f"payload {len(gt['payload_hex']) // 2} B/frame "
          f"[{gt['crc_algo']}]")

    path = Path(gt["file"])
    iq = ingestion.load_iq(path, fs)
    rep = spectral.characterize_spectrum(iq.samples, float(iq.sample_rate))
    det = max(rep.detections, key=lambda d: d.bandwidth_hz)  # the burst
    z_bb, fs_bb = spectral.channelize_burst(iq.samples, float(iq.sample_rate),
                                            det.center_hz, det.bandwidth_hz)
    sr = symbol_rate.estimate_symbol_rate_report(z_bb, fs_bb)
    snr_db = float(spectral.estimate_burst_snr(z_bb, fs_bb, det.bandwidth_hz))
    alpha = float(np.clip(det.bandwidth_hz / sr.baud_hz - 1.0, 0.05, 1.0))
    am = amc.analyze_burst(z_bb, fs_bb, sr.baud_hz, snr_db=snr_db, alpha=alpha)
    sy = sync.synchronize_linear(z_bb, fs_bb, sr.baud_hz, alpha=alpha,
                                 subtype=am.subtype_guess)
    bits, evm_pct, _ = demod.slice_hard(sy.symbols, sy.subtype or "BPSK")
    decoded = pipeline.decode_framed_payload(bits)
    payload = decoded["payload_bytes"]

    n_fail = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal n_fail
        n_fail += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f" ({detail})" if detail else ""))

    print(f"  measured  : {am.best_family}/{am.subtype_guess} "
          f"(conf {am.confidence:.2f}), baud {sr.baud_hz:.1f} Hz, "
          f"{sy.n_symbols} symbols, EVM {evm_pct:.1f}%")

    check("family/subtype == BPSK", am.subtype_guess == "BPSK",
          f"got {am.best_family}/{am.subtype_guess}")
    check("baud within 1% of 125 kHz",
          abs(sr.baud_hz - gt["baud_hz"]) / gt["baud_hz"] < 0.01,
          f"{sr.baud_hz:.1f} Hz")
    check("frame period == group bits",
          any(c["width"] == 64 for c in decoded["chunks"]),
          f"period est + widths {decoded['block_widths']}")
    check("all 4 coded blocks located",
          len(decoded["chunks"]) == 4,
          f"{[c['bits'] for c in decoded['chunks']]}")
    check("interleaver width == 64", set(decoded["block_widths"]) == {64},
          f"{decoded['block_widths']}")
    half = len(gt["payload_hex"]) // 2
    check("beacon halves identical (repetition integrity)",
          payload[:half] == payload[half:2 * half])
    check("decoded payload == ground truth",
          payload[:half].hex() == gt["payload_hex"],
          f"{len(payload)} B decoded vs {half} B truth")
    check("CRC-16/CCITT-FALSE validated",
          any(h["algo"] == gt["crc_algo"] and h["frame_bytes"] == half
              for h in decoded["crc_hits"]),
          f"{decoded['crc_hits']}")
    ascii_head = payload[:13]
    check("ASCII header readable", ascii_head == b"NTRO-SYNTH-V1",
          ascii_head.decode("ascii", "replace"))

    print(f"  payload head  : {payload[:16].hex()}  "
          f"{''.join(chr(c) if 32 <= c < 127 else '.' for c in payload[:16])}")
    print(f"  PHASE-7 VERIFICATION: "
          f"{'ALL PASS' if n_fail == 0 else f'{n_fail} FAILURES'}")
    return n_fail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PS-26147 Phase-1+2 pipeline demo")
    parser.add_argument("--file", type=Path, help="ingest this capture instead of the demo set")
    parser.add_argument("--fs", type=float, default=None, help="sample rate [Hz] for raw IQ files")
    parser.add_argument("--datatype", type=str, default=None,
                        help="raw IQ datatype (cf32_le | cs16 | cu8); default: auto-guess")
    parser.add_argument("--no-synth", action="store_true", help="skip demo synthesis")
    parser.add_argument("--no-phase7", action="store_true",
                        help="skip the Phase-7 synthetic verification stage")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

    if args.file:
        if args.file.suffix.lower() != ".wav" and args.fs is None:
            parser.error("--fs is required for raw IQ files (no header carries it)")
        process_capture(args.file, args.fs, args.datatype)
        return 0

    if not args.no_synth:
        synth_demo_captures()

    print("\n" + "=" * 78)
    print("PS-26147 demo -- synthetic scenes (seed 26147), ground truth:")
    print(f"  [phase 1] CW @ {TONE_FREQ_HZ / 1e3:+.0f} kHz | QPSK @ {QPSK_CENTER_HZ / 1e3:+.0f} kHz, "
          f"baud {1e6 / QPSK_SPS / 1e3:.0f} kHz | AWGN sigma={NOISE_SIGMA} | DC "
          f"{DC_OFFSET.real}{DC_OFFSET.imag:+.2f}j")
    print(f"  [phase 2] BPSK baud {P2_BPSK['baud'] / 1e3:.1f} kHz @ {P2_BPSK['center'] / 1e3:+.0f} kHz | "
          f"16QAM baud {P2_QAM['baud'] / 1e3:.1f} kHz @ {P2_QAM['center'] / 1e3:+.0f} kHz |")
    print(f"            4FSK baud {P2_FSK['baud'] / 1e3:.1f} kHz @ {P2_FSK['center'] / 1e3:+.0f} kHz | "
          f"rect-QPSK baud {P2_RECT['baud'] / 1e3:.1f} kHz @ {P2_RECT['center'] / 1e3:+.0f} kHz")
    print("=" * 78)

    plan = [
        (TESTDIR / "demo_cf32.iq", FS, None),         # auto-guess: modulo + probe
        (TESTDIR / "demo_cs16.iq", FS, None),         # auto-guess descends ladder
        (TESTDIR / "demo_cu8.iq", FS, "cu8"),         # explicit: cu8 is ambiguous
        (TESTDIR / "demo_mono.wav", None, None),      # mono -> Hilbert
        (TESTDIR / "demo_stereo.wav", None, None),    # stereo -> L/R = I/Q
        (TESTDIR / "demo_multiclass.iq", FS, None),   # Phase-2 AMC showcase
        (TESTDIR / "demo_zerorolloff.iq", FS, None),  # Phase-2 fallback showcase
    ]
    for path, fs, dt in plan:
        try:
            process_capture(path, fs, dt)
        except ingestion.IngestionError as exc:
            # Robust per-file error handling: one bad capture must not kill
            # the batch (FastAPI layer maps this to HTTP 422).
            print(f"\n== {path.name} ==\n  INGESTION ERROR: {exc}")
            log.exception("ingestion failed")
    print("\nSigMF sidecars (with ntro:baud/modulation annotations) written "
          "next to each capture in", TESTDIR)

    # ---------------- Phase 7: synthetic reference verification ----------
    if not args.no_phase7:
        try:
            n_fail = run_phase7_verification()
        except Exception as exc:   # the self-check must not crash the demo
            print(f"  PHASE-7 VERIFICATION ERROR: {type(exc).__name__}: {exc}")
            log.exception("phase-7 verification failed")
            n_fail = 1
        return 1 if n_fail else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
