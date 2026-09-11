# PS-26147 — Phase 1: RF Signal Parameter Extraction (Python Orchestration Layer)

MVP pipeline for **ingest → normalize → SigMF → spectral characterization** of
RF captures. Built as pure, stateless functions over `numpy` arrays so the
same code runs offline today and per-chunk inside the future streaming /
FastAPI backend.

## Layout

| File | Role |
|---|---|
| `ingestion.py` | `.iq` / `.wav` loaders, datatype auto-guess, ±1 normalization, DC blocking |
| `metadata.py`  | SigMF `.sigmf-meta` sidecar writer (+ spec-compliant fallback if `sigmf` is absent) |
| `spectral.py`  | Welch PSD, median/MAD CFAR threshold, carrier peak search, x-dB-down bandwidth, burst channelization, in-band SNR |
| `symbol_rate.py` | **(P2)** baud estimation: envelope/phase-transition/delay-multiply spectral lines + zero-crossing fallback |
| `amc.py`       | **(P2)** HOC (C40/C42/C63) extraction, symbol synchronization, FSK-vs-PSK-vs-QAM decision tree |
| `sync.py`      | **(P3)** synchronization chain: M-th-power coarse CFO, Gardner TED + Farrow interpolator, Costas loop, non-coherent FSK discriminator |
| `main.py`      | End-to-end demo: synthesizes known scenes, runs the full Phase-1+2+3 pipeline |
| `requirements.txt` | `numpy`, `scipy`, `sigmf` |
| `testdata/`    | Generated demo captures + their `.sigmf-meta` sidecars |

## Quick start

```bash
pip install -r requirements.txt
python main.py                        # synth + process the 5-file demo set
python main.py --file cap.iq --fs 2.4e6 [--datatype cs16]
python main.py --file rec.wav
```

Library use (e.g. from FastAPI):

```python
import ingestion, spectral, metadata

iq = ingestion.load_iq("cap.iq", sample_rate=2.4e9)          # or load_wav(...)
rep = spectral.characterize_spectrum(iq.samples, iq.sample_rate)
anns = [metadata.build_annotation(d, len(iq)) for d in rep.detections]
metadata.write_sigmf_meta(iq, annotations=anns)
```

## Demo scene ground truth (deterministic, seed 26147)

CW tone **+120 kHz** · shaped QPSK **−250 kHz** (~125 kHz symbol-null BW) ·
AWGN σ=0.05 · injected LO leakage DC **0.06+0.04j**. Expected output: two
detections; tone carrier recovers 120.00 kHz exactly; QPSK center ≈ −249 kHz,
10-dB-down BW ≈ 136 kHz (125 kHz null BW + filter transition band).

## Key DSP decisions

- **Datatype guess** (`ingestion.guess_datatype_auto`): modulo sanity check on
  file size with the ladder `cf32_le → cs16 → cu8`; when size % 8 == 0 (all
  formats byte-legal) a float32 plausibility probe on the first 64 KiB descends
  the ladder if the bytes are clearly not f32 IQ. `cu8` vs `cs16` is
  content-ambiguous — pass `datatype=` explicitly (you know your radio).
- **Mono WAV → analytic signal** via FFT Hilbert (`x + j·H{x}`), cancelling the
  negative-frequency image. Stereo WAV: L=I, R=Q. Analytic captures are
  analyzed on `[0, fs/2)` only — the dead half-band would poison the noise
  statistics (`analysis_band=`).
- **DC blocker**: centered moving-average subtraction (O(N) via cumsum,
  reflect-padded edges) tracking slow LO wander; whole-record mean for short
  captures; `method="recursive"` (one-pole, O(1) state) provided for the
  streaming build.
- **Noise floor / threshold**: `median + k·1.4826·MAD` of the dB-PSD — a
  self-calibrating CFAR energy detector (k=6 → per-bin P_FA ≈ 1e-9). Median/MAD
  resist the signal power that biases mean/std; a fixed-dB offset cannot track
  gain/temperature changes.
- **Carrier**: FFT argmax + 3-point parabolic interpolation in log-power domain
  → ~1/10-bin accuracy (demo: 0.4 Hz at 30.5 Hz bins). For flat-topped
  modulated emissions, `center_hz` (edge midpoint) is the meaningful metric;
  `carrier_hz` is the PSD peak.
- **Bandwidth**: x-dB-down walk on a 3-bin median-filtered PSD with linearly
  interpolated crossings; falls back to threshold-region edges when SNR ≤ x dB.
- **Regions**: wrap-aware (a carrier on DC stitches across ±fs/2) + 2-bin
  morphological closing so skirt notches don't split one emission in two.
- **SigMF**: sidecar beside the data file, `core:` schema keys + `ntro:*`
  vendor extension (declared in `core:extensions`) carrying per-detection
  SNR/BW/carrier. Mono real WAVs map to `ri16_le`, stereo IQ WAVs to
  `ci16_le` (SDR's "cs16" spelled SigMF's way).

## Streaming notes (Phase 2 hooks)

Loaders/DSP are array-in/array-out with no global state; the moving-average DC
blocker already has an O(1)-state recursive twin; `welch_psd` accepts
`nperseg`/`noverlap` for ring-buffer chunking. Remaining work for a streaming
front-end: chunked FramedWelch accumulator, FIR Hilbert (odd taps) replacing
the FFT version, and a hysteresis state machine around the CFAR mask.

## Phase 2 — symbol-rate estimation & AMC

Per Phase-1 detection: `spectral.channelize_burst()` extracts the burst as
complex baseband (DDC + zero-phase Kaiser FIR with 30% passband slack +
decimation), then:

**`symbol_rate.py`** — four baud estimators scored on a common scale (line SNR
vs a *local* median floor of the transformed spectrum), best survivor wins:
1. envelope spectrum of |x|², |x|, x² (needs roll-off α > 0),
2. phase-transition spectrum `|angle(x[n]x*[n−1])|` (works at α = 0 and FSK),
3. delay-and-multiply cyclic-autocorrelation tone, τ swept geometrically with
   an FSK-aware candidate τ = 1/(2·Δtone) from the instantaneous-frequency
   modes,
4. hysteretic zero-crossing count (short-burst fallback).
Harmonic folding guards against locking onto 2·f_baud. Returns float (NaN =
no credible line, e.g. CW); the report form adds method/score/candidates.

**`amc.py`** — FSK gate first (envelope-modulation index `10log10(1 +
var| x|²/mean²)` + instantaneous-frequency state count — constant-envelope
*and* multi-state ⇒ FSK; unshaped PSK is constant-envelope but single-state).
Then: residual-CFO removal via the **4th-power spectral line** (E[x⁴] ≠ 0 for
BPSK/QPSK/QAM; tone at 4·f_res; prominence vs a median-filtered local
baseline), RRC matched filter (roll-off from occupied BW / baud − 1), timing
recovery by maximizing |C42| over one symbol period, unit-power cumulants
**C40/C42/C63** (C63 verified against brute-force partition enumeration), and
a threshold decision tree over (|C40|, |C42|, |C63|) against exact anchors:

| anchor | C40 | C42 | C63 |
|---|---|---|---|
| BPSK  | 2.00 | 2.00 | 14.00 |
| QPSK  | 1.00 | 1.00 |  5.00 |
| 8PSK  | 0.00 | 1.00 |  5.00 |
| 16QAM | 0.68 | 0.68 |  4.04 |
| 64QAM | 0.62 | 0.62 |  5.44 |

Confidence = softmax over anchor distances (with the box decision as a logit
bonus), aggregated to the ranked `{family: score}` dict. Cumulants are
de-biased with the *in-band* burst SNR (`spectral.estimate_burst_snr`) using
C40/C42 noise-bias inversions. Note |C40| = |C42| for square QAM — the
|C40|/|C42| ratio alone cannot separate QAM from PSK; the joint 3-D space
does.

Demo ground truth vs measured (see `python main.py` log): baud estimates
exact to <0.01% on all five modulations; families correct 5/5; cumulants land
on anchors (e.g. BPSK 1.92/1.92/13.6 ≈ 2/2/14 at ~17 dB SNR).

## Phase 3 — synchronization chain (`sync.py`)

Routed per burst by the AMC result; output is a clean **1-sample-per-symbol**
unit-power symbol array (saved as `<capture>_det<k>_<family>_syms.cf32` and
summarized with an I/Q snippet + EVM in the demo log).

Linear branch (PSK/QAM):
1. RRC matched filter at the channelizer rate -> resample to 2 samples/symbol.
2. **Coarse CFO**, M-th power method (M = 2 BPSK, 4 QPSK/QAM, 8 8PSK): x^M
   strips the data, FFT shows the residual tone at M·f_err (prominence vs a
   median-filtered local baseline), parabolic-refined, mixdown.
3. **Fine timing**: Gardner TED at 2 sps,
   e(k) = Re{conj(y(k−T/2))·(y(kT) − y(kT−T))}/P, driving a 2nd-order loop
   whose fractional interval steps a 4-tap **Farrow interpolator**
   (Catmull-Rom cubic). Engineering notes baked into the code: the TED has a
   quasi-stable T/2 null (acquisition grid covers both nulls), the loop gains
   are normalized by the *measured* detector gain Kd (finite-difference of
   the S-curve; assuming Kd=1 left the loop with ~4% of its designed
   restoring force), error feedback is clipped in Kd-normalized units, and
   the delivered symbols come from whichever of {closed-loop, open-loop
   replay} measures the better eye.
4. **Carrier phase**: decision-directed Costas loop (nearest-constellation
   slicer per subtype; locks within the constellation symmetry class).

FSK branch (non-coherent): squared delay-and-multiply discriminator
w(t) = z²(t)·conj(z²(t−τ)) => arg w = 4π·f_inst·τ (symbol phase cancels —
no Costas loop), with the discriminator delay chosen from the measured tone
span so the outer MFSK tones stay inside the fs/(4τ) ambiguity limit.
Symbol timing = fractional-offset search maximizing the Otsu between-class
variance R²; decisions = nearest-tone indices mapped to a unit-power ring.

Measured (demo log): 16QAM 13.0% EVM with **99.8% of symbols within 0.35 of
an ideal grid point**; BPSK 12.8% EVM, 99.9% grid match (closed-loop Gardner
selected); QPSK 29.1% at a 12.8-dB-SNR floor of ~23%; 4FSK uniform 4-tone
occupancy (±2% of truth) with timing R² = 0.98.

---

# Phase 4 -- Demodulation + Interleaver Reverse-Engineering

Modules: `src/demod.py`, `src/interleaver.py` (package shim in `src/__init__.py`
keeps the Phase 1-3 flat modules importable alongside `src.*`).

**src/demod.py**
- `constellation_table(subtype)` -- ideal points + MSB-first Gray label table
  for BPSK / QPSK / 8PSK / 16QAM / 64QAM (unknown -> QPSK fallback).
- `modulate(symbols, subtype)` -- inverse map (label-dict lookup), used by tests.
- `slice_hard(symbols, subtype)` -- nearest-neighbor slicing -> uint8 bits,
  returns `(bits, evm_pct, diag)` with EVM = 100 * sqrt(mean|r-d|^2 / mean|d|^2).

**src/interleaver.py**
- `_pack_rows` uint64 bit-packing + `gf2_rank` (packed Gauss-Jordan over GF(2),
  validated against a brute-force oracle on 9 shapes incl. multi-word W > 64).
- `rank_deficiency_profile` -- deficiency = min(rows, W) - rank against the
  random-matrix null (raw W - rank favors wide-short matrices).
- `detect_block_width` / `block_interleave` / `reverse_block_transpose` /
  `detect_and_reverse` -- spike detection + reversal of an N x W block
  interleaver (row-write, column-read transpose).

Test battery `tests/test_phase4.py`: **ALL PASS** (BER-clean round trips per
subtype with sigma scaled by d_min; EVM matches 100*sqrt(2)*sigma theory;
64x64 spike test -> unique width 64, exact reversal; GF(2) note:
rank([r | 1-r]) = 5, the complement adds the all-ones basis vector).

# Phase 5 -- FEC + Bitstream Layer

Modules: `src/fec.py`, `src/bitstream.py`.

**src/fec.py** -- convolutional codes + Viterbi:
- `code_tables(K, polys)` cached: next-state = reg>>1 with input at MSB,
  output = parity(reg & poly); default K=7 [0o171, 0o133], K=5 [0o35, 0o23].
- `conv_encode(bits, tail=True)` -- interleaved [G0,G1] rows, K-1 tail zeros
  (500 info bits -> (506, 2) coded bits).
- `viterbi_decode(c, tail=True)` -- hard (Hamming) or soft P(bit=1)
  (Euclidean) metrics; per-timestep ACS over explicit predecessor pairs
  p0/p1 (fancy-scatter assignment is last-write-wins and never correct);
  traceback stores the winning input bit AND predecessor selector per
  (t, state).

**src/bitstream.py**
- `frame_period` -- bipolar mismatch autocorrelation, z-scored:
  z(d) = (0.5 - p(d)) * sqrt(n-d) / 0.5; sync markers are a tiny duty cycle
  in long frames, so raw dips drown in variance (needs z >= 5).
- `entropy_profile` / `entropy_boundary` -- sliding 64-bit Bernoulli entropy,
  5-tap smoothed steepest-drop edge.
- `find_sync` -- bipolar correlation ASM search with `tol` bit errors
  (CCSDS ASM 0x1ACFFC1D).
- `crc16` + registry (CCITT-FALSE / XMODEM / AUG-CCITT / IBM / MODBUS / USB /
  X-25 / DNP), `crc_sweep` (both byte orders vs trailing 2 bytes),
  MSB-first pack/unpack.

Test battery `tests/test_phase5.py`: **ALL PASS** -- clean / 3%-error /
soft (sigma=0.6) decodes exact, K=5 variant, frame period 532 @ z=40,
random -> None, entropy edge 820 vs truth 800, ASM offsets tol=1,
CRC sweep identifies 4 algos and rejects corruption.

# Phase 6 -- FastAPI Service

Modules: `src/pipeline.py` (orchestration, importable), `src/server.py`.

- `run_pipeline(path, fs, datatype, center_hz, progress)` -- full chain with
  per-detection error containment (a spurious noise region cannot fail a
  request) and JSON-safe output (constellation vectors included).
- `decode_framed_payload(bits, tol)` -- shared blind extractor: ASM search ->
  inter-chunk spans -> rank-deficiency width detect -> de-transpose ->
  per-row Viterbi -> byte packing -> CRC sweep at frame boundaries.
- REST: `GET /health`, `POST /process {"path","fs","datatype","center_hz"}`;
  WS: `/ws/stream` streams `{"type":"progress", stage, pct}` events then a
  `{"type":"result"}` terminator (pipeline runs in an executor, progress via
  asyncio queue).
- Terminal test: `uvicorn src.server:app --host 0.0.0.0 --port 8000 &`,
  `curl POST /process` on testdata/demo_cf32.iq -> JSON report in 1.6 s;
  bad path -> 400; WS progress streamed; synthetic file -> full protocol
  decode in the response JSON.

# Phase 7 -- Synthetic Ground Truth + Closed-Loop Verification

Module: `src/synthetic_gen.py`; verification wired into `main.py`
(`run_phase7_verification()`).

Reference chain: payload (ASCII header + random + 0xAA pad + CRC-16/CCITT)
-> 26-bit rows -> K=7 conv (64-bit codewords) -> 64x64 block interleave ->
[ASM | blk | ASM | blk | ASM] group, beacon-repeated with guard bits ->
BPSK (RRC 0.35, sps 8) -> CFO +7.3 kHz, fractional timing offset, AWGN 13 dB.

Design note: the pad is 0xAA, not 0x00 -- a zero-padded payload biases the
bitstream (72 % zeros), producing a discrete carrier line + baud-spaced clock
lines that hijack the spectral detector; 0xAA keeps P(1)=0.5 (DC-balanced)
while remaining a near-zero-entropy segment.

Terminal result (`logs/final_p7_full.log`, full `python main.py`, exit 0):
baud 125000.6 vs 125000, AMC BPSK conf 0.84, EVM 13.0 %, frame period 8288
(z=90), all 6 ASMs, 4 blocks x width 64, decoded 832 B, beacon halves
identical, **payload hex == ground truth exactly**, CRC-16/CCITT-FALSE
validated at 416 B frame, ASCII header `NTRO-SYNTH-V1` readable.
**PHASE-7 VERIFICATION: ALL PASS.**
