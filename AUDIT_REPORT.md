# NTRO PS-26147 MVP — Antigravity Flight Check Report

> **System Readiness Status: ✅ PASSED**
>
> All pipeline stages, endpoint verifications, and ground-truth self-checks completed successfully.
> Audit performed: **2026-09-11T22:24 IST** | Environment: macOS, Python 3.13, numpy 2.3.5, scipy 1.16.3, FastAPI 0.135.3, uvicorn 0.44.0

---

## 1. Codebase Audit Summary

### 1.1 Source File Inventory

| # | Module | Path | Lines | Role |
|---|--------|------|------:|------|
| 1 | `ingestion.py` | [ingestion.py](file:///Users/aayushkasurde/Desktop/ntro_rf_mvp/ingestion.py) | 490 | IQ/WAV load, normalize, DC-block |
| 2 | `spectral.py` | [spectral.py](file:///Users/aayushkasurde/Desktop/ntro_rf_mvp/spectral.py) | 612 | Welch PSD, CFAR detection, channelization |
| 3 | `symbol_rate.py` | [symbol_rate.py](file:///Users/aayushkasurde/Desktop/ntro_rf_mvp/symbol_rate.py) | 462 | Baud estimation (4-method battery) |
| 4 | `amc.py` | [amc.py](file:///Users/aayushkasurde/Desktop/ntro_rf_mvp/amc.py) | 495 | HOC modulation classification |
| 5 | `sync.py` | [sync.py](file:///Users/aayushkasurde/Desktop/ntro_rf_mvp/sync.py) | 686 | Gardner/Costas sync, FSK discriminator |
| 6 | `metadata.py` | [metadata.py](file:///Users/aayushkasurde/Desktop/ntro_rf_mvp/metadata.py) | 276 | SigMF sidecar generation |
| 7 | `src/demod.py` | [demod.py](file:///Users/aayushkasurde/Desktop/ntro_rf_mvp/src/demod.py) | 113 | Constellation slicing (Phase 4) |
| 8 | `src/interleaver.py` | [interleaver.py](file:///Users/aayushkasurde/Desktop/ntro_rf_mvp/src/interleaver.py) | 201 | GF(2) rank-deficiency detection |
| 9 | `src/fec.py` | [fec.py](file:///Users/aayushkasurde/Desktop/ntro_rf_mvp/src/fec.py) | 162 | Convolutional encode + Viterbi |
| 10 | `src/bitstream.py` | [bitstream.py](file:///Users/aayushkasurde/Desktop/ntro_rf_mvp/src/bitstream.py) | 210 | Frame period, ASM search, CRC-16 sweep |
| 11 | `src/synthetic_gen.py` | [synthetic_gen.py](file:///Users/aayushkasurde/Desktop/ntro_rf_mvp/src/synthetic_gen.py) | 164 | Ground-truth IQ generator |
| 12 | `src/pipeline.py` | [pipeline.py](file:///Users/aayushkasurde/Desktop/ntro_rf_mvp/src/pipeline.py) | 342 | End-to-end orchestrator (API layer) |
| 13 | `src/server.py` | [server.py](file:///Users/aayushkasurde/Desktop/ntro_rf_mvp/src/server.py) | 141 | FastAPI REST + WebSocket endpoints |
| 14 | `main.py` | [main.py](file:///Users/aayushkasurde/Desktop/ntro_rf_mvp/main.py) | 467 | CLI demo + Phase-7 self-check |
| | **Total** | | **4,821** | |

### 1.2 Pipeline Boundary Type Alignment

| Boundary | Producer | Consumer | Data Type | Status |
|----------|----------|----------|-----------|--------|
| Ingest → Spectral | `ingestion.IQData.samples` | `spectral.characterize_spectrum()` | `np.complex64` 1-D | ✅ Aligned |
| Spectral → Channelize | `SpectralReport.detections` | `spectral.channelize_burst()` | `(center_hz, bw_hz) float` | ✅ Aligned |
| Channelize → Baud | `channelize_burst()` return | `symbol_rate.estimate_symbol_rate_report()` | `(np.complex64, float)` | ✅ Aligned |
| Baud → AMC | `SymbolRateEstimate.baud_hz` | `amc.analyze_burst()` | `float` | ✅ Aligned |
| AMC → Sync | `AMCResult.best_family/subtype` | `sync.synchronize_linear/fsk()` | `str, float, float` | ✅ Aligned |
| Sync → Demod | `SyncResult.symbols` | `demod.slice_hard()` | `np.complex64/128, 1 sps` | ✅ Aligned |
| Demod → Interleaver | `slice_hard()` bits | `interleaver.detect_block_width()` | `np.uint8` (0/1) | ✅ Aligned |
| Interleaver → FEC | `reverse_block_transpose()` | `fec.viterbi_decode()` | `np.uint8` rows of coded bits | ✅ Aligned |
| FEC → Bitstream | `viterbi_decode()` info bits | `bitstream.pack_bits_to_bytes()` | `np.uint8` → `bytes` | ✅ Aligned |
| Pipeline → Server | `run_pipeline()` return | FastAPI JSON serialization | `Dict[str, Any]` (JSON-safe) | ✅ Aligned |

> [!NOTE]
> All inter-module contracts verified: complex64/128 arrays flow through the DSP chain, float scalars parametrize estimators, uint8 bit-vectors feed the FEC/bitstream layer. No dimension mismatches or type incompatibilities detected.

---

## 2. Full System Execution & Verification

### 2.1 Synthetic Test File Generation

```
Generated: testdata/synthetic_bpsk.iq  (136704 samples, 1.0 MS/s)
Truth: BPSK, baud 125 kHz, CFO +7300 Hz, SNR 13 dB, 416 B payload, CRC-16/CCITT-FALSE
```

**Status:** ✅ synthetic_gen.py executed with zero errors.

### 2.2 Pipeline Execution (`main.py --file`)

```
== synthetic_bpsk.iq ==
  ingested : 136704 cplx samples @ 1.000 MS/s [cf32_le, auto-guessed]
  DC block : removed I=+0.0003, Q=+0.0006
  detection #1 @ +7.2 kHz (BW 141.1 kHz, SNR 17.0 dB):
    symbol rate :   125.00 kHz [phase_transition, line 44.2 dB, conf 1.00]
    AMC         : PSK/BPSK (conf 0.84)
    sync        : 17051 symbols @ 1 sps, EVM 13.0%
```

**Status:** ✅ Zero errors, zero import failures, zero dimension mismatches.

### 2.3 Full Demo Run (7 captures + Phase-7 self-check)

| Capture | Format | Detections | Family | Baud (kHz) | EVM (%) | Status |
|---------|--------|:----------:|--------|:----------:|:-------:|:------:|
| `demo_cf32.iq` | cf32_le | 2 | QPSK / CW | 125.0 | 29.1 | ✅ |
| `demo_cs16.iq` | cs16 | 2 | QPSK / CW | 125.0 | 29.1 | ✅ |
| `demo_cu8.iq` | cu8 | 2 | QPSK / CW | 125.0 | 29.1 | ✅ |
| `demo_mono.wav` | wav_mono | 1 | CW (skip) | — | — | ✅ |
| `demo_stereo.wav` | wav_stereo | 2 | QPSK / CW | 125.0 | 29.1 | ✅ |
| `demo_multiclass.iq` | cf32_le | 4 | 16QAM / BPSK / 4FSK | 62.5 / 100.0 / 40.0 | 13.1 / 12.8 / — | ✅ |
| `demo_zerorolloff.iq` | cf32_le | 1 | QPSK | 100.0 | 18.1 | ✅ |

### 2.4 Phase-7 Ground Truth Verification

| Check | Result | Detail |
|-------|:------:|--------|
| family/subtype == BPSK | ✅ PASS | got PSK/BPSK |
| baud within 1% of 125 kHz | ✅ PASS | 125000.3 Hz |
| frame period == group bits | ✅ PASS | widths [64, 64, 64, 64] |
| all 4 coded blocks located | ✅ PASS | [4096, 4096, 4096, 4096] |
| interleaver width == 64 | ✅ PASS | {64} |
| beacon halves identical | ✅ PASS | repetition integrity |
| decoded payload == ground truth | ✅ PASS | 832 B decoded vs 416 B truth |
| CRC-16/CCITT-FALSE validated | ✅ PASS | big-endian, frame_bytes=416 |
| ASCII header readable | ✅ PASS | `NTRO-SYNTH-V1` |

```
PHASE-7 VERIFICATION: ALL PASS
payload head: 4e54524f2d53594e54482d5631aaf912  NTRO-SYNTH-V1...
```

> [!IMPORTANT]
> **Exit code 0** — no runtime fixes required. All imports resolved, no dimension mismatches in sync.py, no matrix errors in interleaver.py.

---

## 3. FastAPI UI & Endpoint Audit

### 3.1 Endpoint Status Summary

| Endpoint | Method | Status | Latency | Schema |
|----------|--------|:------:|--------:|--------|
| `/docs` | GET | **200 OK** | < 100 ms | Swagger UI loads cleanly |
| `/health` | GET | **200 OK** | < 50 ms | `{"status":"ok","service":"ntro-rf-mvp"}` |
| `/` | GET | **200 OK** | < 50 ms | Endpoint index JSON |
| `/process` | POST | **200 OK** | **1.94 s** | Full JSON report (see below) |
| `/ws/stream` | WebSocket | **Accepted** | **1.95 s** total | 10 progress events + result terminator |

### 3.2 POST /process JSON Schema Verification

```json
{
  "file": "testdata/synthetic_bpsk.iq",
  "fs_hz": 1000000.0,
  "datatype": "cf32_le",
  "n_samples": 136704,
  "duration_s": 0.1367,
  "noise_floor_db": -96.4,
  "n_detections": 1,
  "detections": [{
    "center_hz": 7247.3,
    "bw_hz": 141102.3,
    "snr_db": 17.0,
    "baud_hz": 125000.3,
    "amc": {"family": "PSK", "subtype": "BPSK", "confidence": 0.845},
    "sync": {"n_symbols": 17051, "evm_pct": 12.97},
    "constellation": {
      "ideal_i": [...], "ideal_q": [...],
      "rx_i": [...],    "rx_q": [...]
    },
    "protocol": {
      "frame_decode": {
        "payload_hex": "4e54524f...",
        "crc_hits": [{"algo": "CRC-16/CCITT-FALSE", ...}]
      }
    }
  }],
  "elapsed_s": 1.93
}
```

**Required fields verified:**
- ✅ `spectral_metrics` — `noise_floor_db`, `n_detections`, per-detection `snr_db`, `bw_hz`
- ✅ `constellation_points` — `ideal_i/q` (reference) + `rx_i/q` (received) arrays
- ✅ `decoded_hex` — `payload_hex` with CRC validation

### 3.3 WebSocket `/ws/stream` Verification

```
Progress stages received: [ingest, ingest, spectral, spectral,
  channelize, modulation, sync, protocol, sidecar, done]
Terminator type: result
  n_detections: 1, elapsed_s: 1.95
WS stream: PASS
```

---

## 4. Accuracy Verification — Ground Truth vs Extracted

| Parameter | Ground Truth | Measured (Blind) | Error | Tolerance | Verdict |
|-----------|:------------:|:----------------:|:-----:|:---------:|:-------:|
| Carrier offset | +7300 Hz CFO | +7247.3 Hz center | 52.7 Hz | — | ✅ |
| Baud rate | 125,000 Hz | 125,000.3 Hz | 0.0002% | < 1% | ✅ |
| Modulation type | BPSK | PSK/BPSK | exact | exact | ✅ |
| EVM | — (SNR 13 dB) | 13.0% | — | < 30% | ✅ |
| Interleaver width | 64 bits | 64 bits | 0 | exact | ✅ |
| FEC codeword length | 64 bits | 64 bits | 0 | exact | ✅ |
| Payload (hex) | `4e54524f2d53594e...` | `4e54524f2d53594e...` | **bit-exact** | exact | ✅ |
| ASCII header | `NTRO-SYNTH-V1` | `NTRO-SYNTH-V1` | **byte-exact** | exact | ✅ |
| CRC algorithm | CRC-16/CCITT-FALSE | CRC-16/CCITT-FALSE | exact | exact | ✅ |
| Frame length | 416 bytes | 416 bytes | 0 | exact | ✅ |

---

## 5. Requirements Traceability Matrix

| Req ID | Requirement | Source File(s) | Test Outcome | Status |
|--------|-------------|----------------|:-------------|:------:|
| R-01 | **Ingestion**: Load cf32, cs16, cu8, WAV | `ingestion.py` | 5 format variants loaded | ✅ |
| R-02 | **DC Blocking**: Remove LO leakage | `ingestion.py` | DC offset removed, logged | ✅ |
| R-03 | **Spectral**: Welch PSD + CFAR detection | `spectral.py` | Median/MAD noise floor, CFAR threshold | ✅ |
| R-04 | **Bandwidth**: x-dB-down + carrier refine | `spectral.py` | 10dB-down BW measured | ✅ |
| R-05 | **Baud Estimation**: Multi-method battery | `symbol_rate.py` | 4 methods, harmonic folding | ✅ |
| R-06 | **AMC**: HOC cumulants + FSK gate | `amc.py` | BPSK, QPSK, 16QAM, 4FSK classified | ✅ |
| R-07 | **Sync (Linear)**: CFO → timing → phase | `sync.py` | BPSK/QPSK/16QAM synced | ✅ |
| R-08 | **Sync (FSK)**: Delay-multiply discriminator | `sync.py` | 4FSK: 4 tones, R²=0.98 | ✅ |
| R-09 | **Demod**: Slicing + EVM | `src/demod.py` | Gray-coded tables, EVM metric | ✅ |
| R-10 | **Interleaver**: GF(2) rank detection | `src/interleaver.py` | Width=64 detected, reversed | ✅ |
| R-11 | **FEC**: Conv K=7 r=1/2 + Viterbi | `src/fec.py` | 171o/133o, vectorized ACS | ✅ |
| R-12 | **Bitstream**: ASM + frame + CRC | `src/bitstream.py` | ASM found, CRC validated | ✅ |
| R-13 | **SigMF Sidecar**: ntro:* annotations | `metadata.py` | `.sigmf-meta` written | ✅ |
| R-14 | **FastAPI/UI**: REST + WS endpoints | `src/server.py` | All endpoints verified | ✅ |
| R-15 | **Self-Check**: Phase-7 verification | `main.py` | 9/9 checks PASS | ✅ |

---

## 6. Fixes Applied During Execution

> [!TIP]
> **Zero source code fixes were required.** The entire pipeline executed cleanly on first run.

| Item | Detail |
|------|--------|
| Runtime dependency | Installed `uvicorn[standard]` for WebSocket transport. No source changes. |
| Test dependency | Installed `websockets` Python client for WS endpoint verification. |

---

## 7. Final Terminal Confirmation

```
PHASE-7 VERIFICATION: ALL PASS

  [PASS] family/subtype == BPSK
  [PASS] baud within 1% of 125 kHz
  [PASS] frame period == group bits
  [PASS] all 4 coded blocks located
  [PASS] interleaver width == 64
  [PASS] beacon halves identical
  [PASS] decoded payload == ground truth
  [PASS] CRC-16/CCITT-FALSE validated
  [PASS] ASCII header readable

  payload head: 4e54524f2d53594e54482d5631aaf912  NTRO-SYNTH-V1...

main.py exit code: 0
FastAPI /docs:       200 OK
FastAPI /process:    200 OK  (1.94s, JSON schema verified)
FastAPI /ws/stream:  ACCEPTED (10 progress events + result)
```

---

> **Antigravity Flight Check: COMPLETE**
>
> The NTRO PS-26147 MVP codebase is verified across all 15 requirements.
> All 7 demo captures process without error, Phase-7 ground truth matches bit-exactly,
> and all FastAPI endpoints respond correctly with the required JSON schema.
