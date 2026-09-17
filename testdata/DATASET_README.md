# RF MVP Test Dataset (IQ & WAV)

This dataset contains synthetic and quantized RF captures designed for testing signal parameter extraction, carrier detection, symbol rate estimation, automatic modulation classification (AMC), and frame synchronization / decoding.

All files are pre-configured with a default sample rate of **1.0 MHz** ($F_s = 1,000,000\text{ Hz}$).

---

## 1. File Catalog & Specifications

### A. Format & Quantization Test Suite
- **`demo_cf32.iq`** (1.0 MB)
  - **Datatype**: `cf32_le` (Interleaved 32-bit floating point complex I/Q)
  - **Contents**: CW Tone @ +120 kHz, Root-Raised-Cosine QPSK @ -250 kHz (125 kbaud), AWGN noise ($\sigma=0.05$), and simulated LO DC leakage ($0.06 + 0.04j$).
- **`demo_cs16.iq`** (512 KB)
  - **Datatype**: `cs16` (Interleaved 16-bit signed integer complex I/Q)
  - **Contents**: Same dual-signal scene quantized to int16 (HackRF / Airspy format).
- **`demo_cu8.iq`** (256 KB)
  - **Datatype**: `cu8` (Interleaved 8-bit unsigned integer complex I/Q)
  - **Contents**: Same dual-signal scene quantized to uint8 (RTL-SDR format).

### B. Audio / Baseband WAV Files
- **`demo_stereo.wav`** (512 KB)
  - **Format**: 16-bit PCM RIFF Stereo WAV
  - **Mapping**: Left Channel = In-Phase ($I$), Right Channel = Quadrature ($Q$).
  - **Contents**: Full dual-signal RF scene in standard SDR stereo audio format.
- **`demo_mono.wav`** (256 KB)
  - **Format**: 16-bit PCM RIFF Mono WAV
  - **Mapping**: Real audio signal; converted into complex analytic baseband via Hilbert transform.
  - **Contents**: CW tone + noise.

### C. Multi-Modulation & Edge Cases
- **`demo_multiclass.iq`** (1.0 MB)
  - **Datatype**: `cf32_le`
  - **Contents**: 3 simultaneous signals:
    - **BPSK**: baud 100 kHz @ +80 kHz ($RRC, \alpha=0.35$)
    - **16-QAM**: baud 62.5 kHz @ -180 kHz ($RRC, \alpha=0.25$)
    - **4-FSK**: baud 40 kHz, tone spacing 25 kHz @ +230 kHz
- **`demo_zerorolloff.iq`** (1.0 MB)
  - **Datatype**: `cf32_le`
  - **Contents**: Rectangular-pulse (zero roll-off) QPSK @ -40 kHz, baud 100 kHz. Tests delay-multiply and phase-transition fallbacks.

### D. End-to-End Ground Truth Telemetry Captures
- **`synthetic_bpsk.iq`** (1.0 MB)
  - **Datatype**: `cf32_le`
  - **Contents**: Telemetry frame with CCSDS ASM (`0x1ACFFC1D`), $K=7$ rate $1/2$ convolutional encoding, $64\times 64$ block interleaving, CRC-16/CCITT-FALSE, and ASCII header payload (`NTRO-SYNTH-V1`). Accompanying ground truth in `synthetic_bpsk.truth.json`.
- **`qpsk_5db.iq`** & **`antigravity_qpsk_5db.iq`** (534 KB)
  - **Datatype**: `cf32_le`
  - **Contents**: Shaped QPSK signal at $5\text{ dB}$ SNR with known ground truth parameters.
- **`test_qpsk_12.iq`** & **`test_qpsk_snr.iq`** (534 KB)
  - **Datatype**: `cf32_le`
  - **Contents**: Additional QPSK test captures across varying SNR thresholds.

---

## 2. Accompanying Metadata Files
- **`*.sigmf-meta`**: Standard SigMF schema metadata containing core frequency, sample rate, datatype, and channelization annotations.
- **`*.truth.json`**: Ground truth definitions for automated pipeline verification (baud rate, modulation, SNR, payload bytes).
- **`*_syms.cf32`**: Synchronized symbol stream outputs extracted during sync and slicer stages.

---

## 3. Quickstart: Testing the Dataset

### Option 1: Python CLI
```bash
# Float32 IQ (auto-detect or explicit cf32_le)
python main.py --file testdata/demo_cf32.iq --fs 1000000 --datatype cf32_le

# Multi-modulation scene
python main.py --file testdata/demo_multiclass.iq --fs 1000000

# Stereo WAV
python main.py --file testdata/demo_stereo.wav

# Mono WAV
python main.py --file testdata/demo_mono.wav
```

### Option 2: FastAPI Web UI & REST API
```bash
# Start server
uvicorn src.server:app --host 0.0.0.0 --port 8000

# POST request via curl
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"path": "testdata/demo_cf32.iq", "fs": 1000000, "datatype": "cf32_le"}'
```
