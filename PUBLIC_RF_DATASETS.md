# Public Open-Source Datasets for RF Testing & Evaluation

A curated reference guide of open-source RF (Radio Frequency) datasets, over-the-air SDR captures, and baseband audio/IQ recordings for testing automated signal detection, modulation classification (AMC), parameter extraction, and demodulation pipelines.

---

## 1. SigMF Collections & Recordings Repository
* **Source**: [https://github.com/sigmf/sigmf-recordings](https://github.com/sigmf/sigmf-recordings)
* **Format**: Standard **SigMF** (`.sigmf-data` raw IQ pairs paired with `.sigmf-meta` JSON metadata).
* **Signal Types**: Real-world captures including ADS-B (aircraft transponders), FM broadcast radio, ISM band IoT sensors (433 MHz / 915 MHz), and Wi-Fi beacons.
* **Best Used For**:
  - Validating standard SigMF metadata parsers and channelization.
  - Testing carrier frequency and bandwidth extraction against real over-the-air transmissions with noise and fading.
* **How to Ingest**:
  Point the ingestion pipeline directly at the `.sigmf-data` file or use the companion `.sigmf-meta` file for exact sample rate ($F_s$) and center frequency ($f_c$).

---

## 2. RadioML / DeepSig Datasets (RadioML 2016.10a & 2018.01a)
* **Source**: [https://www.deepsig.ai/datasets](https://www.deepsig.ai/datasets)
* **Format**: NumPy (`.npy`) and HDF5 (`.h5`) format containing complex floating-point baseband time-series slices ($128$ or $1024$ samples per window).
* **Signal Types**: 11 to 24 analog and digital modulation schemes:
  - **PSK**: BPSK, QPSK, 8PSK
  - **QAM**: 16-QAM, 64-QAM
  - **FSK / FM**: GFSK, CPFSK, WBFM
  - **AM**: AM-DSB, AM-SSB
  - Imparted with realistic synthetic channel effects: carrier frequency offset (CFO), sample clock offset, multipath fading (Rician / Rayleigh), and AWGN SNR sweeps from **-20 dB to +18 dB**.
* **Best Used For**:
  - Benchmarking Automatic Modulation Classification (AMC) classifiers (e.g., Higher-Order Cumulants $C_{40}, C_{42}, C_{63}$ or ML models).
  - Evaluating classifier breakdown thresholds at low SNR ($< 0\text{ dB}$).

---

## 3. Signal Identification Guide (SigIDWiki)
* **Source**: [https://www.sigidwiki.com/](https://www.sigidwiki.com/)
* **Format**: Direct downloadable `.wav` (mono audio / stereo IQ) and `.iq` / `.raw` files.
* **Signal Types**: Extensive encyclopedia of hundreds of real-world radio signals:
  - VHF/UHF Paging (POCSAG, FLEX)
  - Marine Telemetry (AIS, NAVTEX)
  - Military & HF Data links (STANAG, MIL-STD-188)
  - Satellite Downlinks (NOAA APT, Meteor-M, Inmarsat)
  - Trunked radio systems (TETRA, P25, DMR)
* **Best Used For**:
  - Testing real-world, non-standard waveforms and legacy digital modulations.
  - Validating Hilbert transform conversion from real `.wav` audio to complex analytic baseband.

---

## 4. OpenWebRX & WebSDR Live Captures
* **Source**: [http://www.websdr.org/](http://www.websdr.org/) and [https://www.openwebrx.de/](https://www.openwebrx.de/)
* **Format**: 16-bit PCM RIFF `.wav` files (Stereo IQ: Left = $I$, Right = $Q$, or Mono filtered baseband).
* **Signal Types**: Live over-the-air amateur radio, shortwave broadcast, CW morse code, and time signals recorded via global distributed SDRs.
* **Best Used For**:
  - Immediate real-world testing without needing local SDR hardware (RTL-SDR, HackRF, USRP).
  - Direct compatibility with `load_wav()` stereo and mono audio ingestion routines.

---

## Ingestion & Pipeline Testing Quick Reference

| Source Format | CLI Execution Example | Pipeline Handling |
| :--- | :--- | :--- |
| **Float32 IQ** (`.iq`, `.raw`) | `python main.py --file capture.iq --fs 1000000 --datatype cf32_le` | Direct unit-norm baseband mapping |
| **Int16 / RTL-SDR IQ** | `python main.py --file capture.iq --fs 2048000 --datatype cs16` | Linear de-quantization to $[-1.0, 1.0]$ |
| **Stereo IQ WAV** | `python main.py --file recording.wav` | Auto-detected: $L \rightarrow I$, $R \rightarrow Q$ |
| **Mono Audio WAV** | `python main.py --file audio.wav` | Auto-detected: FFT Hilbert analytic signal |
