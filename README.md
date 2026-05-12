# 🎬 Multimedia Compression Studio

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Signal%20Processing-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-orange?style=for-the-badge)

**A full-featured MP3-like audio encoder and MPEG-like video compressor built from scratch in Python.**

_Digital Signal Processing Practical Exam — Suez Canal University, Faculty of Computers and Informatics, Computer Science Dept._

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [System Architecture](#-system-architecture)
- [Audio Compression Pipeline](#-audio-compression-pipeline)
- [Video Compression Pipeline](#-video-compression-pipeline)
- [GUI Application](#-gui-application)
- [Installation](#-installation)
- [Usage](#-usage)
- [Output Files](#-output-files)
- [Technical Details](#-technical-details)
- [Team](#-team)

---

## 🔍 Overview

**Multimedia Compression Studio** is a desktop application implementing fundamental multimedia compression algorithms from scratch, fulfilling the requirements of the DSP Practical Exam at Suez Canal University. It consists of two independent compression engines with a shared Tkinter GUI:

| Engine               | Algorithm                                     | Standard Resemblance |
| -------------------- | --------------------------------------------- | -------------------- |
| **Audio Compressor** | STFT → Quantization → RLE                     | MP3-like             |
| **Video Compressor** | DCT + Motion Estimation → Huffman → Bitstream | MPEG/H.264-like      |

Both pipelines produce real output files (`.wav`, `.avi`, `.bin`), interactive charts, and measurable compression metrics (SNR, PSNR, Compression Ratio).

---

## ✨ Features

### Audio

- Load real WAV files **or** generate synthetic sine-wave signals
- Additive noise simulation with configurable noise level
- Short-Time Fourier Transform (STFT) for time-frequency analysis
- Configurable quantization levels (4 / 8 / 16 / 32 / 64)
- Run-Length Encoding (RLE) on quantized spectral magnitudes
- Full signal reconstruction via inverse STFT (iSTFT)
- SNR measurement, waveform plots, zoomed difference analysis, and spectrogram display

### Video

- Frame-by-frame video loading with configurable max-frame limit
- YUV color space conversion with **4:2:0 chroma subsampling**
- I-frame (intra) compression: 8×8 DCT → JPEG-standard quantization matrix → zigzag scan → RLE
- P-frame (inter) compression: Three-Step Search motion estimation → residual DCT encoding
- Huffman entropy coding on both intra and inter frame data
- Binary bitstream packaging with frame-type headers
- PSNR per frame, frame-type chart, side-by-side video playback

### GUI

- Tabbed interface (Audio | Video), fully threaded (UI never freezes)
- Live progress bar and scrollable log console
- Embedded Matplotlib charts (Waveforms, Spectrogram, PSNR, Frame Comparison)

---

## 📁 Project Structure

```
multimedia-compression-studio/
│
├── main.py                         # Tkinter GUI entry point
│
├── Audio_Compression/
│   └── audio_encoder.py            # Full audio pipeline
│
├── Video_Compression/
│   └── video_encoder.py            # Full video pipeline
│
└── outputs/                        # Auto-created on first run
    ├── clean_signal.wav
    ├── noisy_signal.wav
    ├── decompressed_audio.wav
    ├── waveforms.png
    ├── zoomed_comparison.png
    ├── spectrogram.png
    ├── video.bin
    ├── reconstructed.avi
    ├── psnr_chart.png
    ├── frame_types.png
    └── frame_comparison.png
```

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────┐
│               main.py  (Tkinter GUI)             │
│  ┌──────────────────┐   ┌──────────────────────┐ │
│  │    AudioTab      │   │      VideoTab        │ │
│  │  (left: config)  │   │  (left: config)      │ │
│  │  (right: charts) │   │  (right: charts)     │ │
│  └────────┬─────────┘   └──────────┬───────────┘ │
│           │ run_async()            │ run_async()  │
└───────────┼────────────────────────┼─────────────┘
            │  daemon thread         │  daemon thread
            ▼                        ▼
┌─────────────────────┐   ┌──────────────────────────┐
│  audio_encoder.py   │   │    video_encoder.py       │
│                     │   │                           │
│  1. Load / Generate │   │  1. Read frames (OpenCV)  │
│  2. Add noise       │   │  2. YUV + 4:2:0 subsample │
│  3. STFT            │   │  3. I/P-frame decision    │
│  4. Quantize        │   │  4. DCT + Quantize        │
│  5. RLE             │   │  5. Motion Estimation     │
│  6. iSTFT → WAV     │   │  6. Huffman Coding        │
│  7. SNR / Plots     │   │  7. Bitstream + PSNR      │
└─────────────────────┘   └──────────────────────────┘
```

---

## 🔊 Audio Compression Pipeline

### Step 1 — Signal Input

Two modes are supported:

**Mode A — Real WAV File**

```
WAV file ──► mono conversion ──► float32 normalization ──► clean_signal
```

**Mode B — Synthetic Signal**

```
Parameters (freq, duration) ──► sine wave generator ──► clean_signal
```

In both modes, Gaussian noise is added and 5 seconds of silence are appended to create the `noisy_signal`.

### Step 2 — STFT (Short-Time Fourier Transform)

```python
f, t, Zxx = signal.stft(noisy_signal, fs, nperseg=1024)
magnitudes = np.abs(Zxx)    # amplitude spectrum
phases     = np.angle(Zxx)  # phase spectrum (preserved for reconstruction)
```

The signal is decomposed into overlapping frames of 1024 samples. Each frame is transformed into the frequency domain, giving a 2D matrix of magnitude and phase.

### Step 3 — Quantization

```
magnitudes ──► uniform scalar quantization ──► quantized
              (configurable levels: 4–64)
```

Quantization reduces the precision of spectral magnitudes, discarding perceptually less-important detail at lower bit depths.

### Step 4 — Run-Length Encoding (RLE)

```
quantized (flattened) ──► RLE pairs ──► [(value, count), ...]
```

Consecutive repeated values (especially zeros from quantization) are compressed into `(value, count)` pairs, dramatically reducing storage for sparse spectra.

### Step 5 — Reconstruction (iSTFT)

```
quantized × exp(j × phases) ──► iSTFT ──► decoded_audio ──► WAV
```

The preserved phase information allows faithful reconstruction of the time-domain signal.

### Metrics

| Metric                | Formula                                 |
| --------------------- | --------------------------------------- |
| **SNR**               | `10 × log₁₀(Σ signal² / Σ noise²)` dB   |
| **Compression Ratio** | `(signal_bits) / (RLE_pairs × 64 bits)` |

---

## 🎥 Video Compression Pipeline

### Step 1 — Frame Input & Color Space

```
BGR frame ──► YUV conversion ──► 4:2:0 chroma subsampling
                                 Y: full resolution
                                 U, V: ½ width × ½ height
```

4:2:0 subsampling exploits the human eye's lower sensitivity to chroma vs. luma, halving chroma data immediately.

### Step 2 — Frame Type Decision

```
Frame index % I_interval == 0 ──► I-frame (intra, self-contained)
                   else        ──► P-frame (predicted from reference)
```

Default: every 10th frame is an I-frame. Configurable via GUI.

### Step 3 — I-Frame Compression

```
channel ──► 8×8 blocks ──► subtract 128 ──► DCT ──► divide by Q ──► round
         ──► zigzag scan ──► RLE ──► Huffman ──► bitstream packet
```

The standard JPEG luminance quantization matrix `Q` is used. DCT concentrates energy into low-frequency coefficients; dividing by `Q` forces high-frequency (visually less important) values to zero.

**Zigzag scan** orders the 8×8 DCT block from low to high frequency, grouping zeros at the end for efficient RLE.

### Step 4 — P-Frame Compression (Three-Step Search)

```
curr_channel ──► Three-Step Search vs. ref_channel
             ──► motion vectors (dy, dx)
             ──► residual = curr − ref_block
             ──► DCT on residual ──► quantize (Q × 1.5)
             ──► RLE ──► Huffman ──► bitstream packet
```

**Three-Step Search** is a fast block-matching algorithm. Starting from a large search window, it narrows the search radius by half each step (3 steps total), drastically reducing motion estimation time vs. full search while maintaining good accuracy.

### Step 5 — Huffman Entropy Coding

```python
# All symbols (RLE pairs, motion vectors, channel IDs) are serialized,
# Huffman-coded, and packed into a binary bitstream with a 9-byte header:
# [4 bytes: frame_index] [1 byte: frame_type] [4 bytes: payload_length]
```

### Step 6 — Multithreaded Processing

All three YUV channels (Y, U, V) are processed concurrently using `concurrent.futures.ThreadPoolExecutor` with 3 workers, reducing encoding time by ~3×.

### Metrics

| Metric                | Formula                                      |
| --------------------- | -------------------------------------------- |
| **PSNR**              | `10 × log₁₀(255² / MSE)` dB                  |
| **Compression Ratio** | `raw_YUV_bytes / compressed_bitstream_bytes` |

---

## 🖥 GUI Application

The application is built with **Tkinter + ttk** and uses `threading` to keep the UI responsive during long encoding jobs.

### Audio Tab

| Panel                      | Contents                                                                                                                                                                           |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Left (Controls)**        | Audio source toggle, file browser, synthetic signal settings (frequency, duration), compression settings (noise level, quantization levels), Run button, progress bar, log console |
| **Waveforms tab**          | 3-panel plot: Clean / Noisy / Decoded signals                                                                                                                                      |
| **Zoomed Differences tab** | 50ms window: Clean, Noisy, Decoded, Noise error, Quantization error                                                                                                                |
| **Spectrogram tab**        | STFT magnitudes vs. Quantized magnitudes (log scale, 0–8 kHz)                                                                                                                      |
| **Statistics tab**         | Source label, SNR (dB), Compression Ratio, RLE pair count                                                                                                                          |

### Video Tab

| Panel                    | Contents                                                                                                                                  |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Left (Controls)**      | Video file picker, max frames, I-frame interval, block size, search area, Run button, Play Before/After button, progress bar, log console |
| **PSNR tab**             | Bar + line chart of PSNR per frame, colored by I/P type                                                                                   |
| **Frame Types tab**      | Step chart of I/P frame sequence                                                                                                          |
| **Frame Comparison tab** | Grid: Original vs. Reconstructed frames                                                                                                   |
| **Statistics tab**       | Average PSNR, Compression Ratio, Total Frames, Bitstream path                                                                             |

---

## ⚙ Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Install Dependencies

```bash
pip install numpy scipy matplotlib opencv-python
```

### Clone & Run

```bash
git clone https://github.com/basem3sam/multimedia-compression-studio.git
cd multimedia-compression-studio
python main.py
```

> **Note:** The `outputs/` directory is created automatically on first run.

### Dependency Reference

| Package         | Version | Purpose                       |
| --------------- | ------- | ----------------------------- |
| `numpy`         | ≥ 1.21  | Array math, signal buffers    |
| `scipy`         | ≥ 1.7   | STFT / iSTFT, WAV I/O         |
| `matplotlib`    | ≥ 3.4   | All charts and plots          |
| `opencv-python` | ≥ 4.5   | DCT, video I/O, frame display |

---

## 🚀 Usage

### Audio Compression

1. Launch the app: `python main.py`
2. Click the **Audio** tab.
3. **Option A — Use your own WAV file:**
   - Check **"Use my own audio file (WAV)"**
   - Click **Browse…** and select a `.wav` file
4. **Option B — Synthetic signal:**
   - Set **Frequency (Hz)** and **Duration (s)**
5. Adjust **Noise Level** and **Quantization Levels**
6. Click **▶ Run Audio Compression**
7. View results across the four chart tabs.

### Video Compression

1. Click the **Video** tab.
2. Click **Select Video** and choose an `.mp4`, `.avi`, or `.mov` file.
3. Configure:
   - **Max Frames** — how many frames to encode
   - **I-Frame Interval** — e.g., `10` means every 10th frame is an I-frame
   - **Block Size** — `8`, `16`, or `32` pixels
   - **Search Area** — motion search window radius
4. Click **Run Video Compression**
5. After completion, click **▶ Play Before/After Video** for a side-by-side comparison.

---

## 📂 Output Files

| File                             | Description                            |
| -------------------------------- | -------------------------------------- |
| `outputs/clean_signal.wav`       | Original signal (without noise)        |
| `outputs/noisy_signal.wav`       | Noisy signal fed into the encoder      |
| `outputs/decompressed_audio.wav` | Reconstructed audio after compression  |
| `outputs/waveforms.png`          | 3-panel waveform comparison            |
| `outputs/zoomed_comparison.png`  | 50ms zoomed window + error signals     |
| `outputs/spectrogram.png`        | STFT magnitude vs. quantized magnitude |
| `outputs/video.bin`              | Compressed binary bitstream            |
| `outputs/reconstructed.avi`      | Decoded video output                   |
| `outputs/psnr_chart.png`         | PSNR per frame (bar + line chart)      |
| `outputs/frame_types.png`        | I/P frame sequence chart               |
| `outputs/frame_comparison.png`   | Original vs. reconstructed frame grid  |

---

## 🔧 Technical Details

### Audio — Quantization Formula

```
step     = (max_val - min_val) / levels
quantized = round((x - min_val) / step) × step + min_val
```

### Video — JPEG Luminance Quantization Matrix

```
Q = [[16,11,10,16,24,40,51,61],
     [12,12,14,19,26,58,60,55],
     [14,13,16,24,40,57,69,56],
     [14,17,22,29,51,87,80,62],
     [18,22,37,56,68,109,103,77],
     [24,35,55,64,81,104,113,92],
     [49,64,78,87,103,121,120,101],
     [72,92,95,98,112,100,103,99]]
```

Residual blocks (P-frames) use `Q × 1.5` — a softer quantizer because residuals have lower energy.

### Bitstream Packet Format

```
┌──────────────┬────────────┬──────────────────┬────────────────────────┐
│ Frame Index  │ Frame Type │  Payload Length  │  Huffman-coded payload │
│   4 bytes    │   1 byte   │    4 bytes       │      N bytes           │
└──────────────┴────────────┴──────────────────┴────────────────────────┘
```

Frame Type: `0x00` = I-frame, `0x01` = P-frame.

---

## 👥 Team

**Suez Canal University — Faculty of Computers and Informatics**
**Computer Science Department — DSP Practical Exam**

| #   | Name                            |
| --- | ------------------------------- |
| 1   | **Basem Esam Omar Gamal Azoum** |
| 2   | Ahmed Adallah Al-Sayed          |
| 3   | Ahmed Ali Al-Sayed              |
| 4   | Ahmed Al-Sayed                  |
| 5   | Ahmed Elsayed Abdo Ali Farag    |
| 6   | Khaled Mohammed Ghoneim         |
| 7   | Mohamed Adel Abdo Haggag        |

---

## 📚 References

- Sayood, K. — _Introduction to Data Compression_, 5th Ed., Morgan Kaufmann
- Richardson, I. — _H.264 and MPEG-4 Video Compression_, Wiley
- Oppenheim & Schafer — _Discrete-Time Signal Processing_, Prentice Hall
- JPEG Standard — ISO/IEC 10918-1
- SciPy Documentation — [`scipy.signal.stft`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html)
- OpenCV Documentation — [`cv2.dct`](https://docs.opencv.org/4.x/d2/de8/group__core__array.html)

---

<div align="center">
Made with ❤️ for the DSP Practical Exam · Suez Canal University · 2025
</div>
