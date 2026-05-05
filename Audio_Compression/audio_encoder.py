import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile


# ───────────────────────────────────────────
# LOAD AUDIO FILE (WAV or MP3)
# ───────────────────────────────────────────
def load_audio_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".wav", ".wave"):
        fs, data = wavfile.read(file_path)
        # Convert to mono if stereo
        if data.ndim == 2:
            data = data.mean(axis=1)
        # Normalize to float32 [-1, 1]
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128) / 128.0
        else:
            data = data.astype(np.float32)
        return fs, data

    else:
        raise ValueError(f"Unsupported audio format: {ext}. Use WAV.")


# ───────────────────────────────────────────
# QUANTIZATION
# ───────────────────────────────────────────
def quantize(data, levels):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return data.copy()
    step      = (max_val - min_val) / levels
    quantized = np.round((data - min_val) / step) * step + min_val
    return quantized


# ───────────────────────────────────────────
# RUN LENGTH ENCODING
# ───────────────────────────────────────────
def run_length_encode(data):
    flat    = data.flatten()
    encoded = []
    count   = 1
    for i in range(1, len(flat)):
        if flat[i] == flat[i - 1]:
            count += 1
        else:
            encoded.append((flat[i - 1], count))
            count = 1
    encoded.append((flat[-1], count))
    return encoded


# ───────────────────────────────────────────
# AUDIO PIPELINE
# ───────────────────────────────────────────
def run_pipeline(params, log, progress, done):
    output_dir = params.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    audio_file  = params.get("audio_file", "")    # ← NEW: path to WAV/MP3
    frequency   = params.get("frequency", 440)
    noise_level = params.get("noise", 0.5)
    duration    = params.get("duration", 1.0)
    q_levels    = params.get("q_levels", 16)

    # ───────────────────────────────────────
    # STEP 1 → SIGNAL GENERATION / LOADING
    # ───────────────────────────────────────
    if audio_file and os.path.exists(audio_file):
        log(f"Loading audio file: {os.path.basename(audio_file)}")
        progress(5)
        try:
            fs, raw_signal = load_audio_file(audio_file)
            log(f"Loaded: {len(raw_signal)/fs:.2f}s @ {fs}Hz")

            # Use the loaded signal as the "clean" signal
            clean_signal = raw_signal

            # Add noise scaled to signal amplitude so it's always visible
            sig_std = np.std(clean_signal) if np.std(clean_signal) > 0 else 1.0
            noise   = np.random.normal(0, noise_level * sig_std, clean_signal.shape)
            silence = np.zeros(fs // 2)   # 0.5s silence appended

            noisy_signal       = np.concatenate((clean_signal + noise, silence))
            clean_with_silence = np.concatenate((clean_signal, silence))
            time_axis          = np.linspace(
                0, len(noisy_signal) / fs, len(noisy_signal)
            )
            source_label = os.path.basename(audio_file)

        except Exception as e:
            log(f"ERROR loading file: {e}")
            log("Falling back to synthetic signal...")
            audio_file = ""   # trigger fallback below

    if not audio_file or not os.path.exists(audio_file):
        # ── Synthetic fallback ──────────────────
        log("Generating synthetic audio signal...")
        progress(5)
        fs = 44100
        t            = np.linspace(0, duration, int(fs * duration))
        clean_signal = np.sin(2 * np.pi * frequency * t)
        noise        = np.random.normal(0, noise_level, clean_signal.shape)
        silence      = np.zeros(fs * 5)   # 5s silence appended

        noisy_signal       = np.concatenate((clean_signal + noise, silence))
        clean_with_silence = np.concatenate((clean_signal, silence))
        time_axis          = np.linspace(0, duration + 5, len(noisy_signal))
        source_label       = f"Synthetic {frequency}Hz"

    progress(10)

    # ── helper: float32 [-1,1] → int16 for universal WAV compatibility ──
    def to_int16(arr):
        arr = np.clip(arr, -1.0, 1.0)
        return (arr * 32767).astype(np.int16)

    # Save WAVs as int16 — float32 WAV confuses many media players
    clean_wav_path = os.path.join(output_dir, "clean_signal.wav")
    noisy_wav_path = os.path.join(output_dir, "noisy_signal.wav")
    wavfile.write(clean_wav_path, fs, to_int16(clean_with_silence))
    wavfile.write(noisy_wav_path, fs, to_int16(noisy_signal))
    log("Saved: clean_signal.wav")
    log("Saved: noisy_signal.wav")

    # ───────────────────────────────────────
    # STEP 2 → STFT
    # ───────────────────────────────────────
    log("Applying STFT...")
    progress(30)

    f, t_stft, Zxx = signal.stft(noisy_signal, fs, nperseg=1024)
    magnitudes      = np.abs(Zxx)
    phases          = np.angle(Zxx)

    # ───────────────────────────────────────
    # STEP 3 → QUANTIZATION
    # ───────────────────────────────────────
    log("Quantizing frequency magnitudes...")
    progress(50)

    quantized = quantize(magnitudes, q_levels)

    # ───────────────────────────────────────
    # STEP 4 → RUN-LENGTH ENCODING
    # ───────────────────────────────────────
    log("Applying Run-Length Encoding...")
    progress(70)

    compressed = run_length_encode(quantized)

    # ───────────────────────────────────────
    # STEP 5 → RECONSTRUCTION
    # ───────────────────────────────────────
    log("Reconstructing audio...")
    progress(85)

    reconstructed_Zxx = quantized * np.exp(1j * phases)
    _, decoded_audio  = signal.istft(reconstructed_Zxx, fs)
    decoded_audio     = decoded_audio[:len(noisy_signal)]

    wav_path = os.path.join(output_dir, "decompressed_audio.wav")
    wavfile.write(wav_path, fs, to_int16(decoded_audio))
    log("Saved: decompressed_audio.wav")

    # ───────────────────────────────────────
    # METRICS
    # ───────────────────────────────────────
    signal_power = np.sum(noisy_signal ** 2)
    noise_power  = np.sum((noisy_signal - decoded_audio) ** 2)

    if noise_power == 0:
        snr = "Infinite"
    else:
        snr = round(10 * np.log10(signal_power / noise_power), 2)

    original_size     = noisy_signal.size * 32
    compressed_size   = len(compressed) * 64
    compression_ratio = round(original_size / compressed_size, 2)

    # ───────────────────────────────────────
    # SAVE WAVEFORM PLOT
    # ───────────────────────────────────────
    log("Saving waveform plot...")
    fig1, axs = plt.subplots(3, 1, figsize=(10, 7))
    axs[0].plot(time_axis, clean_with_silence, color="#2196F3")
    axs[0].set_title(f"Clean Signal  [{source_label}]")
    axs[0].set_ylabel("Amplitude")
    axs[1].plot(time_axis, noisy_signal, color="#F44336")
    axs[1].set_title("Noisy Signal (with silence appended)")
    axs[1].set_ylabel("Amplitude")
    axs[2].plot(time_axis, decoded_audio, color="#4CAF50")
    axs[2].set_title("Decoded (Decompressed) Signal")
    axs[2].set_ylabel("Amplitude")
    axs[2].set_xlabel("Time (s)")
    fig1.suptitle(f"Audio Waveforms — source={source_label}, q_levels={q_levels}")
    fig1.tight_layout()
    waveform_path = os.path.join(output_dir, "waveforms.png")
    fig1.savefig(waveform_path, dpi=150)
    plt.close(fig1)
    log("Saved: waveforms.png")

    # ───────────────────────────────────────
    # SAVE ZOOMED DIFFERENCE PLOT
    # ───────────────────────────────────────
    log("Saving zoomed comparison plot...")

    # Pick a 0.05s window from the middle of the signal (where voice is active)
    mid        = len(noisy_signal) // 3          # avoid silence at end
    zoom_len   = int(fs * 0.05)                  # 50ms window
    start      = max(0, mid - zoom_len // 2)
    end        = start + zoom_len
    t_zoom     = time_axis[start:end]

    clean_zoom    = clean_with_silence[start:end]
    noisy_zoom    = noisy_signal[start:end]
    decoded_zoom  = decoded_audio[start:end]
    diff_noise    = noisy_zoom - clean_zoom
    diff_compress = decoded_zoom - clean_zoom

    fig3, axs3 = plt.subplots(5, 1, figsize=(12, 10))

    axs3[0].plot(t_zoom, clean_zoom,    color="#2196F3", linewidth=1.5)
    axs3[0].set_title("Clean Signal (zoomed 50ms window)")
    axs3[0].set_ylabel("Amplitude")

    axs3[1].plot(t_zoom, noisy_zoom,    color="#F44336", linewidth=1.5)
    axs3[1].set_title("Noisy Signal (zoomed)")
    axs3[1].set_ylabel("Amplitude")

    axs3[2].plot(t_zoom, decoded_zoom,  color="#4CAF50", linewidth=1.5)
    axs3[2].set_title("Compressed/Decoded Signal (zoomed)")
    axs3[2].set_ylabel("Amplitude")

    axs3[3].plot(t_zoom, diff_noise,    color="#FF9800", linewidth=1.2)
    axs3[3].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axs3[3].set_title("Difference: Noisy − Clean  (this is the noise you added)")
    axs3[3].set_ylabel("Amplitude")

    axs3[4].plot(t_zoom, diff_compress, color="#9C27B0", linewidth=1.2)
    axs3[4].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axs3[4].set_title("Difference: Compressed − Clean  (quantization error)")
    axs3[4].set_ylabel("Amplitude")
    axs3[4].set_xlabel("Time (s)")

    fig3.suptitle(f"Zoomed Comparison — q_levels={q_levels}  |  SNR={snr} dB", fontsize=13)
    fig3.tight_layout()
    zoomed_path = os.path.join(output_dir, "zoomed_comparison.png")
    fig3.savefig(zoomed_path, dpi=150)
    plt.close(fig3)
    log("Saved: zoomed_comparison.png")

    # ───────────────────────────────────────
    # SAVE SPECTROGRAM PLOT
    # ───────────────────────────────────────
    log("Saving spectrogram...")

    # Only show frequencies up to 8000 Hz — human voice lives in 80–4000 Hz
    # magnitudes shape: (freq_bins, time_frames)
    freq_bins   = magnitudes.shape[0]
    max_hz      = 8000
    bin_per_hz  = freq_bins / (fs / 2)
    cutoff_bin  = int(max_hz * bin_per_hz)
    mag_crop    = magnitudes[:cutoff_bin, :]
    quant_crop  = quantized[:cutoff_bin, :]

    # Use log scale so quiet parts are visible (add small epsilon to avoid log(0))
    mag_log   = np.log1p(mag_crop)
    quant_log = np.log1p(quant_crop)

    # Frequency labels for Y axis
    freq_labels = np.linspace(0, max_hz, num=6, dtype=int)
    freq_ticks  = [int(hz * bin_per_hz) for hz in freq_labels]

    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(mag_log,   aspect="auto", origin="lower", cmap="inferno")
    axes[0].set_title("STFT Magnitudes — Original (log scale, 0–8kHz)")
    axes[0].set_xlabel("Time frame")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_yticks(freq_ticks)
    axes[0].set_yticklabels(freq_labels)
    fig2.colorbar(im0, ax=axes[0], label="log magnitude")

    im1 = axes[1].imshow(quant_log, aspect="auto", origin="lower", cmap="inferno")
    axes[1].set_title(f"Quantized Magnitudes ({q_levels} levels, log scale)")
    axes[1].set_xlabel("Time frame")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_yticks(freq_ticks)
    axes[1].set_yticklabels(freq_labels)
    fig2.colorbar(im1, ax=axes[1], label="log magnitude")

    fig2.tight_layout()
    spectrogram_path = os.path.join(output_dir, "spectrogram.png")
    fig2.savefig(spectrogram_path, dpi=150)
    plt.close(fig2)
    log("Saved: spectrogram.png")

    progress(100)
    log("Audio compression completed.")
    log(f"All outputs saved to: {output_dir}/")

    data = {
        "clean_signal":      clean_with_silence,
        "noisy_signal":      noisy_signal,
        "decoded_signal":    decoded_audio,
        "time_axis":         time_axis,
        "magnitudes":        mag_log,        # cropped + log scaled for display
        "quantized":         quant_log,      # cropped + log scaled for display
        "freq_labels":       freq_labels,
        "freq_ticks":        freq_ticks,
        "max_hz":            max_hz,
        "snr":               snr,
        "compression_ratio": compression_ratio,
        "rle_pairs":         len(compressed),
        "wav_path":          wav_path,
        "source_label":      source_label,
    }

    done(data)
