import os
import threading
import cv2

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from Audio_Compression.audio_encoder import run_pipeline as run_audio
from Video_Compression.video_encoder import run_pipeline as run_video


# ───────────────────────────────────────────
# THREADING
# ───────────────────────────────────────────
def run_async(func, params, log, progress, done, root):
    threading.Thread(
        target=func,
        args=(
            params,
            lambda m: root.after(0, log, m),
            lambda v: root.after(0, progress, v),
            lambda *d: root.after(0, done, *d),
        ),
        daemon=True,
    ).start()


# ───────────────────────────────────────────
# MATPLOTLIB DISPLAY
# ───────────────────────────────────────────
def draw_figure(container, fig):
    for widget in container.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, container)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


# ───────────────────────────────────────────
# AUDIO TAB
# ───────────────────────────────────────────
class AudioTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.build_ui()

    def build_ui(self):
        left  = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=10, pady=10)

        right = ttk.Notebook(self)
        right.pack(side="right", fill="both", expand=True)

        self.wave_tab   = ttk.Frame(right)
        self.zoom_tab   = ttk.Frame(right)
        self.spec_tab   = ttk.Frame(right)
        self.stats_tab  = ttk.Frame(right)

        right.add(self.wave_tab,  text="Waveforms")
        right.add(self.zoom_tab,  text="🔍 Zoomed Differences")
        right.add(self.spec_tab,  text="Spectrogram")
        right.add(self.stats_tab, text="Statistics")

        # ── Audio Source ────────────────────────────
        ttk.Label(left, text="── Audio Source ──", font=("Arial", 9, "bold")).pack(anchor="w", pady=(0, 4))

        self.use_file = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            left,
            text="Use my own audio file (WAV)",
            variable=self.use_file,
            command=self._toggle_source,
        ).pack(anchor="w")

        # File picker (hidden by default)
        self.file_frame = ttk.Frame(left)
        self.file_frame.pack(fill="x", pady=2)

        self.audio_file = tk.StringVar()
        self.file_entry = ttk.Entry(self.file_frame, textvariable=self.audio_file, state="disabled", width=22)
        self.file_entry.pack(side="left", fill="x", expand=True)
        self.browse_btn = ttk.Button(self.file_frame, text="Browse…", command=self._pick_file, state="disabled")
        self.browse_btn.pack(side="left", padx=(4, 0))

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)

        # ── Synthetic signal settings ────────────────
        ttk.Label(left, text="── Synthetic Signal Settings ──", font=("Arial", 9, "bold")).pack(anchor="w")
        ttk.Label(left, text="(used only when no file is loaded)", foreground="gray").pack(anchor="w")

        self.frequency = tk.DoubleVar(value=440)
        self.duration  = tk.DoubleVar(value=1.0)

        ttk.Label(left, text="Frequency (Hz)").pack(anchor="w", pady=(6, 0))
        self.freq_spin = ttk.Spinbox(left, from_=100, to=5000, textvariable=self.frequency)
        self.freq_spin.pack(fill="x")

        ttk.Label(left, text="Duration (s)").pack(anchor="w", pady=(6, 0))
        self.dur_spin = ttk.Spinbox(left, from_=0.5, to=10, increment=0.5, textvariable=self.duration)
        self.dur_spin.pack(fill="x")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)

        # ── Compression settings ─────────────────────
        ttk.Label(left, text="── Compression Settings ──", font=("Arial", 9, "bold")).pack(anchor="w")

        self.noise = tk.DoubleVar(value=0.05)
        self.bits  = tk.IntVar(value=16)

        ttk.Label(left, text="Noise Level (added on top)").pack(anchor="w", pady=(6, 0))
        ttk.Spinbox(left, from_=0, to=1, increment=0.05, textvariable=self.noise).pack(fill="x")

        ttk.Label(left, text="Quantization Levels").pack(anchor="w", pady=(6, 0))
        ttk.Combobox(left, values=[4, 8, 16, 32, 64], textvariable=self.bits, state="readonly").pack(fill="x")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)

        ttk.Button(left, text="▶  Run Audio Compression", command=self.run).pack(fill="x", pady=4)

        self.progress = ttk.Progressbar(left, maximum=100)
        self.progress.pack(fill="x")

        self.log_box = tk.Text(left, height=12, width=35, font=("Courier", 9))
        self.log_box.pack(fill="both", expand=True, pady=10)

    # ── helpers ─────────────────────────────────────
    def _toggle_source(self):
        state = "normal" if self.use_file.get() else "disabled"
        self.file_entry.config(state=state)
        self.browse_btn.config(state=state)
        # Dim synthetic controls when file mode is on
        dim = "disabled" if self.use_file.get() else "normal"
        self.freq_spin.config(state=dim)
        self.dur_spin.config(state=dim)

    def _pick_file(self):
        path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.wave *.mp3"), ("WAV", "*.wav"), ("MP3", "*.mp3")],
        )
        if path:
            self.audio_file.set(path)
            self.log_message(f"Loaded: {os.path.basename(path)}")

    def log_message(self, message):
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")

    def update_progress(self, value):
        self.progress["value"] = value

    def run(self):
        audio_file = ""
        if self.use_file.get():
            audio_file = self.audio_file.get().strip()
            if not audio_file or not os.path.exists(audio_file):
                messagebox.showerror("File Not Found", "Please select a valid WAV or MP3 file.")
                return

        params = {
            "audio_file": audio_file,          # ← NEW
            "frequency":  self.frequency.get(),
            "noise":      self.noise.get(),
            "duration":   self.duration.get(),
            "q_levels":   self.bits.get(),
            "output_dir": "outputs",
        }
        run_async(
            run_audio, params,
            self.log_message, self.update_progress, self.finished,
            self.winfo_toplevel(),
        )

    def finished(self, data):
        # ── Waveforms ────────────────────────────────
        fig1, axs = plt.subplots(3, 1, figsize=(8, 6))
        axs[0].plot(data["time_axis"], data["clean_signal"],   color="#2196F3")
        axs[0].set_title(f"Clean Signal  [{data.get('source_label', '')}]")
        axs[1].plot(data["time_axis"], data["noisy_signal"],   color="#F44336")
        axs[1].set_title("Noisy Signal (with silence appended)")
        axs[2].plot(data["time_axis"], data["decoded_signal"], color="#4CAF50")
        axs[2].set_title("Decoded (Decompressed) Signal")
        for ax in axs:
            ax.set_ylabel("Amplitude")
        axs[2].set_xlabel("Time (s)")
        fig1.tight_layout()
        draw_figure(self.wave_tab, fig1)

        # ── Zoomed differences ────────────────────────
        zoom_path = os.path.join("outputs", "zoomed_comparison.png")
        if os.path.exists(zoom_path):
            img = plt.imread(zoom_path)
            fig_z, ax_z = plt.subplots(figsize=(10, 8))
            ax_z.imshow(img)
            ax_z.axis("off")
            draw_figure(self.zoom_tab, fig_z)

        # ── Spectrogram ───────────────────────────────
        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
        freq_ticks  = data.get("freq_ticks", [])
        freq_labels = data.get("freq_labels", [])
        max_hz      = data.get("max_hz", 8000)

        axes2[0].imshow(data["magnitudes"], aspect="auto", origin="lower", cmap="inferno")
        axes2[0].set_title(f"STFT Magnitudes (0–{max_hz}Hz, log scale)")
        axes2[0].set_xlabel("Time frame")
        axes2[0].set_ylabel("Frequency (Hz)")
        if len(freq_ticks):
            axes2[0].set_yticks(freq_ticks)
            axes2[0].set_yticklabels(freq_labels)

        axes2[1].imshow(data["quantized"], aspect="auto", origin="lower", cmap="inferno")
        axes2[1].set_title("Quantized Magnitudes")
        axes2[1].set_xlabel("Time frame")
        axes2[1].set_ylabel("Frequency (Hz)")
        if len(freq_ticks):
            axes2[1].set_yticks(freq_ticks)
            axes2[1].set_yticklabels(freq_labels)

        fig2.tight_layout()
        draw_figure(self.spec_tab, fig2)

        # ── Statistics ────────────────────────────────
        for widget in self.stats_tab.winfo_children():
            widget.destroy()
        for s in [
            f"Source: {data.get('source_label', 'N/A')}",
            f"SNR: {data['snr']} dB",
            f"Compression Ratio: {data['compression_ratio']} : 1",
            f"RLE Pairs: {data['rle_pairs']}",
        ]:
            ttk.Label(self.stats_tab, text=s, font=("Arial", 12)).pack(anchor="w", padx=10, pady=5)


# ───────────────────────────────────────────
# VIDEO TAB  (unchanged)
# ───────────────────────────────────────────
class VideoTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.recon_video_path = None
        self.build_ui()

    def build_ui(self):
        left  = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=10, pady=10)

        right = ttk.Notebook(self)
        right.pack(side="right", fill="both", expand=True)

        self.psnr_tab    = ttk.Frame(right)
        self.frame_tab   = ttk.Frame(right)
        self.compare_tab = ttk.Frame(right)
        self.stats_tab   = ttk.Frame(right)

        right.add(self.psnr_tab,    text="PSNR")
        right.add(self.frame_tab,   text="Frame Types")
        right.add(self.compare_tab, text="Frame Comparison")
        right.add(self.stats_tab,   text="Statistics")

        self.video_path  = tk.StringVar()
        self.max_frames  = tk.IntVar(value=30)
        self.i_interval  = tk.IntVar(value=10)
        self.block_size  = tk.IntVar(value=16)
        self.search_area = tk.IntVar(value=8)

        ttk.Button(left, text="Select Video", command=self.pick_video).pack(fill="x")

        ttk.Label(left, text="Max Frames").pack(anchor="w")
        ttk.Spinbox(left, from_=1, to=500, textvariable=self.max_frames).pack(fill="x")

        ttk.Label(left, text="I-Frame Interval").pack(anchor="w")
        ttk.Spinbox(left, from_=1, to=60, textvariable=self.i_interval).pack(fill="x")

        ttk.Label(left, text="Block Size").pack(anchor="w")
        ttk.Combobox(left, values=[8, 16, 32], textvariable=self.block_size, state="readonly").pack(fill="x")

        ttk.Label(left, text="Search Area").pack(anchor="w")
        ttk.Spinbox(left, from_=2, to=32, textvariable=self.search_area).pack(fill="x")

        ttk.Button(left, text="Run Video Compression", command=self.run).pack(fill="x", pady=10)
        ttk.Button(left, text="▶ Play Before/After Video", command=self.play_comparison).pack(fill="x", pady=5)

        self.progress = ttk.Progressbar(left, maximum=100)
        self.progress.pack(fill="x", pady=5)

        self.log = tk.Text(left, height=12, width=35)
        self.log.pack(fill="both", expand=True, pady=10)

    def pick_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if path:
            self.video_path.set(path)
            self.log_message(f"Loaded: {os.path.basename(path)}")

    def log_message(self, message):
        self.log.insert("end", message + "\n")
        self.log.see("end")

    def update_progress(self, value):
        self.progress["value"] = value

    def run(self):
        if not os.path.exists(self.video_path.get()):
            self.log_message("Please select a valid video.")
            return
        params = {
            "video_path":  self.video_path.get(),
            "max_frames":  self.max_frames.get(),
            "i_interval":  self.i_interval.get(),
            "block_size":  self.block_size.get(),
            "search_area": self.search_area.get(),
            "output_dir":  "outputs",
        }
        run_async(run_video, params, self.log_message, self.update_progress, self.finished, self.winfo_toplevel())

    def play_comparison(self):
        orig_path = self.video_path.get()
        if not orig_path or not self.recon_video_path or not os.path.exists(self.recon_video_path):
            self.log_message("No reconstructed video available. Run compression first.")
            return
        threading.Thread(target=self._play_side_by_side, args=(orig_path, self.recon_video_path), daemon=True).start()

    def _play_side_by_side(self, orig_path, recon_path):
        cap_orig  = cv2.VideoCapture(orig_path)
        cap_recon = cv2.VideoCapture(recon_path)
        while True:
            ret1, frame1 = cap_orig.read()
            ret2, frame2 = cap_recon.read()
            if not ret1 or not ret2:
                break
            h1, w1 = frame1.shape[:2]
            h2, w2 = frame2.shape[:2]
            if h1 != h2 or w1 != w2:
                frame1 = cv2.resize(frame1, (w2, h2))
            cv2.putText(frame1, "Original",                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame2, "Reconstructed (Full YUV)",   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            combined = np.hstack((frame1, frame2))
            cv2.imshow("Compression Comparison - Press 'Q' to Exit", combined)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cap_orig.release()
        cap_recon.release()
        cv2.destroyWindow("Compression Comparison - Press 'Q' to Exit")

    def finished(self, data, summary):
        self.recon_video_path = data.get("recon_video_path")

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(data["psnr_values"])
        ax1.set_title("PSNR Per Frame")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("PSNR (dB)")
        draw_figure(self.psnr_tab, fig1)

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        frame_values = [1 if f == "I" else 0 for f in data["frame_types"]]
        ax2.step(range(len(frame_values)), frame_values, where="mid")
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["P", "I"])
        ax2.set_title("Frame Types")
        draw_figure(self.frame_tab, fig2)

        pairs = data["frame_pairs"]
        n     = len(pairs)
        if n > 0:
            fig3, axes = plt.subplots(2, n, figsize=(3 * n, 6))
            if n == 1:
                axes = np.expand_dims(axes, axis=1)
            for col, (orig, recon) in enumerate(pairs):
                axes[0, col].imshow(orig)
                axes[0, col].set_title(f"Original #{col}", fontsize=8)
                axes[0, col].axis("off")
                axes[1, col].imshow(recon)
                axes[1, col].set_title(f"Recon #{col}", fontsize=8)
                axes[1, col].axis("off")
            fig3.suptitle("Original vs Reconstructed Frames")
            fig3.tight_layout()
            draw_figure(self.compare_tab, fig3)

        for widget in self.stats_tab.winfo_children():
            widget.destroy()
        for s in [
            f"Average PSNR: {summary['avg_psnr']:.2f} dB",
            f"Compression Ratio: {summary['ratio']} : 1",
            f"Total Frames: {summary['total_frames']}",
            f"Bitstream: {data['bitstream_path']}",
        ]:
            ttk.Label(self.stats_tab, text=s, font=("Arial", 12)).pack(anchor="w", padx=10, pady=5)


# ───────────────────────────────────────────
# MAIN APP
# ───────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Multimedia Compression Studio")
        self.geometry("1200x700")

        style = ttk.Style()
        style.theme_use("clam")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        notebook.add(AudioTab(notebook), text="Audio")
        notebook.add(VideoTab(notebook), text="Video")


if __name__ == "__main__":
    app = App()
    app.mainloop()