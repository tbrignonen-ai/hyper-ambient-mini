#!/usr/bin/env python3
"""hyperambient-mini — light dictation harness (the "ears" of hyper-ambient only).

Mic -> energy VAD -> parakeet-server (fluid-speech-mini GGUF, CPU) -> live text
-> copy to clipboard, so you can paste into ZCode / the DeepSeek desktop app.

Stdlib + sounddevice only. The STT server is a separate process (start.bat);
this app never touches the GPU, so a running training job is left alone.
"""
from __future__ import annotations

import array
import io
import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
import wave

import sounddevice as sd
import tkinter as tk
from tkinter import ttk

# ---------------------------------------------------------------- config ---
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SERVER_URL = os.environ.get("PARK_SERVER_URL", "http://127.0.0.1:8077")
TRANSCRIPT_FILE = os.path.join(ROOT, "transcript.txt")

SAMPLE_RATE = 16000          # what the model wants
BLOCK_MS = 30                # audio callback granularity
BLOCKSIZE = SAMPLE_RATE * BLOCK_MS // 1000

# --- VAD (energy-first, hyper-ambient style: spare the inference) ---------
SILENCE_TO_END_S = 0.75      # this much trailing quiet ends an utterance
MIN_SPEECH_S = 0.30          # shorter bursts are ignored (clicks, bumps)
MAX_UTTERANCE_S = 25.0       # hard cap, force-flush mid-speech
ADAPT_SECONDS = 2.0          # noise-floor learning window at session start
AGC_FLOOR = 350.0            # absolute RMS below this is never speech (int16)

# Whisper-style recited credits / empty output: drop rather than display.
JUNK_PATTERNS = (
    "subtitles by", "sous titrage", "amara org", "thanks for watching",
    "abonnez vous", "merci d avoir regarde", "subtitle", "www",
)


# ------------------------------------------------------------ audio utils ---
def rms_db_of(raw: bytes) -> float:
    """RMS of an int16 frame, in dBFS-ish arbitrary units. array module: no numpy."""
    samples = array.array("h")
    samples.frombytes(bytes(raw))
    if not samples:
        return 0.0
    acc = 0
    for s in samples:
        acc += s * s
    return (acc / len(samples)) ** 0.5


def resample_to_16k(raw: bytes, src_rate: int) -> bytes:
    """Linear resample int16 mono to 16 kHz (only used if the mic refuses 16k)."""
    if src_rate == SAMPLE_RATE:
        return raw
    in_s = array.array("h")
    in_s.frombytes(bytes(raw))
    step = src_rate / SAMPLE_RATE
    n_out = int(len(in_s) / step)
    out = array.array("h", bytes(2 * n_out))
    for i in range(n_out):
        pos = i * step
        i0 = int(pos)
        frac = pos - i0
        a = in_s[i0] if i0 < len(in_s) else 0
        b = in_s[i0 + 1] if i0 + 1 < len(in_s) else a
        out[i] = int((1 - frac) * a + frac * b)
    return out.tobytes()


def wav_bytes(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


# ------------------------------------------------------------ stt client ---
class ParakeetClient:
    """Minimal OpenAI-compatible /v1/audio/transcriptions client (stdlib)."""

    def __init__(self, url: str, timeout: float = 60.0):
        self.url = url.rstrip("/") + "/v1/audio/transcriptions"
        self.timeout = timeout

    def transcribe(self, pcm: bytes) -> tuple[str, float]:
        boundary = "----hyperambientmini"
        body = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="u.wav"\r\nContent-Type: audio/wav\r\n\r\n'
        ).encode() + wav_bytes(pcm) + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            self.url, data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            text = json.loads(r.read().decode("utf-8")).get("text", "")
        return text.strip(), time.time() - t0

    def ping(self) -> bool:
        try:
            urllib.request.urlopen(self.url, timeout=2)  # expect 4xx: it's alive
            return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            return False


def is_junk(text: str) -> bool:
    t = " ".join(text.lower().split())
    if len(t) < 2:
        return True
    if any(p in t for p in JUNK_PATTERNS):
        return True
    # Repeated-token hallucination ("no, no, no", "let's, let's, let's"):
    # same 1-3 word group repeated 3+ times is never real dictation.
    words = [w.strip(".,!?") for w in t.split() if w.strip(".,!?")]
    for n in (1, 2, 3):
        if len(words) >= n * 3:
            groups = [tuple(words[i:i + n]) for i in range(0, len(words) - n + 1, n)]
            if len(groups) >= 3 and len(set(groups)) == 1:
                return True
    return False


# --------------------------------------------------------------- the app ---
class MiniHarness:
    def __init__(self) -> None:
        self.client = ParakeetClient(SERVER_URL)
        self.recording = False          # mic stream open
        self.listening = threading.Event()  # paused <=> cleared
        self.segment_q: "queue.Queue[bytes]" = queue.Queue()
        self.result_q: "queue.Queue[tuple]" = queue.Queue()
        self.frames: list[bytes] = []   # current utterance buffer
        self.in_speech = False
        self.silence_run = 0.0
        self.speech_run = 0.0
        self.utter_len = 0.0
        self.noise_floor = 120.0
        self.warmup_blocks = 0
        self.copied_upto = "1.0"        # text index: everything before is copied
        self.stats = {"n": 0, "ms": 0.0}
        self.server_ok: bool | None = None
        self._build_ui()
        self._start_worker()
        self._start_mic()
        threading.Thread(target=self._probe_server, daemon=True).start()

    def _probe_server(self) -> None:
        self.server_ok = self.client.ping()
        self.root.after(0, self._set_status)

    # ---------------- UI ----------------
    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("hyperambient-mini")
        self.root.geometry("560x300")
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#14171c")
        try:  # nice-ish dark text area
            self.root.tk_setPalette(background="#14171c", foreground="#e8e8e8")
        except tk.TclError:
            pass

        self.status = tk.Label(self.root, text="starting…", anchor="w",
                               bg="#14171c", fg="#8fb8ff", font=("Segoe UI", 9))
        self.status.pack(fill="x", padx=8, pady=(6, 0))

        self.text = tk.Text(self.root, wrap="word", font=("Segoe UI", 12),
                            bg="#1b1f26", fg="#e8e8e8", relief="flat",
                            padx=10, pady=8, state="disabled")
        self.text.pack(fill="both", expand=True, padx=8, pady=6)

        bar = tk.Frame(self.root, bg="#14171c")
        bar.pack(fill="x", padx=8, pady=(0, 8))

        def btn(label, cmd, accent=False):
            return tk.Button(bar, text=label, command=cmd, relief="flat",
                             bg="#2b3a55" if accent else "#232833",
                             fg="#e8e8e8", activebackground="#3a4d73",
                             font=("Segoe UI", 9, "bold"), padx=10)

        self.btn_mic = btn("⏸ Pause", self.toggle_listen, accent=True)
        self.btn_mic.pack(side="left")
        btn("📋 Copy new", self.copy_new).pack(side="left", padx=(6, 0))
        btn("📋 Copy all", self.copy_all).pack(side="left", padx=(6, 0))
        btn("↵ Newline", self.insert_newline).pack(side="left", padx=(6, 0))
        btn("🗑 Clear", self.clear_text).pack(side="left", padx=(6, 0))

        self.autocopy = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="auto-copy", variable=self.autocopy,
                       bg="#14171c", fg="#e8e8e8", selectcolor="#232833",
                       activebackground="#14171c").pack(side="right")
        self.ontop = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="on-top", variable=self.ontop,
                       command=lambda: self.root.attributes(
                           "-topmost", bool(self.ontop.get())),
                       bg="#14171c", fg="#e8e8e8", selectcolor="#232833",
                       activebackground="#14171c").pack(side="right", padx=(0, 8))

        self.root.after(150, self._poll_results)

    # ---------------- mic + VAD ----------------
    def _start_mic(self) -> None:
        def callback(raw: bytes, frames, time_info, status) -> None:
            self._on_audio(raw, status)

        try:
            self.stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                blocksize=BLOCKSIZE, callback=callback)
            self.mic_rate = SAMPLE_RATE
        except sd.PortAudioError:
            self.stream = sd.RawInputStream(
                samplerate=44100, channels=1, dtype="int16",
                blocksize=44100 * BLOCK_MS // 1000, callback=callback)
            self.mic_rate = 44100
        self.stream.start()
        self.recording = True
        self.listening.set()
        self._set_status()

    def _on_audio(self, raw: bytes, status) -> None:
        if status:
            pass  # overflow on busy machines is survivable; keep going
        if not self.listening.is_set():
            return
        pcm = resample_to_16k(raw, self.mic_rate)
        level = rms_db_of(pcm)

        if self.warmup_blocks < ADAPT_SECONDS * 1000 / BLOCK_MS:
            self.warmup_blocks += 1
            self.noise_floor = 0.9 * self.noise_floor + 0.1 * level
            return
        speech_threshold = max(self.noise_floor * 3.0, AGC_FLOOR)

        if not self.in_speech:
            if level > speech_threshold:
                self.in_speech = True
                self.speech_run = 0.0
                self.silence_run = 0.0
        if self.in_speech:
            self.frames.append(pcm)
            self.utter_len += BLOCK_MS / 1000.0
            if level > speech_threshold * 0.6:
                self.speech_run += BLOCK_MS / 1000.0
                self.silence_run = 0.0
            else:
                self.silence_run += BLOCK_MS / 1000.0
            done = (self.silence_run >= SILENCE_TO_END_S
                    or self.utter_len >= MAX_UTTERANCE_S)
            if done:
                self._flush_utterance()

    def _flush_utterance(self) -> None:
        pcm = b"".join(self.frames)
        self.frames = []
        self.in_speech = False
        self.silence_run = 0.0
        self.utter_len = 0.0
        dur = len(pcm) / 2 / SAMPLE_RATE
        if dur >= MIN_SPEECH_S:
            self.segment_q.put(pcm)

    # ---------------- worker: segments -> server ----------------
    def _start_worker(self) -> None:
        def loop() -> None:
            while True:
                pcm = self.segment_q.get()
                if pcm is None:
                    break
                try:
                    text, latency = self.client.transcribe(pcm)
                    self.result_q.put(("ok", text, latency, len(pcm) / 2 / SAMPLE_RATE))
                except Exception as e:
                    self.result_q.put(("err", str(e), 0.0, 0.0))
        threading.Thread(target=loop, daemon=True).start()

    # ---------------- UI plumbing ----------------
    def _poll_results(self) -> None:
        try:
            while True:
                kind, text, latency, dur = self.result_q.get_nowait()
                if kind == "err":
                    self._set_status(f"⚠ server error: {text[:60]}")
                    continue
                self.stats["n"] += 1
                self.stats["ms"] = latency * 1000
                if text and not is_junk(text):
                    self._append(text + " ")
                    with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
                        f.write(text + "\n")
                    if self.autocopy.get():
                        self.copy_all(silent=True)
                self._set_status()
        except queue.Empty:
            pass
        self.root.after(120, self._poll_results)

    def _append(self, s: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", s)
        self.text.see("end")
        self.text.configure(state="disabled")

    def _set_status(self, extra: str = "") -> None:
        if not self.recording:
            state, color = "mic closed", "#ff8080"
        elif self.listening.is_set():
            state, color = "● listening", "#7ee787"
        else:
            state, color = "⏸ paused", "#f0c674"
        n, ms = self.stats["n"], self.stats["ms"]
        srv = "server ok" if self.server_ok else (
            "server DOWN — run start.bat" if self.server_ok is False else "checking server…")
        self.status.configure(
            text=f"{state}   ·   {srv} {SERVER_URL.split('//')[1]}   ·   "
                 f"{n} utterances, last {ms:.0f} ms   {extra}",
            fg=color)

    # ---------------- buttons ----------------
    def toggle_listen(self) -> None:
        if self.listening.is_set():
            self.listening.clear()
            self.btn_mic.configure(text="▶ Resume")
        else:
            self.listening.set()
            self.btn_mic.configure(text="⏸ Pause")
        self._set_status()

    def copy_all(self, silent: bool = False) -> None:
        content = self.text.get("1.0", "end").strip()
        if not content:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.copied_upto = self.text.index("end-1c")
        if not silent:
            self._set_status("· copied ✓")

    def copy_new(self) -> None:
        fresh = self.text.get(self.copied_upto, "end").strip()
        if not fresh:
            self._set_status("· nothing new")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(fresh)
        self.copied_upto = self.text.index("end-1c")
        self._set_status("· copied new ✓")

    def insert_newline(self) -> None:
        self._append("\n")

    def clear_text(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")
        self.copied_upto = "1.0"
        self.stats = {"n": 0, "ms": 0.0}

    def run(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.mainloop()

    def _close(self) -> None:
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        self.root.destroy()


def selftest() -> int:
    """No-GUI check: server round-trip on a real wav + VAD/junk-guard logic."""
    src = os.path.join(ROOT, "..", "hyper-ambient-src", "data", "in", "question.wav")
    if not os.path.exists(src):
        print("selftest: sample wav not found:", src)
        return 1
    with wave.open(src, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, "expected mono int16"
        raw = w.readframes(w.getnframes())
        if w.getframerate() != SAMPLE_RATE:
            raw = resample_to_16k(raw, w.getframerate())
    cli = ParakeetClient(SERVER_URL)
    if not cli.ping():
        print("selftest: server not reachable at", SERVER_URL)
        return 1
    text, latency = cli.transcribe(raw)
    print(f"selftest: {latency*1000:.0f} ms -> {text!r}")
    assert text, "empty transcription"
    assert not is_junk(text), "valid text flagged as junk"
    assert is_junk(""), "empty should be junk"
    assert is_junk("Subtitles by the Amara.org community"), "credit not caught"
    with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    print("selftest: OK (transcript.txt written)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    MiniHarness().run()
