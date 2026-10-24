#!/usr/bin/env python3
"""hyperambient-mini — light dictation harness (the "ears" of hyper-ambient only).

Mic -> push-to-talk (default) or energy VAD -> parakeet-server
(fluid-speech-mini GGUF, CPU) -> live text -> copy to clipboard,
so you can paste into ZCode / the DeepSeek desktop app.

Stdlib + sounddevice only. The STT server is a separate process (start.bat);
this app never touches the GPU, so a running training job is left alone.

Controls (PTT mode, default): hold F9 anywhere, hold the TALK button, or hold
Space when this window is focused. One activation = one text: the box and the
clipboard are wiped on press, so after release you paste exactly what you said.
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

# Global hold-to-talk hotkey: keys joined by '-'. Works from ANY app (passive
# key-state polling; nothing is consumed, nothing is logged).
HOTKEY = os.environ.get("HAM_HOTKEY", "f9").lower()

# --- VAD (energy-first, hyper-ambient style: spare the inference) ---------
SILENCE_TO_END_S = 0.75      # this much trailing quiet ends an utterance
MIN_SPEECH_S = 0.25          # shorter bursts are ignored (clicks, bumps)
MAX_UTTERANCE_S = 25.0       # hard cap, force-flush mid-speech
ADAPT_SECONDS = 2.0          # noise-floor learning window at session start
AGC_FLOOR = 220.0            # absolute RMS below this is never speech (int16)
NOISE_MULT = 2.5             # speech trigger = noise floor x this

METER_RANGE = 4000.0         # int16 RMS full-scale of the meter bar

# Whisper-style recited credits / empty output: drop rather than display.
JUNK_PATTERNS = (
    "subtitles by", "sous titrage", "amara org", "thanks for watching",
    "abonnez vous", "merci d avoir regarde", "subtitle", "www",
)

# --- palette ---------------------------------------------------------------
BG = "#0f1216"          # window
PANEL = "#171c22"       # meter / input fields
PANEL_HI = "#232a33"    # buttons, hover targets
TEXT = "#e6e9ee"
DIM = "#7d8794"
GREEN = "#69d58c"
RED = "#ff8fa8"
REC_A, REC_B = "#a72c4e", "#8e2440"   # pulsing recording pair
TALK_IDLE, TALK_HOVER = "#31202c", "#43293a"
PLACEHOLDER = "hold F9, speak, release — then Ctrl+V anywhere"


# ------------------------------------------------------------ audio utils ---
def as_samples(raw) -> array.array:
    a = array.array("h")
    a.frombytes(bytes(raw))
    return a


def rms_db_of(raw) -> float:
    """RMS of an int16 frame. array module: no numpy needed."""
    samples = as_samples(raw)
    if not samples:
        return 0.0
    acc = 0
    for s in samples:
        acc += s * s
    return (acc / len(samples)) ** 0.5


def resample_to_16k(raw: bytes, src_rate: int) -> bytes:
    """Linear resample int16 mono to 16 kHz (only used if the mic refuses 16k)."""
    if src_rate == SAMPLE_RATE:
        return bytes(raw)
    in_s = as_samples(raw)
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


# ------------------------------------------------------------ hotkey -----
VK = {"space": 0x20, "shift": 0x10, "ctrl": 0x11, "alt": 0x18, "win": 0x5B}
for _i in range(1, 13):
    VK[f"f{_i}"] = 0x70 + _i - 1
for _c in "abcdefghijklmnopqrstuvwxyz0123456789":
    VK[_c] = ord(_c.upper())


def hotkey_vks(spec: str) -> list[int] | None:
    parts = [p.strip() for p in spec.split("-") if p.strip()]
    if not parts:
        return None
    vks = [VK.get(p) for p in parts]
    return None if any(v is None for v in vks) else vks


# --------------------------------------------------------- tray (win32) ---
import ctypes
from ctypes import byref
from ctypes import wintypes

_user32 = ctypes.windll.user32
_kernel32 = ctypes.windll.kernel32
_gdi32 = ctypes.windll.gdi32
_shell32 = ctypes.windll.shell32

WM_APP = 0x8000
TRAY_CALLBACK = WM_APP + 1
WM_LBUTTONUP, WM_LBUTTONDBLCLK, WM_RBUTTONUP = 0x0202, 0x0203, 0x0205
NIM_ADD, NIM_DELETE = 0, 2
NIF_MESSAGE, NIF_ICON, NIF_TIP = 0x01, 0x02, 0x04
_HWND_MESSAGE = wintypes.HWND(-3)
_LRESULT = ctypes.c_ssize_t

TRAY_WNDPROC = ctypes.WINFUNCTYPE(
    _LRESULT, wintypes.HWND, wintypes.UINT,
    wintypes.WPARAM, wintypes.LPARAM)


class _WNDCLASSW(ctypes.Structure):
    _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", TRAY_WNDPROC),
                ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
                ("hCursor", ctypes.c_void_p), ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR),
                ("lpszClassName", wintypes.LPCWSTR)]


class _ICONINFO(ctypes.Structure):
    _fields_ = [("fIcon", wintypes.BOOL), ("xHotspot", wintypes.DWORD),
                ("yHotspot", wintypes.DWORD), ("hbmMask", wintypes.HBITMAP),
                ("hbmColor", wintypes.HBITMAP)]


class _NOTIFYICONDATAW(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("hWnd", wintypes.HWND),
                ("uID", wintypes.UINT), ("uFlags", wintypes.UINT),
                ("uCallbackMessage", wintypes.UINT), ("hIcon", wintypes.HICON),
                ("szTip", wintypes.WCHAR * 128), ("dwState", wintypes.DWORD),
                ("dwStateMask", wintypes.DWORD),
                ("szInfo", wintypes.WCHAR * 256), ("uVersion", wintypes.UINT),
                ("szInfoTitle", wintypes.WCHAR * 64),
                ("dwInfoFlags", wintypes.DWORD),
                ("guidItem", ctypes.c_byte * 16), ("hBalloonIcon", wintypes.HICON)]


# explicit signatures: on 64-bit windows, default c_int restypes truncate handles
_kernel32.GetModuleHandleW.restype = wintypes.HMODULE
_kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
_user32.CreateWindowExW.restype = wintypes.HWND
_user32.CreateWindowExW.argtypes = [
    wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
_user32.DefWindowProcW.restype = _LRESULT
_user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT,
                                   wintypes.WPARAM, wintypes.LPARAM]
_user32.RegisterClassW.restype = wintypes.ATOM
_user32.RegisterClassW.argtypes = [ctypes.c_void_p]
_shell32.Shell_NotifyIconW.restype = wintypes.BOOL
_shell32.Shell_NotifyIconW.argtypes = [wintypes.DWORD, ctypes.c_void_p]
_gdi32.CreateBitmap.restype = wintypes.HBITMAP
_gdi32.CreateBitmap.argtypes = [ctypes.c_int, ctypes.c_int,
                                wintypes.UINT, wintypes.UINT, ctypes.c_void_p]
_user32.CreateIconIndirect.restype = wintypes.HICON
_user32.CreateIconIndirect.argtypes = [ctypes.c_void_p]


def _make_dot_icon(size: int = 32) -> int:
    """Red recording-dot icon, drawn by hand: no resource files needed."""
    color = bytearray(size * size * 4)                  # BGRA
    mask = bytearray(b"\xff" * (size * size // 8))      # 1bpp, 1 = transparent
    for y in range(size):
        for x in range(size):
            if (x - size / 2 + 0.5) ** 2 + (y - size / 2 + 0.5) ** 2 <= (size * 0.42) ** 2:
                i = y * size + x
                color[i * 4:i * 4 + 4] = b"\x5a\x2c\xa7\xff"   # BGRA #a72c5a
                mask[i // 8] &= ~(0x80 >> (i % 8)) & 0xFF
    hbm_color = _gdi32.CreateBitmap(size, size, 1, 32, bytes(color))
    hbm_mask = _gdi32.CreateBitmap(size, size, 1, 1, bytes(mask))
    ii = _ICONINFO(True, 0, 0, hbm_mask, hbm_color)
    return _user32.CreateIconIndirect(byref(ii))


# --------------------------------------------------------------- the app ---
class MiniHarness:
    def __init__(self) -> None:
        self.client = ParakeetClient(SERVER_URL)
        self.root = tk.Tk()
        self.mode = tk.StringVar(value="ptt")     # "ptt" | "auto"
        self.listening = threading.Event()        # auto mode: paused <=> cleared
        self.ptt_active = False
        self._last_space_press = 0.0              # autorepeat guard
        self.segment_q: "queue.Queue[bytes]" = queue.Queue()
        self.result_q: "queue.Queue[tuple]" = queue.Queue()
        self.ui_q: "queue.Queue[tuple]" = queue.Queue()   # cross-thread UI cmds
        self.frames: list[bytes] = []             # current utterance buffer
        self.in_speech = False
        self.silence_run = 0.0
        self.utter_len = 0.0
        self.noise_floor = 120.0
        self.warmup_blocks = 0
        self.last_level = 0.0
        self.zero_blocks = 0
        self.mic_alive = True
        self._placeholder_on = False
        self.copied_upto = "1.0"                  # text index already copied
        self.stats = {"n": 0, "ms": 0.0}
        self.server_ok: bool | None = None
        self.device_index = None                  # None = system default
        self._build_ui()
        self._start_worker()
        self._start_mic(self.device_index)
        self._start_hotkey(HOTKEY)
        self.root.after(200, self._focus_for_ptt)
        threading.Thread(target=self._probe_server, daemon=True).start()
        self._init_tray()

    def _probe_server(self) -> None:
        self.server_ok = self.client.ping()
        self.root.after(0, self._set_status)

    def _start_hotkey(self, spec: str) -> None:
        """Global hold-to-talk: poll key state from a thread; any app in focus."""
        vks = hotkey_vks(spec)
        if not vks:
            self._set_status(f"⚠ bad HAM_HOTKEY spec: {spec}")
            return
        self.hotkey_label = spec.upper().replace("-", "+")

        def poll() -> None:
            import ctypes
            user32 = ctypes.windll.user32
            held = False
            while True:
                time.sleep(0.02)
                try:
                    down = all(user32.GetAsyncKeyState(v) & 0x8000 for v in vks)
                except Exception:
                    return
                if down != held:
                    held = down
                    self.ui_q.put(("ptt", down))
        threading.Thread(target=poll, daemon=True).start()

    # ---------------- UI ----------------
    def _build_ui(self) -> None:
        self.root.title("hyperambient-mini")
        self.root.geometry("620x420")
        self.root.minsize(560, 400)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=BG)
        # subtle gradient backdrop: visible in the gutters around the panels
        self.bg = tk.Canvas(self.root, highlightthickness=0, bg=BG)
        self.bg.place(x=0, y=0, relwidth=1, relheight=1)
        self.bg.bind("<Configure>", self._paint_bg)
        self.root.option_add("*TCombobox*Listbox.background", PANEL_HI)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#3a4d73")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Dark.TCombobox", fieldbackground=PANEL, background=PANEL_HI,
                        foreground=TEXT, arrowcolor=DIM, bordercolor=PANEL_HI,
                        lightcolor=PANEL, darkcolor=PANEL, insertcolor=TEXT)

        # header: wordmark + mode chip
        head = tk.Frame(self.root, bg=BG)
        head.pack(fill="x", padx=14, pady=(12, 6))
        tk.Label(head, text="hyperambient", bg=BG, fg=TEXT,
                 font=("Segoe UI", 11, "bold")).pack(side="left")
        tk.Label(head, text="mini", bg=BG, fg=DIM,
                 font=("Segoe UI", 11, "bold italic")).pack(side="left")
        self.btn_mode = tk.Label(head, text="", bg=PANEL, fg=DIM, padx=10, pady=2,
                                 font=("Segoe UI", 9, "bold"), cursor="hand2")
        self.btn_mode.pack(side="right")
        self.btn_mode.bind("<Button-1>", lambda e: self._cycle_mode())

        self.status = tk.Label(self.root, text="starting…", anchor="w",
                               bg=BG, fg=DIM, font=("Segoe UI", 9))
        self.status.pack(fill="x", padx=14)

        # level meter: what the mic hears vs the auto-mode trigger
        meter = tk.Frame(self.root, bg=BG)
        meter.pack(fill="x", padx=14, pady=(6, 0))
        tk.Label(meter, text="MIC", bg=BG, fg=DIM,
                 font=("Segoe UI", 7, "bold")).pack(side="left")
        self.meter = tk.Canvas(meter, height=8, bg=PANEL, highlightthickness=0)
        self.meter.pack(fill="x", expand=True, side="left", padx=(8, 8))
        self.meter_label = tk.Label(meter, text="", bg=BG, fg=DIM,
                                    font=("Segoe UI", 8), width=8, anchor="e")
        self.meter_label.pack(side="left")

        self.text = tk.Text(self.root, wrap="word", font=("Segoe UI", 13),
                            bg=PANEL, fg=TEXT, relief="flat", insertbackground=TEXT,
                            padx=14, pady=12, spacing3=4, state="disabled",
                            highlightthickness=1, highlightbackground=PANEL_HI,
                            highlightcolor="#3a4d73")
        self.text.pack(fill="both", expand=True, padx=14, pady=8)
        self.text.tag_configure("placeholder", foreground=DIM,
                                font=("Segoe UI Italic", 13))

        # talk button
        self.btn_talk = tk.Label(
            self.root,
            text="⏺   Hold F9 and speak  —  release to send",
            font=("Segoe UI", 12, "bold"), bg=TALK_IDLE, fg="#ffd9e2",
            padx=14, pady=14, cursor="hand2")
        self.btn_talk.pack(fill="x", padx=14)
        self.btn_talk.bind("<Enter>", lambda e: self._talk_hover(True))
        self.btn_talk.bind("<Leave>", lambda e: self._talk_hover(False))
        # implicit grab: once pressed, the release reaches us even if the
        # pointer slides off the button before letting go
        self.btn_talk.bind("<ButtonPress-1>",
                           lambda e: (self.ptt_start(), self.btn_talk.grab_set()))
        self.btn_talk.bind("<ButtonRelease-1>",
                           lambda e: self._ptt_release())
        self.root.bind("<ButtonRelease-1>", lambda e: self._ptt_release())
        self.root.bind("<Escape>", lambda e: self._ptt_release())

        # actions
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", padx=14, pady=(10, 0))

        def btn(label, cmd, accent=False):
            b = tk.Button(bar, text=label, command=cmd, relief="flat",
                          bg="#2b4a7a" if accent else PANEL_HI, fg=TEXT,
                          activebackground="#3a4d73" if accent else "#313a46",
                          activeforeground="#ffffff", borderwidth=0,
                          font=("Segoe UI", 9, "bold"), padx=14, pady=6,
                          cursor="hand2")
            b.bind("<Enter>", lambda e: b.configure(bg="#3a4d73" if accent else "#313a46"))
            b.bind("<Leave>", lambda e: b.configure(bg="#2b4a7a" if accent else PANEL_HI))
            return b

        btn("📋 Copy", self.copy_all, accent=True).pack(side="left")
        btn("copy new", self.copy_new).pack(side="left", padx=(6, 0))
        btn("↵ newline", self.insert_newline).pack(side="left", padx=(6, 0))
        btn("clear", self.clear_text).pack(side="left", padx=(6, 0))
        self.btn_mic = btn("⏸ pause", self.toggle_listen)
        self.btn_mic.pack(side="left", padx=(6, 0))

        self.autocopy = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="auto-copy", variable=self.autocopy,
                       bg=BG, fg=DIM, selectcolor=PANEL_HI,
                       activebackground=BG, activeforeground=TEXT,
                       font=("Segoe UI", 9)).pack(side="right")

        # footer: mic + hint
        dev = tk.Frame(self.root, bg=BG)
        dev.pack(fill="x", padx=14, pady=(8, 10))
        tk.Label(dev, text="mic", bg=BG, fg=DIM,
                 font=("Segoe UI", 8)).pack(side="left")
        self.device_box = ttk.Combobox(dev, state="readonly", width=42,
                                       style="Dark.TCombobox",
                                       font=("Segoe UI", 9))
        self.device_box.pack(side="left", padx=(8, 0))
        self.device_box.bind("<<ComboboxSelected>>", self._device_changed)
        self._fill_devices()
        tk.Label(dev, text="one activation = one text", bg=BG, fg="#4c545e",
                 font=("Segoe UI", 8)).pack(side="right")

        # Space = push-to-talk while the window has focus (autorepeat-guarded)
        self.root.bind("<KeyPress-space>", self._space_press)
        self.root.bind("<KeyRelease-space>", self._space_release)
        self.root.bind("<Unmap>", self._on_unmap)
        self.root.after(150, self._poll_results)
        self._set_placeholder()

    def _talk_hover(self, on: bool) -> None:
        if not self.ptt_active:
            self.btn_talk.configure(bg=TALK_HOVER if on else TALK_IDLE)

    def _set_placeholder(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", PLACEHOLDER, ("placeholder",))
        self.text.configure(state="disabled")
        self._placeholder_on = True

    def _focus_for_ptt(self) -> None:
        self.root.focus_set()
        self.btn_talk.focus_set()

    def _fill_devices(self) -> None:
        try:
            inputs = [(d["index"], d["name"]) for d in sd.query_devices()
                      if d["max_input_channels"] > 0]
        except Exception:
            inputs = []
        self._devices = inputs
        default = sd.query_devices(kind="input")["index"] if inputs else None
        names, current = [], 0
        for i, (idx, name) in enumerate(inputs):
            mark = "  ← default" if idx == default else ""
            names.append(f"[{idx}] {name}{mark}")
            if idx == default:
                current = i
        self.device_box["values"] = names or ["no input device found"]
        if names:
            self.device_box.current(current)

    def _device_changed(self, _event) -> None:
        if not getattr(self, "_devices", None):
            return
        pos = self.device_box.current()
        if 0 <= pos < len(self._devices):
            self._start_mic(self._devices[pos][0])
            self._set_status(f"· switched to {self._devices[pos][1][:30]}")

    def _cycle_mode(self) -> None:
        self.mode.set("auto" if self.mode.get() == "ptt" else "ptt")
        self._mode_changed()

    def _mode_changed(self) -> None:
        if self.mode.get() == "ptt":
            self.btn_mic.configure(state="disabled")
        else:
            self.btn_mic.configure(state="normal")
            self.listening.set()
        self._set_status()

    # ---------------- mic ----------------
    def _start_mic(self, device_index) -> None:
        def callback(raw, frames, time_info, status) -> None:
            self._on_audio(raw)

        old = getattr(self, "stream", None)
        if old is not None:
            try:
                old.stop()
                old.close()
            except Exception:
                pass
        try:
            self.stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                blocksize=BLOCKSIZE, callback=callback, device=device_index)
            self.mic_rate = SAMPLE_RATE
        except sd.PortAudioError:
            self.stream = sd.RawInputStream(
                samplerate=44100, channels=1, dtype="int16",
                blocksize=44100 * BLOCK_MS // 1000, callback=callback,
                device=device_index)
            self.mic_rate = 44100
        try:
            self.stream.start()
        except Exception as e:
            self.mic_alive = False
            self._set_status(f"⚠ mic failed to open: {e}")
            return
        self.mic_alive = True
        self.warmup_blocks = 0
        self._set_status()

    def _on_audio(self, raw) -> None:
        pcm = resample_to_16k(raw, self.mic_rate)
        level = rms_db_of(pcm)
        self.last_level = level
        self.zero_blocks = self.zero_blocks + 1 if level < 1.0 else 0

        if self.mode.get() == "ptt":
            if self.ptt_active:
                self.frames.append(pcm)
                self.utter_len += BLOCK_MS / 1000.0
                if self.utter_len >= MAX_UTTERANCE_S:
                    self._flush_utterance()   # cap; user can keep talking
                    self.frames = [pcm]
                    self.utter_len = BLOCK_MS / 1000.0
            return

        # auto VAD mode
        if not self.listening.is_set():
            return
        if self.warmup_blocks < ADAPT_SECONDS * 1000 / BLOCK_MS:
            self.warmup_blocks += 1
            self.noise_floor = 0.9 * self.noise_floor + 0.1 * level
            return
        speech_threshold = max(self.noise_floor * NOISE_MULT, AGC_FLOOR)

        if not self.in_speech and level > speech_threshold:
            self.in_speech = True
            self.silence_run = 0.0
        if self.in_speech:
            self.frames.append(pcm)
            self.utter_len += BLOCK_MS / 1000.0
            if level > speech_threshold * 0.55:
                self.silence_run = 0.0
            else:
                self.silence_run += BLOCK_MS / 1000.0
            if self.silence_run >= SILENCE_TO_END_S or self.utter_len >= MAX_UTTERANCE_S:
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

    # ---------------- push-to-talk ----------------
    def ptt_start(self) -> None:
        if self.mode.get() != "ptt" or self.ptt_active or not self.mic_alive:
            return
        self.ptt_active = True
        # one activation = one text: previous utterance and clipboard go away
        self.clear_text()
        try:
            self.root.clipboard_clear()
            self.root.update()
        except Exception:
            pass
        self.frames = []
        self.utter_len = 0.0
        self.btn_talk.configure(bg=REC_A, fg="#ffe8ec",
                                text="●   Listening… release to send")
        self._set_status()

    def _ptt_release(self) -> None:
        try:
            self.btn_talk.grab_release()
        except Exception:
            pass
        self.ptt_stop()

    def ptt_stop(self) -> None:
        if not self.ptt_active:
            return
        self.ptt_active = False
        self.btn_talk.configure(
            bg=TALK_IDLE, fg="#ffd9e2",
            text="⏺   Hold F9 and speak  —  release to send")
        self._flush_utterance()
        self._set_status()

    def _space_press(self, _event) -> None:
        self._last_space_press = time.time()
        self.ptt_start()

    def _space_release(self, _event) -> None:
        # Windows autorepeat interleaves KeyRelease between KeyPress events;
        # only honor a release that is not immediately followed by a repeat.
        if time.time() - self._last_space_press > 0.06:
            self.ptt_stop()
        else:
            self.root.after(80, self._space_release_check)

    def _space_release_check(self) -> None:
        if time.time() - self._last_space_press > 0.06:
            self.ptt_stop()

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
                kind, payload = self.ui_q.get_nowait()
                if kind == "ptt":
                    (self.ptt_start if payload else self._ptt_release)()
        except queue.Empty:
            pass
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
                elif dur > 0.5 and not text:
                    self._set_status("· heard only silence (check mic / level meter)")
                    continue
                self._set_status()
        except queue.Empty:
            pass
        self._draw_meter()
        self._pulse_talk()
        self.root.after(120, self._poll_results)

    def _pulse_talk(self) -> None:
        if self.ptt_active:
            bg = REC_A if int(time.time() * 4) % 2 else REC_B
            self.btn_talk.configure(bg=bg)

    def _draw_meter(self) -> None:
        w = self.meter.winfo_width()
        if w < 20:
            return
        self.meter.delete("all")
        frac = min(self.last_level / METER_RANGE, 1.0)
        color = "#69d58c" if frac < 0.55 else "#e0b34c" if frac < 0.85 else "#e05a5a"
        if frac > 0.01:
            self.meter.create_rectangle(0, 0, int(w * frac), 8, fill=color, width=0)
        if self.mode.get() == "auto" and self.warmup_blocks >= ADAPT_SECONDS * 1000 / BLOCK_MS:
            thr = max(self.noise_floor * NOISE_MULT, AGC_FLOOR) / METER_RANGE
            if 0 < thr < 1:
                x = int(w * min(thr, 1.0))
                self.meter.create_line(x, 0, x, 8, fill="#f0c674", width=2)
        hint = "SILENT — check device / privacy" if self.zero_blocks > 100 \
            else f"{self.last_level:.0f}"
        self.meter_label.configure(text=hint)

    def _append(self, s: str) -> None:
        self.text.configure(state="normal")
        if self._placeholder_on:
            self.text.delete("1.0", "end")
            self._placeholder_on = False
        self.text.insert("end", s)
        self.text.see("end")
        self.text.configure(state="disabled")

    def _set_status(self, extra: str = "") -> None:
        if not self.mic_alive:
            state, color = "mic failed", "#ff8080"
        elif self.mode.get() == "ptt":
            hk = f" · hold {getattr(self, 'hotkey_label', 'F9')} anywhere" if self.server_ok else ""
            state = "● recording" if self.ptt_active else "⏺ push-to-talk ready"
            state += hk
            color = RED if self.ptt_active else GREEN
        elif self.listening.is_set():
            state, color = "● auto listening", GREEN
        else:
            state, color = "⏸ paused", "#f0c674"
        n, ms = self.stats["n"], self.stats["ms"]
        srv = "server ok" if self.server_ok else (
            "server DOWN — run start.bat" if self.server_ok is False else "checking server…")
        self.status.configure(
            text=f"{state}   ·   {srv}   ·   {n} utterances, last {ms:.0f} ms   {extra}",
            fg=color)
        self.btn_mode.configure(
            text=("MODE · PTT" if self.mode.get() == "ptt" else "MODE · AUTO VAD"))

    # ---------------- buttons ----------------
    def toggle_listen(self) -> None:
        if self.listening.is_set():
            self.listening.clear()
            self.btn_mic.configure(text="▶ resume")
        else:
            self.listening.set()
            self.btn_mic.configure(text="⏸ pause")
        self._set_status()

    def copy_all(self, silent: bool = False) -> None:
        if self._placeholder_on:
            return
        content = self.text.get("1.0", "end").strip()
        if not content:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.root.update()          # win the clipboard race with the target app
        self.copied_upto = self.text.index("end-1c")
        if not silent:
            self._set_status("· copied ✓")

    def copy_new(self) -> None:
        if self._placeholder_on:
            self._set_status("· nothing new")
            return
        fresh = self.text.get(self.copied_upto, "end").strip()
        if not fresh:
            self._set_status("· nothing new")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(fresh)
        self.root.update()
        self.copied_upto = self.text.index("end-1c")
        self._set_status("· copied new ✓")

    def insert_newline(self) -> None:
        if self._placeholder_on:
            return
        self._append("\n")

    def clear_text(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")
        self.copied_upto = "1.0"
        self.stats = {"n": 0, "ms": 0.0}
        self._set_placeholder()

    # ---------------- tray ----------------
    def _init_tray(self) -> None:
        """Own a tray icon via a message-only window; no extra dependencies."""
        try:
            self._wm_taskbarcreated = _user32.RegisterWindowMessageW("TaskbarCreated")
            self._wndproc = TRAY_WNDPROC(self._tray_wndproc)
            wc = _WNDCLASSW()
            wc.lpfnWndProc = self._wndproc
            wc.hInstance = _kernel32.GetModuleHandleW(None)
            wc.lpszClassName = "HyperAmbientMiniTray"
            if not _user32.RegisterClassW(byref(wc)):
                raise OSError("RegisterClassW failed")
            # a real (never shown) window, not HWND_MESSAGE: message-only
            # windows are unreliable recipients of tray callbacks on windows
            self._tray_hwnd = _user32.CreateWindowExW(
                0, "HyperAmbientMiniTray", "ham-tray", 0,  # WS_OVERLAPPED, hidden
                0, 0, 0, 0, None, None, wc.hInstance, None)
            self._tray_icon = _make_dot_icon()
            self._nid = _NOTIFYICONDATAW()
            self._nid.cbSize = ctypes.sizeof(_NOTIFYICONDATAW)
            self._nid.hWnd = self._tray_hwnd
            self._nid.uID = 1
            self._nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP
            self._nid.uCallbackMessage = TRAY_CALLBACK
            self._nid.hIcon = self._tray_icon
            self._nid.szTip = "hyperambient-mini — hold F9 to talk"
            ok = self._tray_add()
            print(f"tray icon added: {ok}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"tray unavailable ({e}); window-only mode", flush=True)

    def _tray_add(self) -> bool:
        return bool(_shell32.Shell_NotifyIconW(NIM_ADD, byref(self._nid)))

    def _tray_wndproc(self, hwnd, msg, wparam, lparam):
        try:
            if msg == TRAY_CALLBACK:
                print(f"tray callback: mouse msg {lparam:#x}", flush=True)
                if lparam in (WM_LBUTTONUP, WM_LBUTTONDBLCLK):
                    self.root.after(0, self._tray_toggle)
                elif lparam == WM_RBUTTONUP:
                    self.root.after(0, self._show_tray_menu)
                return 0
            if msg == self._wm_taskbarcreated:      # explorer restarted
                self._tray_add()
                return 0
        except Exception:
            pass
        return _user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _tray_toggle(self) -> None:
        was = self.root.state()
        if was == "withdrawn":
            self.root.deiconify()
            self.root.state("normal")       # deiconify alone may restore iconic
            self.root.attributes("-topmost", True)
            self.root.lift()
            self.root.focus_force()
        else:
            self._hide_to_tray()
        print(f"tray toggle: {was} -> {self.root.state()}", flush=True)

    def _hide_to_tray(self) -> None:
        self.root.withdraw()

    def _show_tray_menu(self) -> None:
        m = tk.Menu(self.root, tearoff=0, bg=PANEL_HI, fg=TEXT,
                    activebackground="#3a4d73", activeforeground="#ffffff",
                    font=("Segoe UI", 9))
        m.add_command(label="Show / hide", command=self._tray_toggle)
        m.add_separator()
        m.add_command(label="Quit", command=self._quit)
        try:
            m.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery(), 0)
        finally:
            m.grab_release()

    def _on_unmap(self, _event) -> None:
        # the minimize button hides to tray instead of clogging the taskbar
        if self.root.state() == "iconic":
            self.root.withdraw()

    def _quit(self) -> None:
        try:
            _shell32.Shell_NotifyIconW(NIM_DELETE, byref(self._nid))
        except Exception:
            pass
        self._close()

    def _paint_bg(self, _event=None) -> None:
        c, w, h = self.bg, self.bg.winfo_width(), self.bg.winfo_height()
        if w < 2 or h < 2:
            return
        c.delete("grad")
        top, bot, steps = (0x18, 0x1e, 0x28), (0x0d, 0x10, 0x15), 48
        for i in range(steps):
            t = i / (steps - 1)
            col = "#%02x%02x%02x" % tuple(
                int(a + (b - a) * t) for a, b in zip(top, bot))
            y0 = int(h * i / steps)
            c.create_rectangle(0, y0, w, int(h * (i + 1) / steps) + 1,
                               fill=col, width=0, tags="grad")
        c.tag_lower("grad")

    def run(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self._hide_to_tray)
        self.root.mainloop()

    def _close(self) -> None:
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        self.root.destroy()


def selftest() -> int:
    """No-GUI check: server round-trip on a real wav + junk-guard logic."""
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
    assert is_junk("No, no, no, no."), "repeat hallucination not caught"
    assert not is_junk("No, that is not what I meant"), "false positive"
    with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    print("selftest: OK (transcript.txt written)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    MiniHarness().run()
