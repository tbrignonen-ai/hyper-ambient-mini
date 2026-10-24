#!/usr/bin/env python3
"""hyperambient-mini — light dictation harness (the "ears" of hyper-ambient only).

Mic -> push-to-talk -> parakeet-server (fluid-speech-mini GGUF, CPU) -> text
-> clipboard, so you can paste into ZCode / the DeepSeek desktop app.

Push-to-talk ONLY (no VAD): hold F9 anywhere, hold the TALK button, or hold
Space when this window is focused. One activation = one text: the box and the
clipboard are wiped on press, so after release you paste exactly what you said.

Stdlib + sounddevice only. The STT server is a separate process (start.bat);
this app never touches the GPU, so a running training job is left alone.
"""
from __future__ import annotations

import array
import io
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave

import sounddevice as sd
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

# ---------------------------------------------------------------- config ---
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# frozen (exe) mode: model, server and transcript live next to the executable
IS_FROZEN = getattr(sys, "frozen", False)
BASE_DIR = os.path.dirname(sys.executable) if IS_FROZEN else ROOT
SERVER_URL = os.environ.get("PARK_SERVER_URL", "http://127.0.0.1:8077")
TRANSCRIPT_FILE = os.path.join(BASE_DIR, "transcript.txt")
SERVER_EXE = os.path.join(BASE_DIR, "bin", "parakeet-server.exe")
MODEL_GGUF = os.path.join(BASE_DIR, "models", "fluid-speech-mini-q2k-q4k.gguf")

SAMPLE_RATE = 16000          # what the model wants
BLOCK_MS = 30                # audio callback granularity
BLOCKSIZE = SAMPLE_RATE * BLOCK_MS // 1000

# Global hold-to-talk hotkey: keys joined by '-'. Works from ANY app (passive
# key-state polling; nothing is consumed, nothing is logged).
HOTKEY = os.environ.get("HAM_HOTKEY", "f9").lower()
SHOW_HOTKEY = os.environ.get("HAM_SHOW_HOTKEY", "ctrl-alt-h").lower()

MIN_SPEECH_S = 0.25          # shorter bursts are ignored (clicks, bumps)
MAX_UTTERANCE_S = 25.0       # hard cap, force-flush mid-speech
METER_RANGE = 4000.0         # int16 RMS full-scale of the meter bar

# Whisper-style recited credits / empty output: drop rather than display.
JUNK_PATTERNS = (
    "subtitles by", "sous titrage", "amara org", "thanks for watching",
    "abonnez vous", "merci d avoir regarde", "subtitle", "www",
)

# --- palette: synthwave chrome over a blanc-cassé text field -------------
BG_TOP = (0x1b, 0x23, 0x50)  # gradient, top (deep indigo blue)
BG_BOT = (0x0c, 0x10, 0x2c)  # gradient, bottom (midnight navy)
PANEL = "#101433"            # dark chrome panels (meter, combobox, menu)
CHIP = "#1a2148"             # quiet chips
CHIP_HOVER = "#232c5c"
EDGE = "#2b3768"             # outlines
TEXT = "#eef1ff"
DIM = "#aab3e8"
DIM2 = "#7a86c9"
PINK = "#ff2e88"             # neon pink
FUCHSIA = "#f222ff"         # synthwave fuchsia
CYAN = "#01cdfe"             # neon cyan
BLUE = "#2d5bff"             # electric blue
ORANGE = "#ff9e3d"           # sunset orange
YELLOW = "#ffe74c"           # sunset yellow
FIELD = "#f1e6c8"            # creamy blanc cassé — the transcript field only
FIELD_TEXT = "#0d8a4e"     # emerald green
TEAL = "#01cdfe"             # legacy name (ready status = cyan)
REC_A, REC_B = "#d81b60", "#b3124d"             # pulsing recording pair
TALK_STOPS = (("#ff2e88", 0.0), ("#ff9e3d", 0.55), ("#ffe74c", 1.0))
PANEL_HI, PANEL_HOVER = CHIP, CHIP_HOVER        # legacy names
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
_user32.GetMessageW.restype = ctypes.c_int
_user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                                wintypes.UINT, wintypes.UINT]
_user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
_user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]


def _make_dot_icon(size: int = 32) -> int:
    """Red recording-dot icon, drawn by hand: no resource files needed."""
    color = bytearray(size * size * 4)                  # BGRA
    mask = bytearray(b"\xff" * (size * size // 8))      # 1bpp, 1 = transparent
    for y in range(size):
        for x in range(size):
            if (x - size / 2 + 0.5) ** 2 + (y - size / 2 + 0.5) ** 2 <= (size * 0.42) ** 2:
                i = y * size + x
                color[i * 4:i * 4 + 4] = b"\x88\x2e\xff\xff"   # BGRA #ff2e88
                mask[i // 8] &= ~(0x80 >> (i % 8)) & 0xFF
    hbm_color = _gdi32.CreateBitmap(size, size, 1, 32, bytes(color))
    hbm_mask = _gdi32.CreateBitmap(size, size, 1, 1, bytes(mask))
    ii = _ICONINFO(True, 0, 0, hbm_mask, hbm_color)
    return _user32.CreateIconIndirect(byref(ii))


def _lerp_stops(stops, t):
    """Multi-stop hex gradient interpolation."""
    if t <= stops[0][1]:
        return stops[0][0]
    for (c0, p0), (c1, p1) in zip(stops, stops[1:]):
        if t <= p1:
            f = (t - p0) / (p1 - p0)
            a = tuple(int(c0[j:j + 2], 16) for j in (1, 3, 5))
            b = tuple(int(c1[j:j + 2], 16) for j in (1, 3, 5))
            return "#%02x%02x%02x" % tuple(
                int(x + (y - x) * f) for x, y in zip(a, b))
    return stops[-1][0]


# --------------------------------------------------------------- the app ---
class MiniHarness:
    def __init__(self) -> None:
        # single instance: a second copy would fight over F9 and the clipboard
        self._lock = socket.socket()
        try:
            self._lock.bind(("127.0.0.1", 8079))
        except OSError:
            _user32.MessageBoxW(
                None, "hyperambient-mini is already running.\n"
                      "Look for the red dot in the Windows tray.",
                "hyperambient-mini", 0x30)
            sys.exit(0)
        self.client = ParakeetClient(SERVER_URL)
        self.root = tk.Tk()
        self.ptt_active = False
        self._talk_hovered = False
        self._last_space_press = 0.0              # autorepeat guard
        self.segment_q: "queue.Queue[bytes]" = queue.Queue()
        self.result_q: "queue.Queue[tuple]" = queue.Queue()
        self.ui_q: "queue.Queue[tuple]" = queue.Queue()   # cross-thread UI cmds
        self.frames: list[bytes] = []             # current utterance buffer
        self.utter_len = 0.0
        self.last_level = 0.0
        self.zero_blocks = 0
        self.mic_alive = True
        self.copied_upto = "1.0"                  # text index already copied
        self.stats = {"n": 0, "ms": 0.0}
        self.server_ok: bool | None = None
        self._server_proc = None
        self.device_index = None                  # None = system default
        self.hotkey_label = "F9"
        self._build_ui()
        self._start_worker()
        self._start_mic(self.device_index)
        self._start_hotkey(HOTKEY)
        self.root.after(200, self._focus_for_ptt)
        self._ensure_server()
        self._init_tray()

    def _ensure_server(self) -> None:
        """Use a running STT server, or spawn one silently next to the exe."""
        def probe_spawn() -> None:
            if self.client.ping():
                self.server_ok = True
                self.root.after(0, self._set_status)
                return
            if not (os.path.exists(SERVER_EXE) and os.path.exists(MODEL_GGUF)):
                self.server_ok = False
                self.root.after(0, lambda: self._set_status(
                    "⚠ bin/parakeet-server.exe or models/ missing"))
                return
            try:
                log = open(os.path.join(BASE_DIR, "server.log"), "ab")
                self._server_proc = subprocess.Popen(
                    [SERVER_EXE, "--model", MODEL_GGUF, "--host", "127.0.0.1",
                     "--port", "8077", "--threads", "4"],
                    stdout=log, stderr=log, cwd=BASE_DIR,
                    creationflags=0x00000008)      # DETACHED_PROCESS: no console
            except Exception as e:
                self.server_ok = False
                self.root.after(0, lambda: self._set_status(f"⚠ {e}"))
                return
            for _ in range(100):                   # up to ~30 s to load 347 MB
                time.sleep(0.3)
                if self.client.ping():
                    self.server_ok = True
                    self.root.after(0, self._set_status)
                    return
            self.server_ok = False
            self.root.after(0, lambda: self._set_status(
                "⚠ server did not come up — see server.log"))
        threading.Thread(target=probe_spawn, daemon=True).start()

    def _start_hotkey(self, spec: str) -> None:
        """Global hold-to-talk: poll key state from a thread; any app in focus."""
        vks = hotkey_vks(spec)
        if not vks:
            self._set_status(f"⚠ bad HAM_HOTKEY spec: {spec}")
            return
        self.hotkey_label = spec.upper().replace("-", "+")

        show_vks = hotkey_vks(SHOW_HOTKEY)
        self.show_hotkey_label = SHOW_HOTKEY.upper().replace("-", "+")

        def poll() -> None:
            user32 = ctypes.windll.user32
            held = False
            shown = False
            while True:
                time.sleep(0.02)
                try:
                    down = all(user32.GetAsyncKeyState(v) & 0x8000 for v in vks)
                    sdown = bool(show_vks) and all(
                        user32.GetAsyncKeyState(v) & 0x8000 for v in show_vks)
                except Exception:
                    return
                if down != held:
                    held = down
                    self.ui_q.put(("ptt", down))
                if sdown != shown:
                    shown = sdown
                    if sdown:
                        self.ui_q.put(("show", True))
        threading.Thread(target=poll, daemon=True).start()

    # ---------------- UI ----------------
    def _build_ui(self) -> None:
        # Windows 11 font stack, with graceful fallback
        fams = set(tkfont.families(self.root))
        f_main = "Segoe UI Variable Text" if "Segoe UI Variable Text" in fams else "Segoe UI"
        f_disp = "Segoe UI Variable Display" if "Segoe UI Variable Display" in fams else f_main
        f_semi = "Segoe UI Semibold" if "Segoe UI Semibold" in fams else f_main
        self.F = {"main": f_main, "disp": f_disp, "semi": f_semi}

        self.root.title("hyperambient-mini")
        self.root.geometry("560x380")
        self.root.minsize(504, 342)
        self.root.attributes("-topmost", True)
        bg_flat = "#%02x%02x%02x" % BG_BOT
        self.root.configure(bg=bg_flat)
        # indigo gradient backdrop: visible in the gutters around the panels
        self.bg = tk.Canvas(self.root, highlightthickness=0, bg=bg_flat)
        self.bg.place(x=0, y=0, relwidth=1, relheight=1)
        self.bg.bind("<Configure>", self._paint_bg)
        self.root.option_add("*TCombobox*Listbox.background", CHIP)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", BLUE)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Dark.TCombobox", fieldbackground=PANEL, background=CHIP,
                        foreground=TEXT, arrowcolor=DIM, bordercolor=EDGE,
                        lightcolor=PANEL, darkcolor=PANEL, insertcolor=TEXT)

        # header: logo + wordmark + keycap chip
        head = tk.Frame(self.root, bg=bg_flat)
        head.pack(fill="x", padx=16, pady=(10, 6))
        # HY monogram: gradient-extruded letters over a cyan baseline
        logo = tk.Canvas(head, width=48, height=34, bg=bg_flat,
                         highlightthickness=0)
        logo.pack(side="left", padx=(0, 10))
        hy_font = (f_disp, 16, "bold italic")
        hy_stops = ((PINK, 0.0), (ORANGE, 0.55), (YELLOW, 1.0))
        for i in range(5, 0, -1):
            logo.create_text(6 + i, 3 + i, anchor="nw", text="HY",
                             fill=_lerp_stops(hy_stops, i / 5), font=hy_font)
        logo.create_text(6, 3, anchor="nw", text="HY", fill="#fdfdff",
                         font=hy_font)
        logo.create_line(4, 30, 46, 30, fill=CYAN, width=2)
        tk.Label(head, text="Hyperambient", bg=bg_flat, fg=TEXT,
                 font=(f_disp, 12, "bold")).pack(side="left")
        tk.Label(head, text="mini", bg=bg_flat, fg=FUCHSIA,
                 font=(f_disp, 12, "bold italic")).pack(side="left")
        tk.Label(head, text="hold to talk", bg=bg_flat, fg=DIM2,
                 font=(f_main, 9)).pack(side="right", padx=(0, 10))
        self.keycap = tk.Label(head, text=" F9 ", bg=PANEL, fg=CYAN,
                               padx=8, pady=2, font=(f_semi, 9),
                               highlightthickness=1, highlightbackground=EDGE)
        self.keycap.pack(side="right")

        self.status = tk.Label(self.root, text="starting…", anchor="w",
                               bg=bg_flat, fg=DIM, font=(f_main, 9))
        self.status.pack(fill="x", padx=16)

        # level meter
        meter = tk.Frame(self.root, bg=bg_flat)
        meter.pack(fill="x", padx=16, pady=(6, 0))
        tk.Label(meter, text="MIC", bg=bg_flat, fg=DIM2,
                 font=(f_semi, 7)).pack(side="left")
        self.meter = tk.Canvas(meter, height=8, bg=PANEL, highlightthickness=0)
        self.meter.pack(fill="x", expand=True, side="left", padx=(8, 8))
        self.meter_label = tk.Label(meter, text="", bg=bg_flat, fg=DIM,
                                    font=(f_main, 8), width=8, anchor="e")
        self.meter_label.pack(side="left")

        # transcript: canvas field (blanc cassé) with watermark + emerald text
        self.transcript = ""
        self.copied_len = 0
        self._tfont = tkfont.Font(root=self.root, font=(f_main, 13))
        self.c_field = tk.Canvas(self.root, bg=FIELD, highlightthickness=1,
                                 highlightbackground="#1d5f70",
                                 highlightcolor=CYAN)
        self.c_field.pack(fill="both", expand=True, padx=16, pady=8)
        self.c_field.bind("<Configure>", self._render_field)
        self._render_field()

        # talk button: synthwave sunset gradient on a canvas
        self.btn_talk = tk.Canvas(self.root, height=52, highlightthickness=0,
                                  bg=bg_flat, cursor="hand2")
        self.btn_talk.pack(fill="x", padx=16)
        self.btn_talk.bind("<Configure>", lambda e: self._draw_talk())
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
        bar = tk.Frame(self.root, bg=bg_flat)
        bar.pack(fill="x", padx=16, pady=(10, 0))

        def btn(label, cmd, bg, hover, fg="#ffffff"):
            b = tk.Button(bar, text=label, command=cmd, relief="flat",
                          bg=bg, fg=fg, activebackground=hover,
                          activeforeground=fg, borderwidth=0,
                          font=(f_semi, 9), padx=14, pady=6, cursor="hand2")
            b.bind("<Enter>", lambda e: b.configure(bg=hover))
            b.bind("<Leave>", lambda e: b.configure(bg=bg))
            return b

        btn("📋 Copy", self.copy_all, PINK, "#ff4d9d").pack(side="left")
        btn("copy new", self.copy_new, BLUE, "#4d6dff").pack(side="left", padx=(6, 0))
        btn("↵ newline", self.insert_newline, CYAN, "#33dcff",
            fg="#0c102c").pack(side="left", padx=(6, 0))
        btn("clear", self.clear_text, YELLOW, "#fff17a",
            fg="#0c102c").pack(side="left", padx=(6, 0))

        self.autocopy = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="auto-copy", variable=self.autocopy,
                       bg=bg_flat, fg=DIM, selectcolor=CHIP,
                       activebackground=bg_flat, activeforeground=TEXT,
                       font=(f_main, 9)).pack(side="right")

        # footer: mic + hint
        dev = tk.Frame(self.root, bg=bg_flat)
        dev.pack(fill="x", padx=16, pady=(8, 10))
        tk.Label(dev, text="mic", bg=bg_flat, fg=DIM2,
                 font=(f_main, 8)).pack(side="left")
        self.device_box = ttk.Combobox(dev, state="readonly", width=42,
                                       style="Dark.TCombobox",
                                       font=(f_main, 9))
        self.device_box.pack(side="left", padx=(8, 0))
        self.device_box.bind("<<ComboboxSelected>>", self._device_changed)
        self._fill_devices()
        tk.Label(dev, text=f"one activation = one text · "
                           f"{SHOW_HOTKEY.replace('-', '+')} show",
                 bg=bg_flat, fg=DIM2, font=(f_main, 8)).pack(side="right")

        # Space = push-to-talk while the window has focus (autorepeat-guarded)
        self.root.bind("<KeyPress-space>", self._space_press)
        self.root.bind("<KeyRelease-space>", self._space_release)
        self.root.bind("<Unmap>", self._on_unmap)
        self.root.after(150, self._poll_results)
        self._set_placeholder()

    def _talk_hover(self, on: bool) -> None:
        self._talk_hovered = on
        if not self.ptt_active:
            self._draw_talk()

    def _draw_talk(self) -> None:
        """Paint the sunset-gradient talk button; recording pulses deep pink."""
        c = self.btn_talk
        c.delete("all")
        w = max(c.winfo_width(), 200)
        h = int(c["height"])
        if self.ptt_active:
            hi = int(time.time() * 4) % 2 == 0
            stops = ((REC_A, 0.0), ("#ff5c3d", 0.55), ("#ffe74c", 1.0)) if hi \
                else ((REC_B, 0.0), ("#d94327", 0.55), ("#e0c832", 1.0))
            label = "●   Listening… release to send"
        elif self._talk_hovered:
            stops = (("#ff4d9d", 0.0), ("#ffb14d", 0.55), ("#fff17a", 1.0))
            label = "⏺   Hold F9 and speak  —  release to send"
        else:
            stops = TALK_STOPS
            label = "⏺   Hold F9 and speak  —  release to send"
        steps = 48
        for i in range(steps):
            t = i / (steps - 1)
            col = _lerp_stops(stops, t)
            x0 = int(w * i / steps)
            c.create_rectangle(x0, 0, int(w * (i + 1) / steps) + 1, h,
                               fill=col, width=0)
        c.create_text(w // 2, h // 2, text=label, fill="#ffffff",
                      font=(self.F["semi"], 12))

    def _set_placeholder(self) -> None:
        self._render_field()

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
        self._set_status()

    def _on_audio(self, raw) -> None:
        pcm = resample_to_16k(raw, self.mic_rate)
        level = rms_db_of(pcm)
        self.last_level = level
        self.zero_blocks = self.zero_blocks + 1 if level < 1.0 else 0
        if self.ptt_active:
            self.frames.append(pcm)
            self.utter_len += BLOCK_MS / 1000.0
            if self.utter_len >= MAX_UTTERANCE_S:
                self._flush_utterance()   # cap; user can keep talking
                self.frames = [pcm]
                self.utter_len = BLOCK_MS / 1000.0

    def _flush_utterance(self) -> None:
        pcm = b"".join(self.frames)
        self.frames = []
        self.utter_len = 0.0
        dur = len(pcm) / 2 / SAMPLE_RATE
        if dur >= MIN_SPEECH_S:
            self.segment_q.put(pcm)

    # ---------------- push-to-talk ----------------
    def ptt_start(self) -> None:
        if self.ptt_active or not self.mic_alive:
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
        self._talk_hovered = False
        self._draw_talk()
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
        self._draw_talk()
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
                elif kind == "show":
                    self._window_toggle()
        except queue.Empty:
            pass
        tq = getattr(self, "_tray_events", None)
        if tq is not None:
            try:
                while True:
                    ev = tq.get_nowait()
                    if ev == "toggle":
                        self._window_toggle()
                    elif ev == "menu":
                        self._show_tray_menu()
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
            self._draw_talk()

    def _draw_meter(self) -> None:
        w = self.meter.winfo_width()
        if w < 20:
            return
        self.meter.delete("all")
        frac = min(self.last_level / METER_RANGE, 1.0)
        color = ("#01cdfe" if frac < 0.4 else "#ffe74c" if frac < 0.7
                 else "#ff9e3d" if frac < 0.9 else FUCHSIA)
        if frac > 0.01:
            self.meter.create_rectangle(0, 0, int(w * frac), 8, fill=color, width=0)
        silent = self.zero_blocks > 100
        hint = "SILENT — check device / privacy" if silent else f"{self.last_level:.0f}"
        self.meter_label.configure(text=hint, fg="#ff5c8a" if silent else DIM)

    def _render_field(self, _event=None) -> None:
        """Watermark sun + wrapped transcript in emerald on blanc cassé."""
        c = self.c_field
        c.delete("all")
        w = c.winfo_width() or 400
        h = c.winfo_height() or 200
        if not self.transcript.strip():
            c.create_text(16, 14, anchor="nw", text=PLACEHOLDER,
                          fill="#93b8a2", font=(self.F["main"], 13, "italic"))
            return
        maxw = max(w - 32, 40)
        lines = []
        for para in self.transcript.split("\n"):
            cur = ""
            for word in para.split(" "):
                trial = (cur + " " + word).strip()
                if self._tfont.measure(trial) <= maxw or not cur:
                    cur = trial
                else:
                    lines.append(cur)
                    cur = word
            lines.append(cur)
        lh = self._tfont.metrics("linespace") + 4
        max_lines = max(int((h - 24) // lh), 1)
        for i, line in enumerate(lines[-max_lines:]):
            c.create_text(16, 12 + i * lh, anchor="nw", text=line,
                          fill=FIELD_TEXT, font=(self.F["main"], 13))

    def _append(self, s: str) -> None:
        self.transcript += s
        self._render_field()

    def _set_status(self, extra: str = "") -> None:
        if not self.mic_alive:
            state, color = "mic failed", "#ff8080"
        elif self.ptt_active:
            state, color = "● recording", FUCHSIA
        else:
            state = f"⏺ push-to-talk ready · hold {self.hotkey_label} anywhere"
            color = TEAL
        n, ms = self.stats["n"], self.stats["ms"]
        srv = "server ok" if self.server_ok else (
            "server DOWN — run start.bat" if self.server_ok is False else "checking server…")
        self.status.configure(
            text=f"{state}   ·   {srv}   ·   {n} utterances, last {ms:.0f} ms   {extra}",
            fg=color)

    # ---------------- buttons ----------------
    def copy_all(self, silent: bool = False) -> None:
        content = self.transcript.strip()
        if not content:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.root.update()          # win the clipboard race with the target app
        self.copied_len = len(self.transcript)
        if not silent:
            self._set_status("· copied ✓")

    def copy_new(self) -> None:
        fresh = self.transcript[self.copied_len:].strip()
        if not fresh:
            self._set_status("· nothing new")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(fresh)
        self.root.update()
        self.copied_len = len(self.transcript)
        self._set_status("· copied new ✓")

    def insert_newline(self) -> None:
        if self.transcript.strip():
            self._append("\n")

    def clear_text(self) -> None:
        self.transcript = ""
        self.copied_len = 0
        self.stats = {"n": 0, "ms": 0.0}
        self._render_field()

    # ---------------- tray ----------------
    def _init_tray(self) -> None:
        """Tray icon owned by its own thread + message pump.

        The wndproc must NEVER touch Tk directly: Tk's mainloop releases the
        GIL while pumping Windows messages, and a ctypes callback re-entering
        tkinter from inside that pump crashes the interpreter
        (PyEval_RestoreThread fatal). Events cross to the Tk thread via a
        plain queue, drained by _poll_results.
        """
        self._tray_events = queue.Queue()

        def tray_thread() -> None:
            try:
                wm_taskbar = _user32.RegisterWindowMessageW("TaskbarCreated")
                proc = TRAY_WNDPROC(self._tray_wndproc)
                wc = _WNDCLASSW()
                wc.lpfnWndProc = proc
                wc.hInstance = _kernel32.GetModuleHandleW(None)
                wc.lpszClassName = "HyperAmbientMiniTray"
                if not _user32.RegisterClassW(byref(wc)):
                    raise OSError("RegisterClassW failed")
                hwnd = _user32.CreateWindowExW(
                    0, "HyperAmbientMiniTray", "ham-tray", 0,
                    0, 0, 0, 0, None, None, wc.hInstance, None)
                nid = _NOTIFYICONDATAW()
                nid.cbSize = ctypes.sizeof(_NOTIFYICONDATAW)
                nid.hWnd = hwnd
                nid.uID = 1
                nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP
                nid.uCallbackMessage = TRAY_CALLBACK
                nid.hIcon = _make_dot_icon()
                nid.szTip = "hyperambient-mini — hold F9 to talk"
                self._tray_proc, self._tray_hwnd = proc, hwnd
                self._nid, self._wm_taskbarcreated = nid, wm_taskbar
                ok = _shell32.Shell_NotifyIconW(NIM_ADD, byref(nid))
                print(f"tray icon added: {ok}", flush=True)
                msg = wintypes.MSG()
                while _user32.GetMessageW(byref(msg), None, 0, 0) > 0:
                    _user32.TranslateMessage(byref(msg))
                    _user32.DispatchMessageW(byref(msg))
            except Exception as e:
                print(f"tray unavailable ({e}); window-only mode", flush=True)

        threading.Thread(target=tray_thread, daemon=True).start()

    def _tray_wndproc(self, hwnd, msg, wparam, lparam):
        # runs on the tray thread: queue only, no Tk calls, no prints of state
        try:
            if msg == TRAY_CALLBACK:
                if lparam in (WM_LBUTTONUP, WM_LBUTTONDBLCLK):
                    self._tray_events.put("toggle")
                elif lparam == WM_RBUTTONUP:
                    self._tray_events.put("menu")
                return 0
            if msg == self._wm_taskbarcreated:      # explorer restarted
                _shell32.Shell_NotifyIconW(NIM_ADD, byref(self._nid))
                return 0
        except Exception:
            pass
        return _user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _window_toggle(self) -> None:
        if self.root.state() == "withdrawn":
            self.root.deiconify()
            self.root.state("normal")       # deiconify alone may restore iconic
            self.root.attributes("-topmost", True)
            self.root.lift()
            self.root.focus_force()
        else:
            self._hide_to_tray()

    def _hide_to_tray(self) -> None:
        self.root.withdraw()

    def _show_tray_menu(self) -> None:
        m = tk.Menu(self.root, tearoff=0, bg=PANEL, fg=TEXT,
                    activebackground=BLUE, activeforeground="#ffffff",
                    font=(self.F["main"], 9))
        m.add_command(label="Show / hide", command=self._window_toggle)
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
        steps = 48
        for i in range(steps):
            t = i / (steps - 1)
            col = "#%02x%02x%02x" % tuple(
                int(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOT))
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
        if self._server_proc is not None:          # we spawned it, we stop it
            try:
                self._server_proc.terminate()
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
