# hyperambient-mini

Light version of [hyper-ambient](https://github.com/tbrignonen-ai/hyper-ambient):
only the ears. **Hold F9, speak, release — the text is in your clipboard.**
Paste anywhere: ZCode, the DeepSeek desktop app, anything.

Local from end to end: your voice never leaves the machine. CPU-only
(~400 MB RAM, **0 GPU**), safe to run next to a training job.

## Quickstart (Windows)

1. Download **`HyperAmbientMini.exe`** from
   [Releases](https://github.com/tbrignonen-ai/hyper-ambient-mini/releases).
2. Double-click. **First run downloads what it needs** — the STT runtime
   (2 MB) and the model (347 MB), once — with progress shown in the window.
3. Hold **F9** anywhere, speak, release → **Ctrl+V** in whatever you were
   typing. One activation = one text: the box and clipboard reset on each
   press, so you always paste exactly what you just said.

| | |
|---|---|
| **F9** (hold) | talk, from any app |
| **Ctrl+Alt+H** | show / hide the window |
| minimize | hides to the tray (pink dot) |
| close (X) or tray → Quit | quits app + server |

English-focused; French comes out usable but is not trained or measured.

## Run from source

```sh
git clone https://github.com/tbrignonen-ai/hyper-ambient-mini
cd hyper-ambient-mini
pip install -r requirements.txt     # sounddevice only
python app/mini_harness.py          # same first-run download as the exe
```

`python app/mini_harness.py --selftest` checks the server round-trip without
the GUI.

## What's inside

| Piece | What |
|---|---|
| **Ears** | [Fluid Speech Mini](https://huggingface.co/FluidVoice-Labs/fluid-speech-mini-gguf) — fine-tuned Parakeet TDT 0.6B, 347 MB, 4.53 % WER on the Open ASR Leaderboard sets (better than whisper-large-v3), robust when another voice is talking in the room |
| **Runtime** | [parakeet.cpp](https://github.com/mudler/parakeet.cpp) v0.6.1 server, CPU build — 0 GB VRAM |
| **App** | tkinter + sounddevice; energy meter, push-to-talk only (no VAD), junk-hallucination filter, tray icon, clipboard auto-copy |
| ZCode extra | a user-scope `/heard` slash command reads `transcript.txt` straight into a ZCode conversation |

Rebuild the exe: `python gen_icon.py && pyinstaller --onefile --noconsole
--name HyperAmbientMini --icon icon.ico --collect-all sounddevice app/mini_harness.py`

## Attribution

Fluid Speech Mini by FluidVoice (CC BY 4.0), built on NVIDIA Parakeet TDT 0.6B v3
and Moondream Parakeet Ultra (CC BY 4.0). Runtime: parakeet.cpp (MIT).
This repo's code: Apache 2.0. Inspired by the architecture of hyper-ambient
(energy VAD before inference, server/host split, presence window).
