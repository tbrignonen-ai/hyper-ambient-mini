# hyperambient-mini

Light version of [hyper-ambient](https://github.com/tbrignonen-ai/hyper-ambient):
only the ears. Speak, watch the text appear, click copy, paste anywhere —
ZCode, the DeepSeek desktop app, anything.

- **STT model:** [Fluid Speech Mini](https://huggingface.co/FluidVoice-Labs/fluid-speech-mini-gguf) (GGUF, released 2026-10-13) — fine-tuned Parakeet TDT 0.6B, 347 MB, 4.53 % WER on the Open ASR Leaderboard sets (better than whisper-large-v3), and specifically robust when another voice is talking in the room. English-focused; the 25-language vocabulary carries usable French (verified on a real clip), but French is not trained/measured — treat it as a bonus.
- **Runtime:** [parakeet.cpp](https://github.com/mudler/parakeet.cpp) v0.6.1 server, **CPU only, 4 threads → 0 GB VRAM, ~390 MB RAM**. Safe next to a running training job.
- **GUI:** tkinter + sounddevice. Always-on-top window, live transcript, `Copy new` / `Copy all`, auto-copy after each utterance, pause button. Mirrors everything to `transcript.txt`.

## Use

1. `start.bat` (starts the server minimized if needed, then the window).
2. **Push-to-talk (default): hold Ctrl+Alt+Space anywhere** — ZCode, the DeepSeek
   app, anything — speak, release. Or hold the on-screen TALK button / Space
   when the window is focused. (Change the combo: `set HAM_HOTKEY=ctrl-alt-v`
   before launching, or any key like `f8`.)
3. Paste (Ctrl+V) into ZCode / the DeepSeek app. With auto-copy on, the whole
   current text is already in the clipboard after every utterance.
   Auto VAD mode (hands-free, a ~0.75 s pause ends an utterance) is a mode
   toggle in the window; the level meter shows the mic level and its trigger.

Python deps: `pip install sounddevice` (only one; everything else is stdlib).

## Files

| Path | Role |
|---|---|
| `start.bat` | launcher (server + GUI) |
| `app/mini_harness.py` | the whole app; `python app/mini_harness.py --selftest` checks the server round-trip without the GUI |
| `models/` | GGUF weights (gitignored, 347 MB) |
| `bin/` | parakeet.cpp v0.6.1 win-cpu binaries (gitignored) |
| `transcript.txt` | rolling mirror of everything heard (gitignored) |

## Attribution

Fluid Speech Mini by FluidVoice (CC BY 4.0), built on NVIDIA Parakeet TDT 0.6B v3
and Moondream Parakeet Ultra (CC BY 4.0). Runtime: parakeet.cpp (MIT).
Inspired by the architecture of hyper-ambient (energy VAD before inference,
server/host split, presence window).

## Pasting into ZCode / DeepSeek desktop

- **Anywhere (DeepSeek app etc.):** the window is always-on-top with `auto-copy`
  on by default — after each utterance the full current text is already in the
  clipboard; just Ctrl+V in the target app. `Copy new` copies only what was
  said since your last copy.
- **ZCode extra:** a user-scope `/heard` command is installed
  (`~/.zcode/commands/heard.md`). Type `/heard` in ZCode and it reads
  `transcript.txt` directly into the conversation — no clipboard needed.
