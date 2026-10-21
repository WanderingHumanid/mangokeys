# MangoKeys Typer: System-Wide Linux Desktop Live Transliteration

A zero-install, terminal-activated daemon that translates Manglish into Malayalam system-wide across all Linux desktop applications (Fedora KDE Plasma / GNOME / Wayland and X11).

```text
╔══════════════════════════════════════════════════════════════════╗
║                 MangoKeys Linux Typer Daemon                 ║
║  Engine: RuleEngine (Deterministic Baseline)                     ║
║  Status: ACTIVE [Manglish Mode]  ·  Hotkey: Right-Alt to Toggle  ║
║  Exit:   Ctrl+C to Quit                                          ║
╚══════════════════════════════════════════════════════════════════╝
[23:42:01] Listening on /dev/input/event3 (AT Translated Set 2 keyboard)
[23:42:05] [Input] 'namaskaram'  ──►  [Malayalam] 'നമസ്കാരം'      (0.08 ms)
[23:42:09] [Input] 'ente'        ──►  [Malayalam] 'എന്റെ'         (0.02 ms)
[23:42:14] [Input] 'kootukara'   ──►  [Malayalam] 'കൂട്ടുകാരാ'     (0.04 ms)
```

---

## 1. Zero System Modification Guarantee

* **No System Package Modifications**: No root packages installed into `/usr`, no modifications to `/etc/` or user profile dotfiles.
* **No Background Systemd Service**: Activated only when you open a terminal and run `python3 typer/run.py`.
* **Zero Leftover State**: Pressing `Ctrl+C` or closing the terminal window immediately destroys virtual devices, unhooks evdev listeners, and completely restores normal system keyboard behavior.

---

## 2. Requirements & Permissions

### System Tools
* **Python 3.10+**
* **evdev** Python package (`python3-evdev`)
* **wl-clipboard** (`wl-copy` / `wl-paste` on Wayland) or **xclip** / **xsel** (on X11)

### Input Device Permissions
Linux protects hardware keystroke devices under `/dev/input/event*` for security. To allow the daemon to detect your keyboard, choose either:

1. **Option A (Recommended for Quick Use)**: Run with `sudo`:
   ```bash
   sudo python3 typer/run.py
   ```
2. **Option B (Run without sudo)**: Add your user to the `input` group:
   ```bash
   sudo usermod -aG input $USER
   ```
   *(Log out and log back in for the group membership to take effect)*.

---

## 3. Quick Start

### Start Daemon
```bash
sudo python3 typer/run.py
```

### Dry-Run Simulation (No Root / No Hardware Required)
Test the transliteration engine and simulated keystroke pipeline immediately:
```bash
python3 typer/run.py --dry-run
```

### Run Self-Tests
Verify the state machine, buffer management, hotkeys, and transliteration engine:
```bash
python3 typer/run.py --test
```

---

## 4. Hotkeys & Interaction

| Key | Action | Description |
| :--- | :--- | :--- |
| **`Right-Alt`** | **Toggle Mode** | Single tap toggles between **Manglish Mode** and **Pass-Through (English) Mode** on the fly. Emits audio/terminal chime. |
| **`Space`** | **Inline Transliterate** | Converts the accumulated Latin word to Malayalam and replaces it inline in your active app. |
| **`Enter`** | **Inline Transliterate** | Converts the accumulated Latin word to Malayalam upon newline. |
| **`Backspace`** | **Pop Character** | Keeps the internal buffer synchronized as you edit characters. |
| **`Esc`** | **Cancel Word** | Clears the word buffer without triggering transliteration. |
| **`Ctrl+C`** | **Clean Quit** | Unhooks listeners, destroys virtual devices, prints stats, and exits cleanly. |

---

## 5. Architecture

* **`typer/run.py`**: CLI entry point with ANSI dashboard and graceful `SIGINT`/`SIGTERM` handlers.
* **`typer/listener.py`**: Intercepts `/dev/input/event*` devices using `evdev`, filters for physical keyboards, and drives the typing state machine.
* **`typer/injector.py`**: High-speed Wayland/X11 keystroke erasure via `uinput` backspaces + atomic Unicode clipboard paste (`wl-copy` / `Ctrl+V`), with background clipboard restoration.
* **`typer/engine.py`**: `DesktopTransliterator` wrapping `model.src.rules.RuleEngine` for instant 0.05 ms zero-dependency transliteration, with automatic graceful upgrade to neural ONNX seq2seq when weights exist.
* **`typer/config.py`**: Centralized keycodes, debounce timings, and delimiter settings.

---

## 6. Command-Line Options

```text
usage: run.py [-h] [--device DEVICE] [--engine {auto,rules,onnx}]
              [--model-dir MODEL_DIR] [--tokenizer TOKENIZER] [--grab]
              [--dry-run] [--test]

options:
  -h, --help            Show this help message and exit
  --device DEVICE       Path to /dev/input/event* device to intercept (auto-detected if omitted)
  --engine {auto,rules,onnx}
                        Transliteration engine to use (default: auto)
  --model-dir MODEL_DIR Custom directory containing ONNX models (encoder.onnx, decoder.onnx)
  --tokenizer TOKENIZER Custom path to tokenizer.json
  --grab                Exclusively grab the keyboard device using evdev (default: passive snooping)
  --dry-run             Run simulation benchmark without accessing /dev/input hardware
  --test                Run comprehensive test suite and exit
```

---

## 7. Troubleshooting

* **`Permission denied accessing /dev/input/event*`**:
  Run with `sudo python3 typer/run.py`, or add your user account to the `input` group via `sudo usermod -aG input $USER`.
* **KDE Plasma Wayland Clipboard**:
  Ensure `wl-clipboard` is installed (`sudo dnf install wl-clipboard` on Fedora). The injector uses `wl-copy -n` to inject exact Unicode strings directly into the focused window without dead-key or layout distortion.
* **Exiting the Daemon**:
  Simply press `Ctrl+C` in the terminal where `run.py` is running. All grabs and virtual devices are closed automatically.
