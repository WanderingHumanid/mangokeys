# MangoKeys Typer: Architecture & User Guide

A zero-install, terminal-activated daemon providing system-wide live Manglish-to-Malayalam transliteration across all Linux desktop applications (Fedora KDE Plasma, GNOME, Wayland, and X11).

---

## 1. Architectural Philosophy & Mission

### The Core Objective
Enable users to type phonetic Manglish (e.g., `ente`, `namaskaram`) into **any** desktop application—such as Firefox, Chrome, Slack, Discord, VS Code, or terminal emulators—and upon pressing **`Space`** or **`Enter`**, have the typed Latin characters instantaneously erased and replaced inline with authentic Malayalam script (`എന്റെ`, `നമസ്കാരം`).

### Invariant Design Principles
1. **Zero Permanent System Modification**:
   - No root packages installed into `/usr` or `/etc`.
   - No persistent `systemd` user or root background daemons.
   - Never alters active XKB keyboard layouts, IBus, or Fcitx configurations.
2. **Terminal-Bound Lifecycle**:
   - The daemon lives strictly within the terminal session that launched it.
   - Pressing `Ctrl+C` immediately destroys the virtual `/dev/uinput` device, unhooks event descriptors, and restores normal keyboard operation without leftover state.
3. **Multi-Keyboard Concurrency**:
   - Automatically discovers and listens to all active physical keyboards concurrently using asynchronous kernel event polling.

---

## 2. End-to-End Execution Flow

```
                      +---------------------------------------+
                      |       User Hardware Keyboard(s)       |
                      +---------------------------------------+
                                          |
                                          | (raw Linux evdev events)
                                          v
                      +---------------------------------------+
                      |         listener.py (evdev)           |
                      |  * Multi-device select.select() loop  |
                      |  * Typing State Machine               |
                      |  * Delimiter & Hotkey Interceptor     |
                      +---------------------------------------+
                                          |
                        [Accumulated Word on Delimiter]
                                          |
                                          v
                      +---------------------------------------+
                      |        engine.py (Transliterator)     |
                      |  * Fast RuleEngine (0.05 ms latency)  |
                      |  * Neural ONNX Seq2Seq fallback       |
                      +---------------------------------------+
                                          |
                          [Transliterated Malayalam Text]
                                          |
                                          v
                      +---------------------------------------+
                      |         injector.py (uinput)          |
                      |  1. Stash current system clipboard   |
                      |  2. Stage Malayalam text (wl-copy)    |
                      |  3. Emit atomic Backspaces (uinput)   |
                      |  4. Emit Ctrl+V Paste (uinput)        |
                      |  5. Asynchronously restore clipboard  |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Focused Application Window (Wayland)  |
                      +---------------------------------------+
```

---

## 3. Subsystem Breakdown

### 3.1 Concurrent Hardware Keyboard Discovery (`listener.py`)
- **Capability Bitmask Filtering**: Scans all `/dev/input/event*` devices, inspects kernel bitmasks for `EV_KEY`, and verifies presence of alphanumeric typing keys (`KEY_A`–`KEY_Z`, `KEY_SPACE`, `KEY_ENTER`).
- **Feedback Loop Prevention**: Automatically ignores mice, power switches, and virtual keyboards belonging to MangoKeys or `uinput`.
- **Multiplexed Polling**: Uses a non-blocking `select.select()` poll loop to monitor multiple physical keyboards (e.g., built-in laptop keyboard and external USB/Bluetooth mechanical keyboard) simultaneously.

### 3.2 Typing State Machine (`listener.py`)
- **Word Buffer**: Accumulates printable Latin characters while tracking modifier keys (`Shift`, `Caps Lock`).
- **Retroflex Consonant Mapping**: Preserves case states so capitalized letters accurately map to Malayalam retroflex consonants (`T` $\to$ ട, `D` $\to$ ഡ, `N` $\to$ ണ, `L` $\to$ ള, `R` $\to$ റ).
- **Buffer Invalidation**: Automatically clears when non-word keys (arrow keys, mouse clicks, page up/down) are pressed to prevent accidental out-of-context injections.

### 3.3 Atomic Keystroke Replacement (`injector.py`)
- **The Indic Unicode Injection Challenge**: Standard keyboard controllers cannot emit arbitrary Unicode code points for complex Indic scripts directly via keycodes because layouts lack keys for multi-byte conjuncts and viramas.
- **The Clipboard Injection Solution**:
  1. *Stash*: Captures existing clipboard content via `wl-paste` (Wayland) or `xclip` / `xsel` (X11).
  2. *Stage*: Copies the transliterated Malayalam text to the clipboard.
  3. *Erase*: Dispatches virtual `KEY_BACKSPACE` events through `/dev/uinput` matching the exact length of the typed word plus delimiter.
  4. *Paste*: Dispatches virtual `Ctrl+V` through `/dev/uinput`.
  5. *Restore*: A background timer restores the user's previous clipboard contents after 200 ms.
- **Pipe Deadlock Prevention**: Subprocess calls are isolated with closed standard file descriptors (`DEVNULL`), preventing background Wayland clipboard brokers from deadlocking the process.

### 3.4 Transliteration Engine (`engine.py`)
- **Deterministic Baseline (`RuleEngine`)**: Shipped with zero external dependencies, executing in ~0.05 ms per word.
- **Neural ONNX Upgrade**: Seamlessly loads `encoder.onnx` and `decoder.onnx` character-level Transformer models from `assets/release/model/` when ONNX Runtime is available.

---

## 4. Hardware Permissions & Security

Linux protects `/dev/input/event*` and `/dev/uinput` from unprivileged snooping. MangoKeys provides two clean methods for access:

### Option A: POSIX Access Control Lists (Recommended)
Grant hardware permissions directly to your desktop user without needing to run the daemon as root:
```bash
sudo setfacl -m u:$USER:rw /dev/uinput /dev/input/event*
```
*Note: The unified `launcher.py` script automatically configures this with a single prompt.*

### Option B: Direct Sudo
Run the script directly with root privileges:
```bash
sudo python3 typer/run.py
```

---

## 5. Hotkeys & Operational Controls

| Hotkey / Trigger | Action | Behavioral Description |
| :--- | :--- | :--- |
| **`Right-Alt`** | **Mode Toggle** | Instantly switches between **Manglish Mode** and **Pass-Through (English) Mode**. Plays a terminal bell sound. |
| **`Space`** | **Word Commit** | Transliterates the buffered word and replaces it inline with trailing space. |
| **`Enter`** | **Newline Commit** | Transliterates the buffered word and preserves the newline. |
| **`Backspace`** | **Buffer Edit** | Removes the last character from the buffer to keep synchronization. |
| **`Esc`** | **Cancel Word** | Clears the word buffer without triggering transliteration. |
| **`Ctrl+C`** | **Clean Teardown** | Destroys virtual devices, closes device handles, prints session statistics, and exits. |

---

## 6. How to Run

### Method 1: Via the Unified Launcher (Recommended)
Launch via the interactive terminal dashboard:
```bash
python3 launcher.py
```
Select **Option 1 (Typer Daemon)** or pass the headless flag:
```bash
python3 launcher.py --typer
```

### Method 2: Direct Execution
```bash
# Direct run (requires ACL or sudo)
python3 typer/run.py

# With custom ONNX models
python3 typer/run.py --engine onnx --model-dir assets/release/model

# Dry-run benchmark simulation (no root/hardware required)
python3 typer/run.py --dry-run
```

### Method 3: Automated Self-Tests
Verify the state machine, buffer management, hotkeys, and transliteration engine:
```bash
python3 typer/test_typer.py
```
