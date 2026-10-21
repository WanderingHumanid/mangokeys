"""Linux evdev keyboard listener and keystroke state machine.

Intercepts hardware keyboard events under /dev/input/event*, tracks typed Latin
words, detects delimiter triggers (Space, Enter), and handles the Right-Alt toggle.
"""

from __future__ import annotations

import os
import select
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import evdev
from evdev import ecodes

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typer.config import DEFAULT_CONFIG, TyperConfig  # noqa: E402

# Build static mapping of evdev keycodes to [a-z]
KEY_TO_CHAR: dict[int, str] = {
    getattr(ecodes, f"KEY_{chr(c)}"): chr(c).lower()
    for c in range(ord("A"), ord("Z") + 1)
    if hasattr(ecodes, f"KEY_{chr(c)}")
}

# Delimiters that trigger word conversion
TRIGGER_KEYS: set[int] = {
    ecodes.KEY_SPACE,
    ecodes.KEY_ENTER,
}


class KeyboardPermissionError(PermissionError):
    """Raised when the process lacks permission to read /dev/input devices."""


def find_keyboard_devices() -> list[evdev.InputDevice]:
    """Scan and return all active physical hardware keyboard devices.

    Filters out mice, power buttons, and synthetic virtual keyboards.
    Raises KeyboardPermissionError if /dev/input/event* cannot be read.
    """
    dev_paths = evdev.list_devices()
    if not dev_paths:
        # Check if /dev/input exists and has permissions
        if os.path.exists("/dev/input"):
            events = [f for f in os.listdir("/dev/input") if f.startswith("event")]
            if events:
                # Event files exist but evdev couldn't open them -> permission error
                raise KeyboardPermissionError(
                    "Permission denied accessing /dev/input/event*.\n"
                    "Fix this by either:\n"
                    "  1) Running with sudo: sudo python3 typer/run.py\n"
                    "  2) Or adding your user to the input group:\n"
                    f"     sudo usermod -aG input {os.environ.get('USER', '$USER')}\n"
                    "     (and logging out and back in)"
                )
        return []

    keyboards: list[evdev.InputDevice] = []
    permission_denied_count = 0

    for path in dev_paths:
        try:
            device = evdev.InputDevice(path)
        except PermissionError:
            permission_denied_count += 1
            continue
        except Exception:
            continue

        # Ignore our own virtual keyboard or other uinput devices
        name_lower = device.name.lower()
        if "mangokeys" in name_lower or "uinput" in name_lower:
            continue

        caps = device.capabilities()
        if ecodes.EV_KEY not in caps:
            continue

        supported_keys = set(caps[ecodes.EV_KEY])

        # A real keyboard must have most alphabet keys, Space, and Enter
        alpha_count = sum(1 for code in KEY_TO_CHAR if code in supported_keys)
        has_space = ecodes.KEY_SPACE in supported_keys
        has_enter = ecodes.KEY_ENTER in supported_keys

        if alpha_count >= 20 and has_space and has_enter:
            keyboards.append(device)

    if not keyboards and permission_denied_count > 0:
        raise KeyboardPermissionError(
            "Permission denied accessing /dev/input/event* devices.\n"
            "Please run with 'sudo python3 typer/run.py' or add your user to the 'input' group."
        )

    # Sort keyboards so built-in AT keyboard / standard USB keyboards are first
    def _rank(dev: evdev.InputDevice) -> int:
        name = dev.name.lower()
        if "at translated" in name:
            return 0
        if "keyboard" in name and "usb" in name:
            return 1
        if "keyboard" in name:
            return 2
        return 3

    keyboards.sort(key=_rank)
    return keyboards


@dataclass
class TypingState:
    """Internal state machine tracking buffer, active mode, and modifier keys."""

    buffer: list[str] = field(default_factory=list)
    manglish_mode: bool = True
    shift_pressed: bool = False
    caps_lock: bool = False

    @property
    def current_word(self) -> str:
        return "".join(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()

    def pop(self) -> str | None:
        return self.buffer.pop() if self.buffer else None

    def push(self, char: str) -> None:
        self.buffer.append(char)


class KeyboardWatcher:
    """Monitors raw evdev keyboard events and drives the word typing state machine."""

    def __init__(
        self,
        config: TyperConfig = DEFAULT_CONFIG,
        on_word_trigger: Callable[[str, int, str], None] | None = None,
        on_toggle_mode: Callable[[bool], None] | None = None,
        on_buffer_change: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.state = TypingState(manglish_mode=True)
        self.on_word_trigger = on_word_trigger
        self.on_toggle_mode = on_toggle_mode
        self.on_buffer_change = on_buffer_change
        self._running = False

    def process_key(self, code: int, value: int) -> None:
        """Process a single key event (code and value: 0=up, 1=down, 2=repeat).

        Deterministic logic separated from I/O for direct testability.
        """
        # 1. Hotkey: Right-Alt single-tap toggles Manglish Mode ON / OFF
        if code == self.config.KEY_RIGHTALT:
            if value == 1:  # Only toggle on initial press down, ignore repeat/up
                self.state.manglish_mode = not self.state.manglish_mode
                self.state.clear()
                if self.on_toggle_mode:
                    self.on_toggle_mode(self.state.manglish_mode)
                if self.on_buffer_change:
                    self.on_buffer_change("")
            return

        # 2. Track modifier keys (Shift and CapsLock) regardless of active mode
        if code in (self.config.KEY_LEFTSHIFT, self.config.KEY_RIGHTSHIFT):
            self.state.shift_pressed = value in (1, 2)
            return

        if code == self.config.KEY_CAPSLOCK:
            if value == 1:
                self.state.caps_lock = not self.state.caps_lock
            return

        # If Manglish mode is disabled (pass-through English mode), do not buffer
        if not self.state.manglish_mode:
            return

        # Key release events for non-modifiers can be ignored
        if value == 0:
            return

        # 3. Handle ESC: clears buffer without triggering transliteration
        if code == self.config.KEY_ESC:
            if self.state.buffer:
                self.state.clear()
                if self.on_buffer_change:
                    self.on_buffer_change("")
            return

        # 4. Handle Backspace: pop last character
        if code == self.config.KEY_BACKSPACE:
            if self.state.buffer:
                self.state.pop()
                if self.on_buffer_change:
                    self.on_buffer_change(self.state.current_word)
            return

        # 5. Handle Delimiters (Space, Enter)
        if code in self.config.DELIMITERS:
            delimiter_char = self.config.DELIMITERS[code]
            if self.state.buffer:
                word = self.state.current_word
                self.state.clear()
                if self.on_buffer_change:
                    self.on_buffer_change("")
                if self.on_word_trigger:
                    self.on_word_trigger(word, code, delimiter_char)
            return

        # 6. Handle Alphabetic characters [a-zA-Z]
        if code in KEY_TO_CHAR:
            base_char = KEY_TO_CHAR[code]
            is_upper = self.state.shift_pressed ^ self.state.caps_lock
            char = base_char.upper() if is_upper else base_char

            if len(self.state.buffer) < self.config.MAX_WORD_LENGTH:
                self.state.push(char)
                if self.on_buffer_change:
                    self.on_buffer_change(self.state.current_word)
            return

        # 7. Non-word characters (arrows, symbols, numbers, tab, navigation)
        # Reset word buffer to avoid corrupting text replacement if cursor moved
        if self.state.buffer:
            self.state.clear()
            if self.on_buffer_change:
                self.on_buffer_change("")

    def listen(
        self,
        device: evdev.InputDevice,
        stop_event: threading.Event | None = None,
        grab: bool = False,
    ) -> None:
        """Listen to keyboard events from the specified device until stopped."""
        self._running = True
        grabbed = False

        if grab:
            try:
                device.grab()
                grabbed = True
            except Exception:
                grabbed = False

        try:
            while self._running and (stop_event is None or not stop_event.is_set()):
                # Use select with timeout so we can periodically check stop_event
                r, _, _ = select.select([device.fd], [], [], self.config.INPUT_POLL_TIMEOUT)
                if not r:
                    continue

                for event in device.read():
                    if event.type == ecodes.EV_KEY:
                        self.process_key(event.code, event.value)
        finally:
            if grabbed:
                try:
                    device.ungrab()
                except Exception:
                    pass
            self._running = False

    def stop(self) -> None:
        """Signal the listener loop to stop."""
        self._running = False
