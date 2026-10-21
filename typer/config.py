"""Configuration settings and constants for the MangoKeys Typer daemon.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TyperConfig:
    """Runtime configuration for the desktop transliteration daemon."""

    # Hotkeys (Linux evdev keycodes)
    # 100 = KEY_RIGHTALT, 1 = KEY_ESC, 14 = KEY_BACKSPACE, 57 = KEY_SPACE, 28 = KEY_ENTER
    KEY_RIGHTALT: int = 100
    KEY_ESC: int = 1
    KEY_BACKSPACE: int = 14
    KEY_SPACE: int = 57
    KEY_ENTER: int = 28
    KEY_LEFTSHIFT: int = 42
    KEY_RIGHTSHIFT: int = 54
    KEY_CAPSLOCK: int = 58

    # Delimiter characters mapped from keycodes
    DELIMITERS: dict[int, str] = None  # Initialized in __post_init__

    # Timing and Debounce settings (seconds)
    BACKSPACE_DELAY: float = 0.003
    PASTE_DELAY: float = 0.025
    RESTORE_CLIPBOARD_DELAY: float = 0.50
    INPUT_POLL_TIMEOUT: float = 0.05

    # Audio/Visual feedback
    BEEP_ON_TOGGLE: bool = True

    # Buffer constraints
    MAX_WORD_LENGTH: int = 64

    # Delimiter erasing: in passive snooping mode on Linux desktop,
    # the physical Space key has already hit the focused window by the time
    # our trigger fires. Erasing typed_len + 1 characters removes both the
    # trailing Space and the typed Latin word before pasting Malayalam + Space.
    ERASE_DELIMITER: bool = True

    # Virtual keyboard device name
    VIRTUAL_DEVICE_NAME: str = "mangokeys-virtual-keyboard"

    # Default paths for ML model search (relative to repository root)
    ONNX_SEARCH_DIRS: tuple[str, ...] = (
        "assets/release/model",
        "runs/demo/onnx-int8",
        "runs/ml_mini/onnx",
        "assets/release/prefix",
    )
    TOKENIZER_SEARCH_PATHS: tuple[str, ...] = (
        "assets/release/model/tokenizer.json",
        "runs/demo/onnx-int8/tokenizer.json",
        "runs/ml_mini/tokenizer.json",
    )

    # Visualizer integration settings
    VISUALIZER_HOST: str = "127.0.0.1"
    VISUALIZER_PORT: int = 5000
    BROADCAST_TO_VISUALIZER: bool = True

    def __post_init__(self) -> None:
        if self.DELIMITERS is None:
            self.DELIMITERS = {
                self.KEY_SPACE: " ",
                self.KEY_ENTER: "\n",
            }


# Default global configuration instance
DEFAULT_CONFIG = TyperConfig()
