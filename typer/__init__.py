"""MangoKeys System-Wide Linux Desktop Live Transliteration Daemon."""

from typer.config import DEFAULT_CONFIG, TyperConfig
from typer.engine import DesktopTransliterator, TransliterationResult

try:
    from typer.injector import KeystrokeInjector
    from typer.listener import KeyboardWatcher, find_keyboard_devices
except ImportError:
    KeystrokeInjector = None  # type: ignore
    KeyboardWatcher = None  # type: ignore
    find_keyboard_devices = None  # type: ignore

__all__ = [
    "DEFAULT_CONFIG",
    "TyperConfig",
    "DesktopTransliterator",
    "TransliterationResult",
    "KeystrokeInjector",
    "KeyboardWatcher",
    "find_keyboard_devices",
]
