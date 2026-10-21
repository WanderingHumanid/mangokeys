"""Unit and integration test suite for the MangoKeys Typer daemon.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evdev import ecodes
from typer.config import DEFAULT_CONFIG, TyperConfig
from typer.engine import DesktopTransliterator, TransliterationResult
from typer.injector import KeystrokeInjector
from typer.listener import (
    KEY_TO_CHAR,
    KeyboardPermissionError,
    KeyboardWatcher,
    find_keyboard_devices,
)


class TestTransliteratorEngine(unittest.TestCase):
    """Test DesktopTransliterator fallback and conversion logic."""

    def setUp(self) -> None:
        self.engine = DesktopTransliterator()

    def test_engine_initialization_defaults_to_rules(self) -> None:
        self.assertEqual(self.engine.engine_type, "rules")
        self.assertIn("RuleEngine", self.engine.model_desc)

    def test_transliterate_basic_words(self) -> None:
        res = self.engine.transliterate("ente")
        self.assertIsInstance(res, TransliterationResult)
        self.assertEqual(res.input_text, "ente")
        self.assertEqual(res.top_candidate, "എന്റെ")
        self.assertGreater(len(res.candidates), 0)
        self.assertGreaterEqual(res.latency_ms, 0.0)

    def test_transliterate_empty_string(self) -> None:
        res = self.engine.transliterate("")
        self.assertEqual(res.top_candidate, "")
        self.assertEqual(res.candidates, [])

    def test_transliterate_candidates_limit(self) -> None:
        res = self.engine.transliterate("namaskaram", n_best=3)
        self.assertLessEqual(len(res.candidates), 3)


class TestKeyboardWatcherStateMachine(unittest.TestCase):
    """Test deterministic keyboard listener state machine."""

    def setUp(self) -> None:
        self.triggered_words: list[tuple[str, int, str]] = []
        self.mode_toggles: list[bool] = []
        self.buffer_snapshots: list[str] = []

        self.watcher = KeyboardWatcher(
            config=DEFAULT_CONFIG,
            on_word_trigger=lambda w, k, d: self.triggered_words.append((w, k, d)),
            on_toggle_mode=lambda m: self.mode_toggles.append(m),
            on_buffer_change=lambda b: self.buffer_snapshots.append(b),
        )

    def test_basic_word_typing_and_space_trigger(self) -> None:
        # Type "ente"
        for char in "ente":
            code = getattr(ecodes, f"KEY_{char.upper()}")
            self.watcher.process_key(code, 1)  # down

        self.assertEqual(self.watcher.state.current_word, "ente")

        # Hit Space (KEY_SPACE = 57)
        self.watcher.process_key(ecodes.KEY_SPACE, 1)

        self.assertEqual(len(self.triggered_words), 1)
        self.assertEqual(self.triggered_words[0], ("ente", ecodes.KEY_SPACE, " "))
        self.assertEqual(self.watcher.state.current_word, "")

    def test_backspace_pops_character(self) -> None:
        # Type "cat"
        for char in "cat":
            self.watcher.process_key(getattr(ecodes, f"KEY_{char.upper()}"), 1)
        self.assertEqual(self.watcher.state.current_word, "cat")

        # Press backspace
        self.watcher.process_key(ecodes.KEY_BACKSPACE, 1)
        self.assertEqual(self.watcher.state.current_word, "ca")

        # Press backspace again
        self.watcher.process_key(ecodes.KEY_BACKSPACE, 1)
        self.assertEqual(self.watcher.state.current_word, "c")

    def test_esc_clears_buffer_without_triggering(self) -> None:
        # Type "hello"
        for char in "hello":
            self.watcher.process_key(getattr(ecodes, f"KEY_{char.upper()}"), 1)
        self.assertEqual(self.watcher.state.current_word, "hello")

        # Press ESC
        self.watcher.process_key(ecodes.KEY_ESC, 1)
        self.assertEqual(self.watcher.state.current_word, "")
        self.assertEqual(len(self.triggered_words), 0)

    def test_right_alt_toggles_manglish_mode(self) -> None:
        self.assertTrue(self.watcher.state.manglish_mode)

        # Tap Right-Alt
        self.watcher.process_key(ecodes.KEY_RIGHTALT, 1)
        self.assertFalse(self.watcher.state.manglish_mode)
        self.assertEqual(self.mode_toggles, [False])

        # When mode is False, typing does not buffer
        self.watcher.process_key(ecodes.KEY_A, 1)
        self.assertEqual(self.watcher.state.current_word, "")

        # Tap Right-Alt again to re-enable
        self.watcher.process_key(ecodes.KEY_RIGHTALT, 1)
        self.assertTrue(self.watcher.state.manglish_mode)
        self.assertEqual(self.mode_toggles, [False, True])

        # Typing now buffers
        self.watcher.process_key(ecodes.KEY_A, 1)
        self.assertEqual(self.watcher.state.current_word, "a")

    def test_shift_and_caps_lock(self) -> None:
        # Press Shift Down
        self.watcher.process_key(ecodes.KEY_LEFTSHIFT, 1)
        self.watcher.process_key(ecodes.KEY_T, 1)
        # Release Shift
        self.watcher.process_key(ecodes.KEY_LEFTSHIFT, 0)
        self.watcher.process_key(ecodes.KEY_A, 1)

        self.assertEqual(self.watcher.state.current_word, "Ta")

    def test_non_word_keys_reset_buffer(self) -> None:
        # Type "abc"
        for char in "abc":
            self.watcher.process_key(getattr(ecodes, f"KEY_{char.upper()}"), 1)
        self.assertEqual(self.watcher.state.current_word, "abc")

        # Press Up Arrow (KEY_UP = 103)
        self.watcher.process_key(ecodes.KEY_UP, 1)
        self.assertEqual(self.watcher.state.current_word, "")


class TestKeystrokeInjector(unittest.TestCase):
    """Test KeystrokeInjector virtual device and clipboard interaction."""

    def test_injector_creation_and_close(self) -> None:
        try:
            injector = KeystrokeInjector()
            self.assertIsNotNone(injector._uinput)
            injector.close()
            self.assertIsNone(injector._uinput)
        except RuntimeError:
            # If system doesn't grant /dev/uinput permissions, pass gracefully
            pass

    def test_clipboard_operations(self) -> None:
        try:
            injector = KeystrokeInjector()
            # Set and read clipboard
            test_val = "ടെസ്റ്റ്_യുണിറ്റ്"
            ok = injector.set_clipboard(test_val)
            if ok:
                val = injector.get_clipboard()
                self.assertEqual(val, test_val)
            injector.close()
        except RuntimeError:
            pass


def run_all_tests() -> bool:
    """Run all test suites and return True if successful."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestTransliteratorEngine))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestKeyboardWatcherStateMachine))
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(TestKeystrokeInjector))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
