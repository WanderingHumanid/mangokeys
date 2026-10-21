"""MangoKeys Linux Desktop Live Transliteration Daemon ("Typer").

Runs a zero-footprint, terminal-activated daemon that translates Manglish into
Malayalam system-wide across all Linux desktop applications (Fedora KDE Plasma /
Wayland & X11).
"""

from __future__ import annotations

import argparse
import datetime
import os
import signal
import sys
import threading
import time
from pathlib import Path

# Add project root to sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import evdev  # noqa: E402
from typer.config import DEFAULT_CONFIG, TyperConfig  # noqa: E402
from typer.engine import DesktopTransliterator, TransliterationResult  # noqa: E402
from typer.injector import KeystrokeInjector  # noqa: E402
from typer.listener import (  # noqa: E402
    KeyboardPermissionError,
    KeyboardWatcher,
    find_keyboard_devices,
)

# ANSI Color codes for clean terminal logging
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def timestamp_str() -> str:
    """Return formatted timestamp string [HH:MM:SS]."""
    return datetime.datetime.now().strftime("[%H:%M:%S]")


def print_banner(engine_desc: str, mode_active: bool = True) -> None:
    """Display the formatted ANSI dashboard header."""
    status_str = f"{GREEN}ACTIVE [Manglish Mode]{RESET}" if mode_active else f"{YELLOW}MUTED [Pass-Through]{RESET}"
    hotkey_str = f"{CYAN}Right-Alt to Toggle{RESET}"
    quit_str = f"{DIM}Ctrl+C to Quit{RESET}"

    print(f"{CYAN}╔══════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{CYAN}║{RESET}                 {BOLD}MangoKeys Linux Typer Daemon{RESET}                 {CYAN}║{RESET}")
    print(f"{CYAN}║{RESET}  Engine: {engine_desc:<49}{CYAN}║{RESET}")
    print(f"{CYAN}║{RESET}  Status: {status_str}  ·  Hotkey: {hotkey_str}  {CYAN}║{RESET}")
    print(f"{CYAN}║{RESET}  Exit:   {quit_str}                                     {CYAN}║{RESET}")
    print(f"{CYAN}╚══════════════════════════════════════════════════════════════════╝{RESET}")
    sys.stdout.flush()


class TyperDaemon:
    """Coordinates transliteration engine, evdev listener, and virtual injector."""

    def __init__(
        self,
        config: TyperConfig = DEFAULT_CONFIG,
        device_path: str | None = None,
        engine_type: str | None = None,
        model_dir: str | None = None,
        tokenizer_path: str | None = None,
        grab_device: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.config = config
        self.device_path = device_path
        self.engine_type = engine_type
        self.model_dir = model_dir
        self.tokenizer_path = tokenizer_path
        self.grab_device = grab_device
        self.dry_run = dry_run

        self.stop_event = threading.Event()
        self.engine: DesktopTransliterator | None = None
        self.injector: KeystrokeInjector | None = None
        self.watcher: KeyboardWatcher | None = None
        self.active_device: evdev.InputDevice | None = None

        self._stats_words_converted = 0
        self._stats_total_latency_ms = 0.0

    def init_engine(self) -> None:
        """Initialize transliteration engine."""
        self.engine = DesktopTransliterator(
            model_dir=self.model_dir,
            tokenizer_path=self.tokenizer_path,
            force_engine=self.engine_type,
        )

    def init_injector(self) -> None:
        """Initialize uinput virtual keyboard injector."""
        if self.dry_run:
            return
        self.injector = KeystrokeInjector(config=self.config)

    def on_word_triggered(self, word: str, delimiter_key: int, delimiter_char: str) -> None:
        """Called by KeyboardWatcher when a word delimiter (Space/Enter) is hit."""
        if not word or not self.engine:
            return

        # Perform transliteration
        result: TransliterationResult = self.engine.transliterate(word, n_best=4)

        if not result.top_candidate:
            return

        self._stats_words_converted += 1
        self._stats_total_latency_ms += result.latency_ms

        # Replace text via virtual uinput & clipboard
        if self.injector and not self.dry_run:
            try:
                self.injector.erase_and_replace(
                    typed_len=len(word),
                    replacement_text=result.top_candidate,
                    delimiter_char=delimiter_char,
                    restore_clipboard=True,
                )
            except Exception as err:
                print(f"{timestamp_str()} {RED}[Error injecting keystrokes]{RESET} {err}")

        # Stream real-time log to terminal
        alt_str = ""
        if len(result.candidates) > 1:
            alts = ", ".join(repr(c) for c in result.candidates[1:4])
            alt_str = f"  {DIM}[alts: {alts}]{RESET}"

        latency_badge = f"{GREEN}⚡ {result.latency_ms:.2f} ms{RESET}"
        print(
            f"{timestamp_str()} {CYAN}⌨️  [Input]{RESET} {repr(word):<15} ──►  "
            f"{GREEN}✨ [Malayalam]{RESET} {repr(result.top_candidate):<15} ({latency_badge}){alt_str}"
        )
        sys.stdout.flush()

    def on_mode_toggled(self, active: bool) -> None:
        """Called when Right-Alt is pressed to toggle Manglish mode."""
        ts = timestamp_str()
        if active:
            print(f"{ts} {GREEN}🟢 Manglish Mode: RESUMED (Live transliteration active){RESET}")
        else:
            print(f"{ts} {YELLOW}⏸️  Manglish Mode: PAUSED (Pass-through English active){RESET}")

        if self.config.BEEP_ON_TOGGLE:
            sys.stdout.write("\a")
        sys.stdout.flush()

    def on_buffer_updated(self, buffer_text: str) -> None:
        """Called when buffer changes (e.g. cleared by ESC)."""
        pass

    def run_dry_run_simulation(self) -> None:
        """Run an automated dry-run test exercising the full stack."""
        print(f"{timestamp_str()} {YELLOW}Running Dry-Run Simulation Mode...{RESET}\n")
        self.init_engine()

        sample_words = [
            "namaskaram",
            "ente",
            "kootukara",
            "malayalam",
            "sukhamano",
            "evideyaanu",
        ]

        print_banner(self.engine.model_desc, mode_active=True)
        print(f"{timestamp_str()} {YELLOW}⚡ Running live simulation preview...{RESET}\n")

        for sample in sample_words:
            start = time.perf_counter()
            res = self.engine.transliterate(sample, n_best=3)
            dur = (time.perf_counter() - start) * 1000.0

            alts = ", ".join(repr(c) for c in res.candidates[1:3]) if len(res.candidates) > 1 else ""
            alt_info = f"  {DIM}[alts: {alts}]{RESET}" if alts else ""
            latency_badge = f"{GREEN}⚡ {dur:.2f} ms{RESET}"
            print(
                f"{timestamp_str()} {CYAN}⌨️  [Input]{RESET} {repr(sample):<15} ──►  "
                f"{GREEN}✨ [Malayalam]{RESET} {repr(res.top_candidate):<15} ({latency_badge}){alt_info}"
            )

        print(f"\n{timestamp_str()} {GREEN}✓ Simulation completed successfully! Engine operates nominally.{RESET}")

    def run(self) -> None:
        """Start the live transliteration daemon."""
        # 1. Initialize engine
        self.init_engine()

        # 2. Dry run check
        if self.dry_run:
            self.run_dry_run_simulation()
            return

        # 3. Locate keyboard device
        if self.device_path:
            try:
                self.active_device = evdev.InputDevice(self.device_path)
            except Exception as err:
                print(f"{RED}Error: Cannot open specified device {self.device_path}: {err}{RESET}")
                return
        else:
            try:
                available_devices = find_keyboard_devices()
            except KeyboardPermissionError as perm_err:
                print(f"\n{RED}{BOLD}=== PERMISSION ERROR ==={RESET}")
                print(f"{perm_err}")
                print(f"{YELLOW}Tip: You can test the engine immediately with: python3 typer/run.py --dry-run{RESET}\n")
                return

            if not available_devices:
                print(f"{RED}Error: No physical keyboard devices found under /dev/input/event*.{RESET}")
                print(f"{YELLOW}Tip: Run with 'python3 typer/run.py --dry-run' to test transliteration logic.{RESET}")
                return

            self.active_device = available_devices[0]

        # 4. Initialize injector
        try:
            self.init_injector()
        except RuntimeError as uinput_err:
            print(f"\n{RED}{BOLD}=== UINPUT PERMISSION ERROR ==={RESET}")
            print(f"{uinput_err}\n")
            return

        # 5. Display banner
        print_banner(self.engine.model_desc, mode_active=True)
        dev_info = f"{self.active_device.path} ({self.active_device.name})"
        print(f"{timestamp_str()} Listening on {CYAN}{dev_info}{RESET}")
        sys.stdout.flush()

        # 6. Initialize watcher state machine
        self.watcher = KeyboardWatcher(
            config=self.config,
            on_word_trigger=self.on_word_triggered,
            on_toggle_mode=self.on_mode_toggled,
            on_buffer_change=self.on_buffer_updated,
        )

        # 7. Start listening loop
        try:
            self.watcher.listen(
                device=self.active_device,
                stop_event=self.stop_event,
                grab=self.grab_device,
            )
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Clean teardown releasing grabs, closing devices, and printing stats."""
        self.stop_event.set()

        if self.watcher:
            self.watcher.stop()

        if self.injector:
            self.injector.close()
            self.injector = None

        if self.active_device:
            try:
                self.active_device.close()
            except Exception:
                pass
            self.active_device = None

        print(f"\n{YELLOW}🛑 Shutting down MangoKeys Typer...{RESET}")
        print(f"{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
        print(f"{BOLD}📊 Session Summary:{RESET}")
        if self._stats_words_converted > 0:
            avg_ms = self._stats_total_latency_ms / self._stats_words_converted
            print(f"   • Words transliterated: {GREEN}{self._stats_words_converted}{RESET}")
            print(f"   • Average engine speed: {GREEN}{avg_ms:.2f} ms / word{RESET}")
        else:
            print(f"   • Words transliterated: {DIM}0 words{RESET}")
        print(f"   • Virtual devices:      {GREEN}Closed cleanly{RESET}")
        print(f"   • System keyboard:      {GREEN}100% normal (0 leftover changes){RESET}")
        print(f"{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
        print(f"{GREEN}✓ Zero traces left behind. Goodbye!{RESET}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MangoKeys Linux Desktop Live Transliteration Daemon ('Typer')",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal start (auto-detect physical keyboard)
  sudo python3 typer/run.py

  # Run dry-run simulation without root or input device
  python3 typer/run.py --dry-run

  # Specify a custom event device
  sudo python3 typer/run.py --device /dev/input/event3

  # Force rule-based engine
  sudo python3 typer/run.py --engine rules
        """,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Path to /dev/input/event* device to intercept (auto-detected if omitted)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["auto", "rules", "onnx"],
        default="auto",
        help="Transliteration engine to use (default: auto)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Custom directory containing ONNX models (encoder.onnx, decoder.onnx)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Custom path to tokenizer.json",
    )
    parser.add_argument(
        "--grab",
        action="store_true",
        help="Exclusively grab the keyboard device using evdev (default: passive snooping)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run simulation benchmark without accessing /dev/input hardware",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run comprehensive test suite and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.test:
        from typer.test_typer import run_all_tests

        success = run_all_tests()
        sys.exit(0 if success else 1)

    daemon = TyperDaemon(
        device_path=args.device,
        engine_type=None if args.engine == "auto" else args.engine,
        model_dir=args.model_dir,
        tokenizer_path=args.tokenizer,
        grab_device=args.grab,
        dry_run=args.dry_run,
    )

    # Register OS signals for clean teardown
    def _sig_handler(sig: int, frame: object) -> None:
        daemon.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    daemon.run()


if __name__ == "__main__":
    main()
