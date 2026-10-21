"""Keystroke erasure and Unicode text injector for Linux Wayland & X11 desktops.

Uses:
1. evdev.UInput virtual keyboard to emit KEY_BACKSPACE bursts and Ctrl+V.
2. wl-copy (Wayland) / xclip (X11) to inject pure Unicode text via clipboard without layout issues.
3. Background clipboard restoration so the user's prior copied content remains intact.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import evdev
from evdev import ecodes

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typer.config import DEFAULT_CONFIG, TyperConfig  # noqa: E402


class KeystrokeInjector:
    """Manages virtual keyboard keystroke emission and clipboard replacement."""

    def __init__(self, config: TyperConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._uinput: evdev.UInput | None = None

        # Resolve target desktop user & uid for clipboard operations under sudo
        self._target_user = os.environ.get("SUDO_USER") or os.environ.get("USER") or "vishnunandan555"
        self._target_uid = os.environ.get("SUDO_UID") or str(os.getuid())
        if self._target_uid == "0" and os.path.exists("/run/user/1000"):
            self._target_uid = "1000"

        # Check Wayland presence (environment variable or active session socket)
        wayland_socket = Path(f"/run/user/{self._target_uid}/wayland-0")
        has_wayland_socket = wayland_socket.exists() or bool(os.environ.get("WAYLAND_DISPLAY"))
        self._has_wayland = bool(has_wayland_socket and shutil.which("wl-copy"))
        self._has_xclip = bool(shutil.which("xclip"))
        self._has_xsel = bool(shutil.which("xsel"))
        self._restore_timer: threading.Timer | None = None
        self._lock = threading.Lock()

        self._init_uinput()

    def _init_uinput(self) -> None:
        """Create the virtual keyboard device via /dev/uinput."""
        capabilities = {
            ecodes.EV_KEY: [
                ecodes.KEY_BACKSPACE,
                ecodes.KEY_LEFTCTRL,
                ecodes.KEY_V,
                ecodes.KEY_LEFTSHIFT,
                ecodes.KEY_INSERT,
                ecodes.KEY_SPACE,
                ecodes.KEY_ENTER,
            ]
        }
        try:
            self._uinput = evdev.UInput(
                events=capabilities,
                name=self.config.VIRTUAL_DEVICE_NAME,
                bustype=ecodes.BUS_USB,
            )
        except Exception as err:
            raise RuntimeError(
                f"Failed to create virtual input device on /dev/uinput: {err}\n"
                f"Ensure your user has access to /dev/uinput or run with sudo."
            ) from err

    # ----------------------------------------------------------- Clipboard Ops

    def _run_clipboard_cmd(
        self,
        base_cmd: list[str],
        input_text: str | None = None,
        timeout: float = 0.25,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Execute a clipboard command with proper Wayland user session environment."""
        env = os.environ.copy()
        if "XDG_RUNTIME_DIR" not in env:
            env["XDG_RUNTIME_DIR"] = f"/run/user/{self._target_uid}"
        if "WAYLAND_DISPLAY" not in env:
            env["WAYLAND_DISPLAY"] = "wayland-0"

        # When setting clipboard with wl-copy, wl-copy forks a background daemon to serve
        # clipboard data. If capture_output=True is passed, Python waits forever on the
        # inherited stdout/stderr pipes, causing a timeout and killing wl-copy!
        # Therefore, we only capture output when reading (wl-paste), not when copying.
        stdout_dest = subprocess.PIPE if capture_output else subprocess.DEVNULL
        stderr_dest = subprocess.PIPE if capture_output else subprocess.DEVNULL

        if os.geteuid() == 0 and self._target_user and self._target_user != "root":
            cmd = [
                "sudo", "-u", self._target_user,
                f"XDG_RUNTIME_DIR={env['XDG_RUNTIME_DIR']}",
                f"WAYLAND_DISPLAY={env['WAYLAND_DISPLAY']}",
            ] + base_cmd
            try:
                proc = subprocess.run(
                    cmd,
                    input=input_text,
                    text=True,
                    stdout=stdout_dest,
                    stderr=stderr_dest,
                    timeout=timeout,
                    close_fds=True,
                )
            except Exception:
                fallback = [
                    "runuser", "-u", self._target_user,
                    "-w", "XDG_RUNTIME_DIR,WAYLAND_DISPLAY,PATH", "--",
                    "env",
                    f"XDG_RUNTIME_DIR={env['XDG_RUNTIME_DIR']}",
                    f"WAYLAND_DISPLAY={env['WAYLAND_DISPLAY']}",
                ] + base_cmd
                proc = subprocess.run(
                    fallback,
                    input=input_text,
                    text=True,
                    stdout=stdout_dest,
                    stderr=stderr_dest,
                    timeout=timeout,
                    close_fds=True,
                )
            return proc
        else:
            return subprocess.run(
                base_cmd,
                input=input_text,
                text=True,
                stdout=stdout_dest,
                stderr=stderr_dest,
                timeout=timeout,
                env=env,
                close_fds=True,
            )

    def get_clipboard(self) -> str | None:
        """Read the current text content of the system clipboard."""
        try:
            if self._has_wayland:
                proc = self._run_clipboard_cmd(["wl-paste", "-n"], timeout=0.25, capture_output=True)
                if proc.returncode == 0:
                    return proc.stdout
            elif self._has_xclip:
                proc = self._run_clipboard_cmd(["xclip", "-selection", "clipboard", "-o"], timeout=0.25, capture_output=True)
                if proc.returncode == 0:
                    return proc.stdout
            elif self._has_xsel:
                proc = self._run_clipboard_cmd(["xsel", "--clipboard", "--output"], timeout=0.25, capture_output=True)
                if proc.returncode == 0:
                    return proc.stdout
        except Exception:
            pass
        return None

    def set_clipboard(self, text: str) -> bool:
        """Copy text to the system clipboard."""
        try:
            if self._has_wayland:
                proc = self._run_clipboard_cmd(["wl-copy", "-n", "--", text], timeout=0.25, capture_output=False)
                return proc.returncode == 0
            elif self._has_xclip:
                proc = self._run_clipboard_cmd(["xclip", "-selection", "clipboard"], input_text=text, timeout=0.25, capture_output=False)
                return proc.returncode == 0
            elif self._has_xsel:
                proc = self._run_clipboard_cmd(["xsel", "--clipboard", "--input"], input_text=text, timeout=0.25, capture_output=False)
                return proc.returncode == 0
        except Exception as e:
            print(f"\033[91m[Typer Clipboard Exception]\033[0m {e}", file=sys.stderr)
            pass
        return False

    def _restore_clipboard_worker(self, saved_content: str) -> None:
        """Background worker that restores previous clipboard content."""
        with self._lock:
            self.set_clipboard(saved_content)

    def schedule_clipboard_restore(self, saved_content: str | None) -> None:
        """Schedule asynchronous clipboard restoration after PASTE completes."""
        if saved_content is None:
            return

        with self._lock:
            if self._restore_timer and self._restore_timer.is_alive():
                self._restore_timer.cancel()

            self._restore_timer = threading.Timer(
                self.config.RESTORE_CLIPBOARD_DELAY,
                self._restore_clipboard_worker,
                args=(saved_content,),
            )
            self._restore_timer.daemon = True
            self._restore_timer.start()

    # ---------------------------------------------------------- Injection Flow

    def send_backspaces(self, count: int) -> None:
        """Emit multiple backspaces to erase typed characters."""
        if not self._uinput or count <= 0:
            return

        for _ in range(count):
            self._uinput.write(ecodes.EV_KEY, ecodes.KEY_BACKSPACE, 1)
            self._uinput.write(ecodes.EV_KEY, ecodes.KEY_BACKSPACE, 0)
            self._uinput.syn()
            if self.config.BACKSPACE_DELAY > 0:
                time.sleep(self.config.BACKSPACE_DELAY)

    def send_paste(self) -> None:
        """Emit Ctrl+V keystroke combination to paste clipboard content."""
        if not self._uinput:
            return

        # Give the target application time to finish erasing and prepare for paste
        if self.config.PASTE_DELAY > 0:
            time.sleep(self.config.PASTE_DELAY)

        # Press Ctrl + V
        self._uinput.write(ecodes.EV_KEY, ecodes.KEY_LEFTCTRL, 1)
        self._uinput.syn()
        time.sleep(0.005)
        self._uinput.write(ecodes.EV_KEY, ecodes.KEY_V, 1)
        self._uinput.syn()

        # Hold briefly so Wayland compositor registers keypress combination
        time.sleep(0.010)

        # Release V + Ctrl
        self._uinput.write(ecodes.EV_KEY, ecodes.KEY_V, 0)
        self._uinput.syn()
        time.sleep(0.005)
        self._uinput.write(ecodes.EV_KEY, ecodes.KEY_LEFTCTRL, 0)
        self._uinput.syn()

    def erase_and_replace(
        self,
        typed_len: int,
        replacement_text: str,
        delimiter_char: str = "",
        restore_clipboard: bool = True,
    ) -> None:
        """Execute atomic replacement of typed Latin characters with Malayalam Unicode.

        Workflow:
        1. Read and stash existing clipboard content.
        2. Copy replacement Malayalam text (plus delimiter) to system clipboard.
        3. Emit backspaces to delete typed Latin word (plus registered delimiter).
        4. Emit Ctrl+V to paste the Malayalam Unicode string.
        5. Restore previous clipboard content asynchronously.
        """
        if typed_len <= 0 or not replacement_text:
            return

        # 1. Stash current clipboard
        stashed_clipboard = self.get_clipboard() if restore_clipboard else None

        # 2. Text to inject
        if self.config.ERASE_DELIMITER and delimiter_char:
            backspace_count = typed_len + len(delimiter_char)
            text_to_paste = replacement_text + delimiter_char
        else:
            backspace_count = typed_len
            text_to_paste = replacement_text

        # 3. Copy to clipboard with safety guard
        ok = self.set_clipboard(text_to_paste)
        if not ok:
            print(
                f"\033[91m[Typer Warning]\033[0m Clipboard copy failed for {replacement_text!r}. "
                f"Aborting replacement to protect typed text.",
                file=sys.stderr,
            )
            return

        # 4. Erase Latin characters only AFTER clipboard successfully holds replacement
        self.send_backspaces(backspace_count)

        # 5. Paste Malayalam Unicode
        self.send_paste()

        # 6. Restore original clipboard
        if restore_clipboard and stashed_clipboard is not None:
            self.schedule_clipboard_restore(stashed_clipboard)

    def close(self) -> None:
        """Clean teardown: cancel timers and destroy virtual device."""
        with self._lock:
            if self._restore_timer and self._restore_timer.is_alive():
                self._restore_timer.cancel()
                self._restore_timer = None

        if self._uinput:
            try:
                self._uinput.close()
            except Exception:
                pass
            self._uinput = None
