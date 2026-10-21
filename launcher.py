#!/usr/bin/env python3
"""MangoKeys Mission Control & Real-Time TUI Launcher.

Oversees both the System-Wide Typer daemon and the Neural Transformer Visualizer:
- Clean auto-elevation on startup so sudo password is never eaten by the TUI
- Fully unbuffered real-time log streaming for Typer and Visualizer
- Single-key hotkey controls (no Enter required)
- Strict duplicate process prevention
- Responsive terminal layout that never breaks borders
- Web browser launcher for the Visualizer UI
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
import urllib.request
import webbrowser
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent
TYPER_SCRIPT = REPO_ROOT / "typer" / "run.py"
VISUALIZER_SCRIPT = REPO_ROOT / "visualizer" / "server.py"
VISUALIZER_VENV_PY = REPO_ROOT / "visualizer" / ".venv" / "bin" / "python"
LOG_FILE = Path("/tmp/mangokeys.log")

# Terminal ANSI colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CLEAR_SCREEN = "\033[2J\033[H"


def timestamp_str() -> str:
    return datetime.datetime.now().strftime("[%H:%M:%S]")


def has_hardware_access() -> bool:
    """Check if current process has permission to read hardware input devices."""
    if os.geteuid() == 0:
        return True
    try:
        for ev in Path("/dev/input").glob("event*"):
            try:
                with open(ev, "rb"):
                    return True
            except PermissionError:
                continue
    except Exception:
        pass
    return False


class ProcessManager:
    """Discovers, starts, and stops Typer and Visualizer processes safely."""

    @staticmethod
    def get_pids_for(pattern: str) -> list[int]:
        """Scan Linux /proc for running processes matching pattern."""
        my_pid = os.getpid()
        pids: list[int] = []
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                pid = int(entry)
                if pid == my_pid:
                    continue
                try:
                    with open(f"/proc/{pid}/cmdline", "rb") as f:
                        cmd = f.read().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
                        if pattern in cmd and "launcher.py" not in cmd:
                            pids.append(pid)
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
        except Exception:
            pass
        return pids

    @classmethod
    def get_typer_pids(cls) -> list[int]:
        return cls.get_pids_for("typer/run.py")

    @classmethod
    def get_visualizer_pids(cls) -> list[int]:
        return cls.get_pids_for("visualizer/server.py")

    @classmethod
    def terminate_pids(cls, pids: list[int], log_fn) -> None:
        """Gracefully terminate processes with SIGTERM, falling back to SIGKILL."""
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                log_fn(f"{YELLOW}Sent stop signal to PID {pid}{RESET}")
            except ProcessLookupError:
                continue
            except PermissionError:
                try:
                    subprocess.run(["sudo", "kill", "-15", str(pid)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    log_fn(f"{YELLOW}Sent sudo stop signal to PID {pid}{RESET}")
                except Exception as err:
                    log_fn(f"{RED}Permission error stopping PID {pid}: {err}{RESET}")

        time.sleep(0.4)
        for pid in pids:
            try:
                os.kill(pid, 0)
                try:
                    os.kill(pid, signal.SIGKILL)
                except PermissionError:
                    subprocess.run(["sudo", "kill", "-9", str(pid)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                log_fn(f"{RED}Force-stopped PID {pid}{RESET}")
            except (ProcessLookupError, OSError):
                pass

    @staticmethod
    def check_visualizer_health() -> tuple[bool, str]:
        """Check if visualizer HTTP endpoint is healthy."""
        try:
            req = urllib.request.Request("http://127.0.0.1:5000/api/health", headers={"User-Agent": "MangoKeysLauncher"})
            with urllib.request.urlopen(req, timeout=0.35) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return True, f"HEALTHY ({data.get('params', '5.37M')}, {data.get('device', 'cpu').upper()})"
        except Exception:
            pass
        return False, "UNAVAILABLE"


class MangoKeysLauncher:
    """Interactive TUI dashboard overseer."""

    def __init__(self, debug: bool = False, start_both: bool = False) -> None:
        self.debug = debug
        self.start_both = start_both
        self.running = True
        self.logs: collections.deque[str] = collections.deque(maxlen=30)
        self.typer_subproc: subprocess.Popen | None = None
        self.vis_subproc: subprocess.Popen | None = None
        self.log_lock = threading.Lock()
        self._orig_termios = None
        self.tick_count = 0
        self.needs_render = threading.Event()

        # Reset log file
        try:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write(f"--- MangoKeys Session Log Started {timestamp_str()} ---\n")
        except Exception:
            pass

    def setup_terminal(self) -> None:
        """Put stdin in cbreak raw mode to read single keypresses."""
        if self.debug or not sys.stdin.isatty():
            return
        fd = sys.stdin.fileno()
        try:
            self._orig_termios = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            self._orig_termios = None

    def restore_terminal(self) -> None:
        """Restore normal cooked terminal mode."""
        if self.debug:
            return
        if self._orig_termios is not None and sys.stdin.isatty():
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._orig_termios)
            except Exception:
                pass

    def acquire_sudo(self) -> bool:
        """Prompt user for sudo password cleanly and grant input device permissions."""
        if os.geteuid() == 0 or has_hardware_access():
            return True

        # Temporarily restore normal terminal mode so sudo password prompt is fully visible and interactive
        if not self.debug:
            self.restore_terminal()
        print(f"\n{CYAN}{BOLD}[MangoKeys Typer]{RESET} Live desktop typing requires hardware input access.")
        print(f"{YELLOW}Please enter your password to grant keyboard device access:{RESET}")
        try:
            user = os.environ.get("USER", "vishnunandan555")
            event_devs = [str(p) for p in Path("/dev/input").glob("event*")]
            cmd = ["sudo", "setfacl", "-m", f"u:{user}:rw", "/dev/uinput"] + event_devs
            p = subprocess.run(cmd)
            success = (p.returncode == 0)
        except KeyboardInterrupt:
            self.running = False
            print()
            return False
        except Exception:
            success = False
        finally:
            if not self.debug:
                self.setup_terminal()

        if not success:
            if self.running:
                self.log(f"{RED}Hardware authorization cancelled or failed.{RESET}")
        else:
            self.log(f"{GREEN}✓ Keyboard access granted. Typer running natively in your desktop session.{RESET}")
        return success

    def log(self, message: str) -> None:
        ts = timestamp_str()
        entry = f"{ts} {message}"
        with self.log_lock:
            self.logs.append(entry)
        if self.debug:
            print(entry, flush=True)
        else:
            self.needs_render.set()
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                # Strip ANSI colors for disk log
                clean_msg = message
                for c in (CYAN, GREEN, YELLOW, MAGENTA, RED, BOLD, DIM, RESET):
                    clean_msg = clean_msg.replace(c, "")
                f.write(f"{ts} {clean_msg}\n")
        except Exception:
            pass

    def start_visualizer(self) -> None:
        existing = ProcessManager.get_visualizer_pids()
        if existing:
            self.log(f"{YELLOW}Visualizer server already active (PID: {existing[0]}){RESET}")
            return

        py_bin = str(VISUALIZER_VENV_PY) if VISUALIZER_VENV_PY.exists() else sys.executable
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        try:
            self.vis_subproc = subprocess.Popen(
                [py_bin, "-u", str(VISUALIZER_SCRIPT), "--port", "5000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )

            def _reader():
                if not self.vis_subproc or not self.vis_subproc.stdout:
                    return
                for line in iter(self.vis_subproc.stdout.readline, ""):
                    line_s = line.strip()
                    if line_s and not line_s.startswith("="):
                        self.log(f"{MAGENTA}[VISUALIZER]{RESET} {line_s}")

            threading.Thread(target=_reader, daemon=True).start()
            self.log(f"{GREEN}✓ Visualizer started (PID: {self.vis_subproc.pid}) · URL: http://localhost:5000{RESET}")
        except Exception as err:
            self.log(f"{RED}Failed to start Visualizer: {err}{RESET}")

    def stop_visualizer(self) -> None:
        pids = ProcessManager.get_visualizer_pids()
        if self.vis_subproc and self.vis_subproc.pid not in pids:
            pids.append(self.vis_subproc.pid)

        if not pids:
            self.log(f"{YELLOW}Visualizer server is already stopped.{RESET}")
            return

        self.log(f"{YELLOW}Stopping Visualizer (PID: {pids[0]})...{RESET}")
        ProcessManager.terminate_pids(pids, self.log)
        self.vis_subproc = None
        self.log(f"{GREEN}✓ Visualizer stopped successfully.{RESET}")

    def start_typer(self, dry_run: bool = False) -> bool:
        existing = ProcessManager.get_typer_pids()
        if existing:
            self.log(f"{YELLOW}Typer daemon already active (PID: {existing[0]}){RESET}")
            return True

        if dry_run:
            cmd = [sys.executable, "-u", str(TYPER_SCRIPT), "--dry-run"]
        else:
            if not has_hardware_access():
                if not self.acquire_sudo():
                    return False

            # Run directly as desktop user whenever permissions allow!
            if has_hardware_access() and os.geteuid() != 0:
                cmd = [sys.executable, "-u", str(TYPER_SCRIPT)]
            else:
                wayland_disp = os.environ.get("WAYLAND_DISPLAY", "wayland-0")
                uid = str(os.getuid())
                xdg_runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{uid}")
                cmd = [
                    "sudo",
                    f"WAYLAND_DISPLAY={wayland_disp}",
                    f"XDG_RUNTIME_DIR={xdg_runtime}",
                    sys.executable,
                    "-u",
                    str(TYPER_SCRIPT),
                ]

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        try:
            self.typer_subproc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )

            def _reader():
                if not self.typer_subproc or not self.typer_subproc.stdout:
                    return
                for line in iter(self.typer_subproc.stdout.readline, ""):
                    line_s = line.strip()
                    # In debug mode show everything, otherwise filter decorative box borders for TUI tail
                    if line_s:
                        if self.debug or not any(line_s.startswith(c) for c in ("╔", "║", "╚", "─")):
                            self.log(f"{CYAN}[TYPER]{RESET} {line_s}")

            threading.Thread(target=_reader, daemon=True).start()
            mode_desc = "Dry-Run Simulation" if dry_run else "System-Wide Live Typing"
            self.log(f"{GREEN}✓ Typer started (PID: {self.typer_subproc.pid}) · Mode: {mode_desc}{RESET}")
            return True
        except Exception as err:
            self.log(f"{RED}Failed to start Typer: {err}{RESET}")
            return False

    def stop_typer(self) -> None:
        pids = ProcessManager.get_typer_pids()
        if self.typer_subproc and self.typer_subproc.pid not in pids:
            pids.append(self.typer_subproc.pid)

        if not pids:
            self.log(f"{YELLOW}Typer daemon is already stopped.{RESET}")
            return

        self.log(f"{YELLOW}Stopping Typer (PID: {pids[0]})...{RESET}")
        ProcessManager.terminate_pids(pids, self.log)
        self.typer_subproc = None
        self.log(f"{GREEN}✓ Typer daemon stopped successfully.{RESET}")

    def restart_active(self) -> None:
        typer_active = bool(ProcessManager.get_typer_pids())
        vis_active = bool(ProcessManager.get_visualizer_pids())

        if not typer_active and not vis_active:
            self.log(f"{YELLOW}No services are running to restart. Press [3] to start both!{RESET}")
            return

        self.log(f"{YELLOW}Restarting active services...{RESET}")
        if typer_active:
            self.stop_typer()
        if vis_active:
            self.stop_visualizer()

        time.sleep(0.5)
        if vis_active:
            self.start_visualizer()
        if typer_active:
            self.start_typer()

    def open_browser(self) -> None:
        vis_pids = ProcessManager.get_visualizer_pids()
        if not vis_pids:
            self.log(f"{YELLOW}Visualizer server not running. Starting it now...{RESET}")
            self.start_visualizer()
            time.sleep(1.0)

        url = "http://localhost:5000"
        self.log(f"{CYAN}🌐 Launching {url} in your default browser...{RESET}")
        try:
            webbrowser.open(url)
        except Exception as err:
            self.log(f"{RED}Could not open browser: {err}{RESET}")

    def render_tui(self) -> None:
        """Render clean, responsive dashboard that never wraps or breaks borders."""
        cols, rows = shutil.get_terminal_size((80, 24))
        width = max(60, min(cols, 100))
        hr = "─" * width

        typer_pids = ProcessManager.get_typer_pids()
        vis_pids = ProcessManager.get_visualizer_pids()
        is_healthy, health_msg = ProcessManager.check_visualizer_health()

        typer_status = f"{GREEN}● RUNNING{RESET} (PID: {typer_pids[0]})" if typer_pids else f"{DIM}○ STOPPED{RESET}"
        vis_status = f"{GREEN}● RUNNING{RESET} (PID: {vis_pids[0]})" if vis_pids else f"{DIM}○ STOPPED{RESET}"

        if typer_pids and vis_pids and is_healthy:
            sync_status = f"{GREEN}● LINKED & ACTIVE (Desktop keystrokes stream to Web UI){RESET}"
        elif typer_pids and not vis_pids:
            sync_status = f"{CYAN}⌨️  TYPER ONLY (Press [2] to start visualizer link){RESET}"
        elif vis_pids and not typer_pids:
            sync_status = f"{MAGENTA}🧠 VISUALIZER ONLY (Press [1] to start system-wide typer){RESET}"
        else:
            sync_status = f"{YELLOW}○ IDLE (Press [3] to start both){RESET}"

        auth_badge = (
            f"{GREEN}✓ ROOT / HARDWARE PERMISSION: ACTIVE{RESET}"
            if os.geteuid() == 0
            else f"{YELLOW}ℹ️  USER: STANDARD (Run with 'sudo python3 launcher.py' for live hardware access){RESET}"
        )

        output_lines = [
            CLEAR_SCREEN,
            f"{CYAN}{hr}{RESET}",
            f"{BOLD}  MANGO KEYS  ·  MISSION CONTROL & LAUNCHER{RESET}",
            f"  {DIM}Zero-Install System-Wide Typer + Real-Time Neural Transformer Visualizer{RESET}",
            f"{CYAN}{hr}{RESET}",
            f"  {auth_badge}",
            f"\n{BOLD}─── SERVICES MONITOR ────────────────────────────────────────────────────────{RESET}",
            f"  {BOLD}⌨️  TYPER DAEMON     :{RESET}  [{typer_status}]",
            f"      {DIM}Function: Auto-converts Manglish on Space/Enter  ·  Right-Alt to Pause{RESET}",
            f"  {BOLD}🧠 VISUALIZER SERVER:{RESET}  [{vis_status}]  ·  URL: {CYAN}http://localhost:5000{RESET}",
            f"      {DIM}Health  : {health_msg}{RESET}",
            f"  {BOLD}⚡ DESKTOP SYNC     :{RESET}  [{sync_status}]",
            f"\n{BOLD}─── COMMAND ACTIONS ─────────────────────────────────────────────────────────{RESET}",
            f"  {BOLD}[1]{RESET} Toggle Typer       {BOLD}[3]{RESET} Start Both         {BOLD}[O]{RESET} Open Web Visualizer",
            f"  {BOLD}[2]{RESET} Toggle Visualizer  {BOLD}[4]{RESET} Stop Both          {BOLD}[D]{RESET} Run Dry-Run Simulation",
            f"  {BOLD}[R]{RESET} Restart Active     {BOLD}[C]{RESET} Clear Logs         {BOLD}[Space]{RESET} Refresh Now  {BOLD}[Q]{RESET} Quit",
            f"\n{BOLD}─── LIVE ACTIVITY STREAM ────────────────────────────────────────────────────{RESET}",
        ]

        # Calculate remaining height for logs dynamically
        fixed_lines = len(output_lines) + 2
        log_capacity = max(6, min(rows - fixed_lines, 14))

        with self.log_lock:
            display_logs = list(self.logs)[-log_capacity:]

        if not display_logs:
            output_lines.append(f"  {DIM}(No recent events. Press [1] or [2] above to start services.){RESET}")
        else:
            for l in display_logs:
                output_lines.append(f"  {l}")

        SPINNER = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
        spinner_char = SPINNER[self.tick_count % len(SPINNER)]

        output_lines.append(f"{CYAN}{hr}{RESET}")
        output_lines.append(
            f" {CYAN}{spinner_char} LIVE{RESET} · {DIM}Auto-refresh: 0.5s · [Space] Force Refresh · Time: {timestamp_str()}{RESET}"
        )

        sys.stdout.write("\n".join(output_lines) + "\n")
        sys.stdout.flush()

    def run_debug(self) -> None:
        """Run in headless debug mode with pure streaming logs (no interactive TUI)."""
        self.log(f"{GREEN}=== MangoKeys Debug Mode Active (Pure Log Stream) ==={RESET}")

        def _sig_handler(sig: int, frame: object) -> None:
            self.running = False

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        try:
            if self.start_both:
                self.log(f"{CYAN}Auto-starting both Typer and Visualizer...{RESET}")
                if not self.start_typer():
                    # Sudo prompt was cancelled by user
                    return
                time.sleep(0.3)
                self.start_visualizer()
            else:
                self.log(f"{YELLOW}Services not started. (Use --both to start automatically){RESET}")

            self.log(f"{GREEN}✓ Ready! Type in any desktop application (e.g. blankslate.io).{RESET}")
            self.log(f"{DIM}Press Ctrl+C at any time to stop services and exit cleanly.{RESET}")

            while self.running:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_typer()
            self.stop_visualizer()
            self.log(f"{GREEN}✓ All services stopped. MangoKeys exited cleanly.{RESET}\n")

    def run(self) -> None:
        """Main event loop handling raw keypresses and live dashboard updates."""
        if self.debug:
            self.run_debug()
            return

        self.log(f"{GREEN}Mission Control ready. Press [3] to start both Typer & Visualizer.{RESET}")
        if self.start_both:
            self.log("Auto-launching both Visualizer & Typer (--both)...")
            self.start_visualizer()
            time.sleep(0.3)
            self.start_typer()

        self.setup_terminal()
        last_render = 0.0

        try:
            while self.running:
                now = time.time()
                self.tick_count += 1
                if self.needs_render.is_set() or (now - last_render >= 0.5):
                    self.needs_render.clear()
                    self.render_tui()
                    last_render = now

                # Poll for user keypress (timeout 0.2s)
                r, _, _ = select.select([sys.stdin], [], [], 0.2)
                if r:
                    ch = sys.stdin.read(1)
                    if not ch:
                        continue

                    key = ch.lower()
                    if key in ("q", "\x03"):  # 'q' or Ctrl+C
                        self.log("Exiting Mission Control...")
                        self.render_tui()
                        break
                    elif key in (" ", "\n", "\r", "u"):
                        self.render_tui()
                        last_render = time.time()
                    elif key == "1":
                        if ProcessManager.get_typer_pids():
                            self.stop_typer()
                        else:
                            self.start_typer()
                        self.render_tui()
                        last_render = time.time()
                    elif key == "2":
                        if ProcessManager.get_visualizer_pids():
                            self.stop_visualizer()
                        else:
                            self.start_visualizer()
                        self.render_tui()
                        last_render = time.time()
                    elif key == "3":
                        self.log("Launching both Visualizer & Typer...")
                        self.start_visualizer()
                        time.sleep(0.3)
                        self.start_typer()
                        self.render_tui()
                        last_render = time.time()
                    elif key == "4":
                        self.log("Stopping all active services...")
                        self.stop_typer()
                        self.stop_visualizer()
                        self.render_tui()
                        last_render = time.time()
                    elif key in ("o", "b"):
                        self.open_browser()
                        self.render_tui()
                        last_render = time.time()
                    elif key == "r":
                        self.restart_active()
                        self.render_tui()
                        last_render = time.time()
                    elif key == "d":
                        self.log("Executing Typer dry-run benchmark test...")
                        self.start_typer(dry_run=True)
                        self.render_tui()
                        last_render = time.time()
                    elif key == "c":
                        with self.log_lock:
                            self.logs.clear()
                        self.log("Activity logs cleared.")
                        self.render_tui()
                        last_render = time.time()

        finally:
            self.restore_terminal()
            print(f"\n{CYAN}MangoKeys Mission Control exited cleanly.{RESET}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MangoKeys Mission Control & Real-Time Launcher",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Automatically launch both Typer and Visualizer on startup",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (pure streaming logs without interactive TUI)",
    )
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        debug_mode = args.debug or not sys.stdin.isatty()
        launcher = MangoKeysLauncher(debug=debug_mode, start_both=args.both)
        launcher.run()
    except KeyboardInterrupt:
        print(f"\n{CYAN}🛑 MangoKeys exited cleanly.{RESET}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
