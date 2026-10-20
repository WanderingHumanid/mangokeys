"""Measure how the demo feels while typing.

    python -m scripts.bench_typing --url http://localhost:8077

Types a word one character at a time and reports the longest gap between
animation frames. That gap is the time the main thread was blocked, which is
what the user feels as a stutter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

CHROMIUM = Path.home() / ".cache/ms-playwright/chromium-1243/chrome-linux64/chrome"

WORDS = ["ente peru", "kerala", "njangalkum", "aavashyappettappol"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8077")
    parser.add_argument("--delay", type=int, default=120, help="ms between keystrokes")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(CHROMIUM) if CHROMIUM.exists() else None,
            args=["--no-sandbox"],
        )
        page = browser.new_page()
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_function(
            "document.getElementById('status').textContent.includes('ready')",
            timeout=120000,
        )

        page.evaluate(
            """() => {
                window.frames = [];
                let last = performance.now();
                const tick = () => {
                    const now = performance.now();
                    window.frames.push(now - last);
                    last = now;
                    requestAnimationFrame(tick);
                };
                requestAnimationFrame(tick);
            }"""
        )

        for word in WORDS:
            page.fill("#word", "")
            page.wait_for_timeout(300)
            page.evaluate("() => { window.frames = []; }")
            for char in word:
                page.type("#word", char, delay=args.delay)
            page.wait_for_timeout(1200)
            stats = page.evaluate(
                """() => {
                    const f = window.frames.slice(2);
                    f.sort((a, b) => b - a);
                    return {
                        longest: Math.round(f[0] || 0),
                        over50: f.filter((x) => x > 50).length,
                        over100: f.filter((x) => x > 100).length,
                        total: f.length,
                    };
                }"""
            )
            print(
                f"  {word:22} longest frame gap {stats['longest']:5} ms   "
                f"frames over 50 ms: {stats['over50']:3}   over 100 ms: {stats['over100']:3}"
            )

        # How long the model takes to come back with an answer.
        print("\n  round trip from the keystroke to the output on screen")
        for word in WORDS:
            elapsed = page.evaluate(
                """async (word) => {
                    const input = document.getElementById('word');
                    const output = document.getElementById('model-output');
                    input.value = '';
                    input.dispatchEvent(new Event('input'));
                    await new Promise((r) => setTimeout(r, 400));
                    const before = output.textContent;
                    const start = performance.now();
                    input.value = word;
                    input.dispatchEvent(new Event('input'));
                    while (output.textContent === before) {
                        await new Promise((r) => setTimeout(r, 5));
                    }
                    return performance.now() - start;
                }""",
                word,
            )
            print(f"  {word:22} {elapsed:6.0f} ms")

        browser.close()


if __name__ == "__main__":
    main()
