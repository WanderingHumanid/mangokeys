"""Check the expo demo in a real browser.

    python -m scripts.check_demo --url http://localhost:8077

Loads the page, waits for the model to load, types a word and reads the
transliteration back. This is the only check that covers the browser runtime,
the wasm build and the page wiring together.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CHROMIUM = Path.home() / ".cache/ms-playwright/chromium-1243/chrome-linux64/chrome"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8077")
    parser.add_argument("--word", default="ente peru")
    parser.add_argument("--shot", default=None, help="save a screenshot here")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(CHROMIUM) if CHROMIUM.exists() else None,
            args=["--no-sandbox"],
        )
        page = browser.new_page()
        errors: list[str] = []

        def note_console(message) -> None:
            # The source map is not shipped, and the browser asks for it anyway.
            if message.type != "error" or ".map" in message.text:
                return
            errors.append(message.text)

        def note_error(error) -> None:
            if ".map" in str(error):
                return
            errors.append(str(error))

        page.on("pageerror", note_error)
        page.on("console", note_console)

        page.goto(args.url, wait_until="networkidle")
        page.wait_for_function(
            "document.getElementById('status').textContent.includes('ready')",
            timeout=120000,
        )
        print(f"status: {page.locator('#status').inner_text()}")

        page.fill("#word", args.word)
        page.wait_for_timeout(4000)

        output = page.locator("#model-output").inner_text()
        rules = page.locator("#rule-output").inner_text()
        candidates = page.locator("#candidate-list").inner_text()

        print(f"input       {args.word}")
        print(f"model       {output}")
        print(f"rules       {rules}")
        print(f"candidates  {candidates.replace(chr(10), ' | ')}")

        rows = page.locator("#bench-table tbody tr").count()
        print(f"benchmark rows: {rows}")

        if args.shot:
            page.screenshot(path=args.shot, full_page=True)
            print(f"screenshot {args.shot}")

        browser.close()

    if errors:
        print("\npage errors:")
        for error in errors:
            print(f"  {error}")
        raise SystemExit(1)

    if not output.strip():
        print("the model produced nothing")
        raise SystemExit(1)
    print("\nthe demo works")


if __name__ == "__main__":
    main()
