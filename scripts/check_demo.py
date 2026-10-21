"""Look at the demo in a real browser.

    python -m scripts.check_demo --url http://localhost:8077

Loads the page, waits for the model to load, types a word, presses the space
bar and reads the field back. The screenshots are the evidence that the page
works and that it looks the way it is supposed to look.
"""

from __future__ import annotations

import argparse
from pathlib import Path

CHROMIUM = Path.home() / ".cache/ms-playwright/chromium-1243/chrome-linux64/chrome"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8077")
    parser.add_argument("--word", default="ente peru")
    parser.add_argument("--shot", default=None, help="save a screenshot here")
    parser.add_argument("--shot-after", default=None, help="save a screenshot after the space bar")
    parser.add_argument("--shot-mobile", default=None, help="save a mobile screenshot here")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=960)
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(CHROMIUM) if CHROMIUM.exists() else None,
            args=["--no-sandbox"],
        )
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        errors: list[str] = []
        missing: list[str] = []

        def note_console(message) -> None:
            if message.type == "error":
                errors.append(message.text)

        def note_error(error) -> None:
            errors.append(str(error))

        def note_response(response) -> None:
            if response.status >= 400:
                missing.append(f"{response.status} {response.url}")

        page.on("pageerror", note_error)
        page.on("console", note_console)
        page.on("response", note_response)

        page.goto(args.url, wait_until="networkidle")
        page.wait_for_function(
            "document.getElementById('status').textContent.includes('model ready')",
            timeout=120000,
        )
        print(f"status: {page.locator('#status').inner_text()}")

        page.click("#field")
        page.fill("#field", args.word)
        page.wait_for_timeout(2500)

        candidates = page.locator("#strip .candidate").all_inner_texts()
        print(f"typed       {args.word}")
        print(f"candidates  {' | '.join(candidates)}")

        if args.shot:
            page.screenshot(path=args.shot, full_page=True)
            print(f"screenshot {args.shot}")

        # The space bar commits the highlighted candidate.
        page.keyboard.press("Space")
        page.wait_for_timeout(1200)
        value = page.locator("#field").input_value()
        print(f"after space {value}")

        rows = page.locator("#bench-table tbody tr").count()
        print(f"benchmark rows: {rows}")

        if args.shot_after:
            page.screenshot(path=args.shot_after, full_page=True)
            print(f"screenshot {args.shot_after}")

        # Tapping a candidate replaces the word in the field.
        page.fill("#field", "ente")
        page.wait_for_timeout(2000)
        second = page.locator("#strip .candidate").nth(1)
        tapped = second.inner_text()
        second.click()
        page.wait_for_timeout(400)
        after_tap = page.locator("#field").input_value()
        print(f"tapped        {tapped}")
        print(f"after tap    {after_tap}")
        if after_tap.strip() != tapped:
            print("\ntapping a candidate did not replace the word")
            raise SystemExit(1)

        # The keys of the keyboard on the page type into the field.
        page.fill("#field", "")
        page.click("#keys .key[data-key='e']")
        page.click("#keys .key[data-key='n']")
        page.wait_for_timeout(300)
        typed = page.locator("#field").input_value()
        print(f"typed with the keys {typed}")
        if typed != "en":
            print("\nthe keys on the page do not type")
            raise SystemExit(1)

        # The numbers layer.
        page.click("#keys .key[data-action='numbers']")
        page.wait_for_timeout(200)
        first = page.locator("#keys .key[data-key='1']").count()
        page.click("#keys .key[data-action='letters']")
        print(f"numbers layer {first} keys")
        if first != 1:
            print("\nthe numbers layer did not open")
            raise SystemExit(1)

        # The backspace deletes in the field.
        page.click("#keys .key[data-key='backspace']")
        page.wait_for_timeout(200)
        print(f"after backspace {page.locator('#field').input_value()}")
        if page.locator("#field").input_value() != "e":
            print("\nthe backspace did not delete")
            raise SystemExit(1)

        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.goto(args.url, wait_until="networkidle")
        mobile.wait_for_function(
            "document.getElementById('status').textContent.includes('model ready')",
            timeout=120000,
        )
        mobile.click("#field")
        mobile.fill("#field", args.word)
        mobile.wait_for_timeout(2500)
        if args.shot_mobile:
            mobile.screenshot(path=args.shot_mobile, full_page=True)
            print(f"screenshot {args.shot_mobile}")

        browser.close()

    if missing:
        print("\nmissing resources:")
        for item in missing:
            print(f"  {item}")

    real_errors = [
        error
        for error in errors
        if "404" not in error or "Failed to load resource" not in error
    ]
    if real_errors:
        print("\npage errors:")
        for error in real_errors:
            print(f"  {error}")
        raise SystemExit(1)

    if not candidates:
        print("\nthe suggestion strip is empty")
        raise SystemExit(1)
    if value.strip() == args.word:
        print("\nthe space bar did not commit anything")
        raise SystemExit(1)
    print("\nthe demo works")


if __name__ == "__main__":
    main()
