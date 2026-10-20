"""Measure the demo latency in a real browser.

    python -m scripts.bench_demo --url http://localhost:8077

Reports the time the model takes for a word, split into the parts that
matter, so it is clear where the time goes.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

CHROMIUM = Path.home() / ".cache/ms-playwright/chromium-1243/chrome-linux64/chrome"

WORDS = ["ente", "peru", "kerala", "sundari", "njangalkum", "ningalkkum", "aavashyappettappol"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8077")
    parser.add_argument("--runs", type=int, default=3)
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

        info = page.evaluate(
            """() => ({
                wasmPaths: ort.env.wasm.wasmPaths,
                threads: ort.env.wasm.numThreads,
                simd: ort.env.wasm.simd,
                hardware: navigator.hardwareConcurrency,
            })"""
        )
        print(f"runtime: {info}")

        results = []
        for word in WORDS:
            times = []
            for _ in range(args.runs):
                page.fill("#word", word)
                page.wait_for_timeout(50)
                timing = page.evaluate(
                    """async (word) => {
                        const start = performance.now();
                        await window.engine.candidates(word, 5);
                        return performance.now() - start;
                    }""",
                    word,
                )
                times.append(timing)
            best = min(times)
            results.append(best)
            print(f"  {word:22} {best:7.0f} ms")

        print(f"\nmedian {statistics.median(results):.0f} ms   mean {statistics.mean(results):.0f} ms")
        browser.close()


if __name__ == "__main__":
    main()
