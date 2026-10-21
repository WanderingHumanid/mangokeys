"""Tests that the web demo and the Android app are the same keyboard.

The demo is meant to look and behave like the app, so the two have to agree
on the keys. These tests read the Kotlin file and the JavaScript file and
compare them, which is the cheapest way to stop them from drifting apart.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KOTLIN = ROOT / "apps/android/src/main/kotlin/com/mangokeyboard/KeyboardLayout.kt"
JAVASCRIPT = ROOT / "apps/expo-web/keyboard.js"


def unescape(values: list[str]) -> list[str]:
    """The two languages escape the quote differently, so undo both."""
    out = []
    for value in values:
        out.append(value.replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\"))
    return out


def kotlin_rows(name: str) -> list[list[str]]:
    text = KOTLIN.read_text(encoding="utf-8")
    match = re.search(rf"val {name} = listOf\((.*?)\n    \)", text, re.S)
    assert match, f"no {name} in the Kotlin file"
    rows = []
    for row in re.findall(r"listOf\((.*?)\)", match.group(1)):
        rows.append(unescape(re.findall(r'"((?:[^"\\]|\\.)*)"', row)))
    return rows


def javascript_rows(name: str) -> list[list[str]]:
    text = JAVASCRIPT.read_text(encoding="utf-8")
    match = re.search(rf"const {name} = \[(.*?)\n\];", text, re.S)
    assert match, f"no {name} in the JavaScript file"
    rows = []
    for row in re.findall(r"\[(.*?)\]", match.group(1)):
        found = re.findall(r'"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'', row)
        rows.append(unescape([a or b for a, b in found]))
    return rows


@pytest.mark.parametrize("name", ["LETTERS", "NUMBERS"])
def test_the_rows_are_the_same(name: str):
    kotlin_name = "LETTERS" if name == "LETTERS" else "NUMBERS"
    js_name = "LETTER_ROWS" if name == "LETTERS" else "NUMBER_ROWS"
    assert kotlin_rows(kotlin_name) == javascript_rows(js_name)


def test_the_three_rows_of_letters():
    rows = kotlin_rows("LETTERS")
    assert len(rows) == 3
    assert len(rows[0]) == 10
    assert len(rows[1]) == 9
    assert len(rows[2]) == 7


def test_the_numbers_layer_has_a_way_back():
    numbers = kotlin_rows("NUMBERS")
    assert len(numbers) == 3
    assert all(len(row) == 10 for row in numbers[:2])


def test_the_web_demo_has_the_same_palette_as_the_app():
    """The colours of the demo are the colours of the keyboard on the phone."""
    settings = (ROOT / "apps/android/src/main/kotlin/com/mangokeyboard/KeyboardSettings.kt").read_text(
        encoding="utf-8"
    )
    style = (ROOT / "apps/expo-web/style.css").read_text(encoding="utf-8")
    for colour in ("#1B1D21", "#31353B", "#3A3F46", "#E8EAED", "#2C5F8A"):
        assert colour in settings, f"{colour} is not in the app"
        assert colour.lower() in style.lower(), f"{colour} is not in the demo"


def test_the_demo_does_not_ship_the_direction_contract():
    """The brief is for the people who work on the page, not for the browser."""
    brief = ROOT / ".impeccable/briefs/web-demo.md"
    if not brief.exists():
        pytest.skip("no brief")
    words = ["DIRECTION CONTRACT", "THESIS", "OWN-WORLD", "FIRST VIEWPORT"]
    for name in ("index.html", "style.css", "app.js", "keyboard.js"):
        text = (ROOT / "apps/expo-web" / name).read_text(encoding="utf-8")
        for word in words:
            assert word not in text, f"{name} carries the contract"
