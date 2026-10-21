"""Tests for the documentation.

The README is the first thing anybody reads about the project, and it has a
few rules that are easy to break by accident: no em dashes, no emojis, and
every number in it has to be one that a script in this repository produced.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

# The characters that must not appear. An em dash and an en dash are the
# long dashes, and the emoji blocks are the pictographs.
LONG_DASHES = "—–"
EMOJI_RANGES = (
    (0x1F300, 0x1FAFF),
    (0x2600, 0x27BF),
    (0x1F000, 0x1F2FF),
)


def is_emoji(character: str) -> bool:
    code = ord(character)
    return any(low <= code <= high for low, high in EMOJI_RANGES)


def test_the_readme_exists():
    assert README.exists()


@pytest.mark.parametrize("name", ["README.md", "CHANGELOG.md", "docs/RELEASING.md"])
def test_no_long_dashes(name: str):
    text = (ROOT / name).read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), 1):
        for character in line:
            assert character not in LONG_DASHES, f"{name}:{number} has a long dash"


@pytest.mark.parametrize("name", ["README.md", "CHANGELOG.md", "docs/RELEASING.md"])
def test_no_emojis(name: str):
    text = (ROOT / name).read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), 1):
        for character in line:
            assert not is_emoji(character), f"{name}:{number} has an emoji"


def test_the_readme_names_both_ways_to_run_it():
    text = README.read_text(encoding="utf-8")
    assert "The web demo" in text
    assert "The Android app" in text
    assert "scripts.build_web" in text
    assert "assembleDebug" in text


def test_the_readme_has_the_machine_learning_section():
    text = README.read_text(encoding="utf-8")
    assert "## The machine learning" in text
    for word in ("Aksharantar", "augmentation", "transformer", "beam", "CER", "top-1"):
        assert word in text


def test_the_numbers_in_the_readme_are_the_measured_ones():
    """The headline numbers have to match the files the evaluation wrote."""
    text = README.read_text(encoding="utf-8")
    results = json.loads((ROOT / "evals" / "results" / "eval.json").read_text(encoding="utf-8"))
    suggestions = json.loads(
        (ROOT / "evals" / "results" / "suggestions.json").read_text(encoding="utf-8")
    )
    model = results[0]
    assert f"{model['top1'] * 100:.1f}%" in text
    assert f"{model['cer'] * 100:.1f}%" in text
    with_list = suggestions["with_word_list"]
    assert f"{with_list['top1'] * 100:.1f}%" in text
    assert f"{with_list['cer'] * 100:.1f}%" in text


def test_the_readme_has_no_broken_relative_links():
    text = README.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        path = target.split("#")[0]
        assert (ROOT / path).exists(), f"the link to {target} is broken"
