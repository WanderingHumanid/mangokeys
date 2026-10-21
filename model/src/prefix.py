"""Prefix training pairs for the transliterator.

A keyboard has to answer two questions at once: what has the user written so
far, and which word did they mean. The transliterator was trained to answer
only the second one, so when it is given half a word it finishes the word
instead of stopping in the middle of it. That is why it cannot complete a
word, and why the word list has nothing to search in.

This module manufactures the missing training data. From one pair, such as

    kerala    കേരളം

it writes the pairs that a keyboard needs

    k         ക
    ke        കേ
    ker       കേര
    kera      കേര
    keral     കേരള

The roman side is cut at every character and the native side is cut where the
matching consonant falls. The cut of the native side is found by counting the
consonants, because the two sides are not written one letter at a time: a
single Malayalam letter is a letter plus a vowel sign, and a cluster is two
letters.

A pair is only written when the two sides agree on the number of consonants.
When they do not, the roman spelling is using a letter for something the
native side does not have, such as the m that stands for the anusvara, and
there is no safe way to cut the word.
"""

from __future__ import annotations

from model.src.rules import CLUSTERS, CONSONANTS

# The letters that carry a vowel.
VOWEL_LETTERS = set("aeiou") | {"A", "E", "O"}

# The marks that turn a consonant into a long or a different vowel.
VOWEL_SIGNS = set("ാിീുൂൃൄ൅െേൈൊോൌൗ")

# The old chillu letters. They stand for a consonant of their own.
CHILLUS = set("ൺൻർൽൾൿ")

CLUSTER_KEYS = sorted(CLUSTERS, key=len, reverse=True)
CONSONANT_KEYS = sorted(CONSONANTS, key=len, reverse=True)

MALAYALAM_FIRST = 0x0D15
MALAYALAM_LAST = 0x0D39


def _match(text: str, index: int, keys: list[str]) -> str | None:
    for key in keys:
        if text.startswith(key, index):
            return key
    return None


def scan_roman(roman: str) -> tuple[list[int], list[bool], list[str]]:
    """Walk the roman spelling and note the consonants.

    Returns the number of consonants in every prefix of the string, whether
    the last consonant of that prefix is followed by a vowel, and the roman
    key of every consonant in order.
    """
    counts = [0] * (len(roman) + 1)
    signed = [False] * (len(roman) + 1)
    keys: list[str] = []

    index = 0
    count = 0
    has_sign = False
    consonant_end = -1
    while index < len(roman):
        cluster = _match(roman, index, CLUSTER_KEYS)
        consonant = None if cluster else _match(roman, index, CONSONANT_KEYS)
        if cluster is None and consonant is None:
            if index == consonant_end and roman[index] in VOWEL_LETTERS:
                has_sign = True
            index += 1
            counts[index] = count
            signed[index] = has_sign
            continue

        if cluster is not None:
            key = cluster
            length = len(cluster)
            total = 2
        else:
            key = consonant
            doubled = len(consonant) == 1 and roman[index + 1 : index + 2] == consonant
            length = 2 if doubled else len(consonant)
            total = 2 if doubled else 1

        has_sign = False
        for step in range(1, length + 1):
            index += 1
            # A cluster is two consonants, so the cut in the middle of it
            # already accounts for one of them.
            share = 1 if step < length else total - length + 1
            count += max(share, 0)
            keys.extend([key] * max(share, 0))
            counts[index] = count
            signed[index] = has_sign
        consonant_end = index

    return counts, signed, keys


def scan_native(native: str) -> list[int]:
    """The codepoint position of every consonant in a native word."""
    positions: list[int] = []
    for index, char in enumerate(native):
        if char in CHILLUS:
            positions.append(index)
            continue
        if MALAYALAM_FIRST <= ord(char) <= MALAYALAM_LAST:
            positions.append(index)
    return positions


def _align(roman_keys: list[str], native_count: int) -> list[int | None] | None:
    """Match the roman consonants to the native ones.

    Returns the native index of every roman consonant, or None for the roman
    consonant that has no native partner.
    """
    if len(roman_keys) == native_count:
        return list(range(native_count))

    # The roman m can stand for the anusvara, which is not a consonant.
    if len(roman_keys) == native_count + 1:
        for position, key in enumerate(roman_keys):
            if key != "m":
                continue
            alignment: list[int | None] = [None] * len(roman_keys)
            native_index = 0
            for index in range(len(roman_keys)):
                if index == position:
                    continue
                alignment[index] = native_index
                native_index += 1
            return alignment
    return None


def prefix_targets(roman: str, native: str) -> list[tuple[str, str]]:
    """The prefix pairs of one word.

    The roman side is cut at every character except the end, because the
    whole word is already in the training data. The native side is cut where
    the matching consonant falls.
    """
    if not roman or not native:
        return []

    counts, signed, keys = scan_roman(roman)
    positions = scan_native(native)
    if not positions:
        return []

    alignment = _align(keys, len(positions))
    if alignment is None:
        return []

    out: list[tuple[str, str]] = []
    for cut in range(1, len(roman)):
        used = counts[cut]
        matched = [index for index in alignment[:used] if index is not None]
        if not matched:
            end = positions[0]
        else:
            end = positions[matched[-1]] + 1

        if signed[cut] and end < len(native) and native[end] in VOWEL_SIGNS:
            end += 1

        target = native[:end]
        if not target or len(target) >= len(native):
            continue
        out.append((roman[:cut], target))

    return out


def consonant_count(text: str) -> int:
    """The number of consonants in a native word."""
    return len(scan_native(text))
