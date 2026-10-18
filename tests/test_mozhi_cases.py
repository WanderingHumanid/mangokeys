"""The rule engine has to follow the Mozhi scheme wherever it is unambiguous."""

from pathlib import Path

from model.src.rules import RuleEngine

CASES = Path(__file__).resolve().parents[1] / "evals" / "mozhi_cases" / "mozhi_cases.tsv"


def load_cases() -> list[tuple[str, str]]:
    cases = []
    with CASES.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            cases.append((parts[0], parts[1]))
    return cases


def test_rule_engine_follows_the_mozhi_scheme():
    engine = RuleEngine()
    failures = [(roman, expected, engine.convert_top(roman)) for roman, expected in load_cases()]
    failures = [f for f in failures if f[1] != f[2]]
    assert not failures, failures
