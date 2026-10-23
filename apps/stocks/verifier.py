"""Rule-based number check that runs before the judge.

Every number in a thesis's thesis, evidence, catalyst and risks fields must
appear on the stock card or in the investor profile; otherwise the thesis
scores 0 without a judge call. A cited number may be rounded (card 24.6% ->
"25%"), but not changed. The falsifier and horizon are exempt because they
set new thresholds and durations by design.
"""

import re

CHECKED_FIELDS = ("thesis", "evidence", "catalyst", "risks")

# Digits not glued to a preceding letter, so "Q3" or "C-127"-style IDs inside words
# are skipped but "3M" (three-month) and "12.9" are read.
_NUMBER = re.compile(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?")


def _values(text: str) -> list[str]:
    return [m.replace(",", "").rstrip(".") for m in _NUMBER.findall(text)]


def _matches(cited: str, allowed: set[float]) -> bool:
    decimals = len(cited.split(".")[1]) if "." in cited else 0
    x = float(cited)
    return any(round(a, decimals) == x for a in allowed)


def verify(fields: dict, *sources: str) -> dict:
    """{"passed", "unmatched", "checked"} for the thesis fields against the source texts."""
    allowed = {float(v) for s in sources for v in _values(s)}
    cited = []
    for key in CHECKED_FIELDS:
        value = fields.get(key) or ""
        for text in (value if isinstance(value, list) else [value]):
            cited += _values(str(text))
    unmatched = sorted({c for c in cited if not _matches(c, allowed)})
    return {"passed": not unmatched, "unmatched": unmatched, "checked": len(cited)}
