"""Investor profiles: one fixed text per profile, shown to the proposer and the judge.

Real holdings can go in apps/stocks/profiles_private.py (git-ignored) as a
PROFILES dict; its entries are added to, or replace, the ones below.
"""

from . import config, items

# Neutral on purpose: holdings or a risk limit in the profile made most theses argue
# from the profile ("diversify away from tech") instead of from the stock.
DEFAULT_PROFILE = "neutral"

PROFILES = {
    "neutral": """Investor profile
- Horizon: hold for 3 to 12 months.
- Objective: beat the S&P 500 after costs.
- Constraints: US large caps only; long only, no leverage, no options.""",
}

try:
    from .profiles_private import PROFILES as _PRIVATE

    PROFILES = {**PROFILES, **_PRIVATE}
except ImportError:
    pass


def get_profile_text(name: str) -> str:
    if name not in PROFILES:
        raise ValueError(f"Unknown profile {name!r}; known: {sorted(PROFILES)}")
    return PROFILES[name]


def profile_tag(name: str) -> str:
    """The profile's token in archive filenames, with the card snapshot and mode,
    e.g. neutral-2026-04-30-named or neutral-2026-04-30-anon."""
    return f"{name}-{items.snapshot_date()}-{'anon' if config.ANONYMIZE else 'named'}"
