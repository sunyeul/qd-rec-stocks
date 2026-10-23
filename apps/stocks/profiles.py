"""Investor profiles: one fixed text per profile, shown to the proposer and the judge.

Real holdings can go in apps/stocks/profiles_private.py (git-ignored) as a
PROFILES dict; its entries are added to, or replace, the ones below.
"""

from . import config, items

DEFAULT_PROFILE = "base"

PROFILES = {
    "base": """Investor profile
- Horizon: hold for 3 to 12 months.
- Objective: beat the S&P 500 after costs, with drawdowns no worse than the market's.
- Risk limit: a single stock down 30% from entry triggers a review.
- Constraints: US large caps only; no leverage, no short positions, no options.
- Current holdings: overweight large US technology stocks, so ideas outside that cluster add more diversification.""",
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
    e.g. base-2026-10-23-named-all or base-2026-10-23-anon-top3. The last part is
    always present so that one tag is never a prefix of another."""
    pool = f"top{config.PER_SECTOR}" if config.PER_SECTOR else "all"
    return f"{name}-{items.snapshot_date()}-{'anon' if config.ANONYMIZE else 'named'}-{pool}"
