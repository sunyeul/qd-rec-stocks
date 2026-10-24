"""TypeSafe Jev through OpenRouter's Decisions API: bounded questions answered with probabilities.

Used as the judge when config.JUDGE_MODEL starts with "typesafe/". Jev bills input
tokens only, and its probabilities give continuous scores instead of integer ones.
"""

import asyncio
import os

import httpx

URL = "https://openrouter.ai/api/alpha/decisions"
_RETRY_WAITS = (2, 5, 10)


async def decide(model: str, state: dict, questions: dict) -> dict:
    """{question name: answer} for one state; retried on network or server errors."""
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    last = None
    for wait in (*_RETRY_WAITS, None):
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(URL, headers=headers, json={"model": model, "state": state, "questions": questions})
            if r.status_code < 500 and r.status_code != 429:
                r.raise_for_status()
                return r.json()["answers"]
            last = RuntimeError(f"Jev HTTP {r.status_code}: {r.text[:200]}")
        except (httpx.TransportError, httpx.TimeoutException) as e:
            last = e
        if wait is None:
            break
        await asyncio.sleep(wait)
    raise RuntimeError(f"Jev request failed: {last}")


def expected_level(answer: dict) -> float:
    """Probability-weighted level of a Score answer (0 = first criterion)."""
    return sum(int(level) * p for level, p in answer["probabilities"].items())
