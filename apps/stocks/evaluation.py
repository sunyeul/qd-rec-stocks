"""
The number check plus the judge's rubric score, the novelty check and the blind pairwise duel.
"""

import random

from shared.llm import acreate, client, loads_lenient, parse_json_bool, parse_judge_score

from . import jev
from .config import JUDGE_MODEL, REASONING_EFFORT
from .items import item_text
from .prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_PROMPT,
    NOVELTY_CHECK_SYSTEM_PROMPT,
    NOVELTY_CHECK_USER_PROMPT,
    PAIRWISE_DUEL_SYSTEM_PROMPT,
    PAIRWISE_DUEL_USER_PROMPT,
)
from .verifier import verify

# criterion -> maximum points; they add up to the 0-10 score
RUBRIC = {"evidence_match": 3, "profile_fit": 3, "risk_falsifier": 2, "catalyst": 2}

# The same rubric as Jev Score questions, worst level first (level i = i points).
JEV = JUDGE_MODEL.startswith("typesafe/")
JEV_RUBRIC = {
    "evidence_match": ("Is the thesis's evidence on the stock card, and does it support the conclusion?", [
        "Evidence missing, not on the card, or contradicting the conclusion",
        "Some card evidence, loosely connected to the conclusion",
        "Card evidence supports most of the case",
        "Specific card evidence clearly supports the whole case, without ignoring contrary lines"]),
    "profile_fit": ("Does the idea fit the investor's horizon, objective and constraints?", [
        "Does not fit, or argues against buying", "Fits weakly or generically",
        "Fits most of the profile", "Fits the horizon, objective and constraints well"]),
    "risk_falsifier": ("Are the main risks named, and is the falsifier observable and specific?", [
        "Risks or falsifier missing or vague", "Partly specific", "Main risks named; falsifier observable and specific"]),
    "catalyst": ("Is the catalyst concrete and plausible within the horizon?", [
        "No real catalyst or a generic one", "Plausible but vague", "Concrete and plausible within the horizon"]),
}


async def judge_score_async(investor_profile: str, stock: dict, hypothesis: str, fields: dict) -> dict:
    """Score a thesis 0-10. A thesis citing numbers that are on neither the card
    nor the profile scores 0 without a judge call."""
    card = item_text(stock)
    check = verify(fields, card, investor_profile)
    if not check["passed"]:
        return {"score": 0, "verifier": check,
                "reasoning": f"Number check failed: {', '.join(check['unmatched'])} not on the card"}
    if JEV:
        answers = await jev.decide(JUDGE_MODEL, {"investor": investor_profile, "stock_card": card, "thesis": hypothesis},
                                   {k: {"type": "score", "instructions": q, "criteria": levels}
                                    for k, (q, levels) in JEV_RUBRIC.items()})
        result = {k: round(jev.expected_level(answers[k]), 3) for k in JEV_RUBRIC}
        result["score"] = round(sum(result.values()), 2)  # continuous 0-10
        result["verifier"] = check
        return result

    response = await acreate(client(),
        model=JUDGE_MODEL,
        reasoning_effort=REASONING_EFFORT,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": JUDGE_USER_PROMPT.format(
                investor_profile=investor_profile, stock_card=card, hypothesis=hypothesis)},
        ],
        response_format={"type": "json_object"},
    )
    result = loads_lenient(response.choices[0].message.content)
    for key, top in RUBRIC.items():
        result[key] = parse_judge_score(result.get(key), high=top)
    result["score"] = sum(result[k] for k in RUBRIC)
    result["verifier"] = check
    return result


MAX_PARSE_ATTEMPTS = 3


def _format_entry_plain(entry: dict, header: str) -> str:
    """Format a thesis entry with a free-form header (no scores)."""
    return f"""### {header}

**Stock card:**
{item_text(entry)}

**Thesis:**
{entry.get("hypothesis", "")}
"""


async def check_novelty_async(
    investor_profile: str,
    new_entry: dict,
    neighbors: list[tuple[float, dict]],
) -> dict:
    """Judge novelty against the nearest neighbors.

    Raise RuntimeError after MAX_PARSE_ATTEMPTS unusable replies.
    """
    existing_text = "\n".join(
        _format_entry_plain(nb, f"Existing thesis {i+1}")
        for i, (_sim, nb) in enumerate(neighbors)
    )
    new_text = _format_entry_plain(new_entry, "New thesis")
    if JEV:
        answers = await jev.decide(JUDGE_MODEL, {"investor": investor_profile, "existing_theses": existing_text,
                                                 "new_thesis": new_text}, {"is_novel": {
            "type": "noul",
            "instructions": "Is the new thesis a meaningfully different investment idea from every existing thesis? "
                            "Compare the return driver (valuation re-rating, earnings momentum, turnaround, secular "
                            "growth, defensive income, event), the main risk exposure, and the stock.",
            "criteria": {"true": "A distinct return driver or risk exposure, even on the same stock.",
                         "false": "The same driver and exposure as an existing thesis, on the same or another stock."}}})
        p = answers["is_novel"]["noul"]
        return {"is_novel": p > 0.5, "novelty_reasoning": f"Jev P(novel)={p:.2f}"}

    for _attempt in range(MAX_PARSE_ATTEMPTS):
        response = await acreate(client(),
            model=JUDGE_MODEL,
            reasoning_effort=REASONING_EFFORT,
            messages=[
                {"role": "system", "content": NOVELTY_CHECK_SYSTEM_PROMPT},
                {"role": "user", "content": NOVELTY_CHECK_USER_PROMPT.format(
                    investor_profile=investor_profile, existing_text=existing_text, new_text=new_text)},
            ],
            response_format={"type": "json_object"},
        )
        try:
            result = loads_lenient(response.choices[0].message.content)
        except ValueError:
            continue
        is_novel = parse_json_bool(result.get("is_novel"))
        if is_novel is not None:
            return {"is_novel": is_novel, "novelty_reasoning": result.get("novelty_reasoning", "")}
    raise RuntimeError(f"novelty check reply unreadable {MAX_PARSE_ATTEMPTS} times (model={JUDGE_MODEL})")


async def pairwise_duel_async(
    investor_profile: str,
    new_entry: dict,
    nearest_entry: dict,
) -> dict:
    """Compare the new entry and its nearest neighbor under shuffled A/B labels.

    Hide scores and which entry is new. Raise RuntimeError after
    MAX_PARSE_ATTEMPTS unusable replies.
    """
    pair = [(new_entry, True), (nearest_entry, False)]
    random.shuffle(pair)
    new_label = "A" if pair[0][1] else "B"
    duel_text = "\n".join(
        _format_entry_plain(entry, f"Thesis {label}")
        for label, (entry, _is_new) in zip("AB", pair)
    )
    if JEV:
        answers = await jev.decide(JUDGE_MODEL, {"investor": investor_profile, "theses": duel_text}, {"winner": {
            "type": "choice",
            "instructions": "Which thesis is the better idea for this investor, judged on how well the card's evidence "
                            "supports it, fit with the investor, concreteness of catalyst and falsifier, and risk handling?",
            "criteria": {"A": "Thesis A is stronger.", "B": "Thesis B is stronger."}}})
        winner = answers["winner"]["choice"]
        return {"new_wins": winner == new_label, "new_label": new_label,
                "comparison_reasoning": f"Jev P(A)={answers['winner']['probabilities'].get('A', 0):.2f}"}

    for _attempt in range(MAX_PARSE_ATTEMPTS):
        response = await acreate(client(),
            model=JUDGE_MODEL,
            reasoning_effort=REASONING_EFFORT,
            messages=[
                {"role": "system", "content": PAIRWISE_DUEL_SYSTEM_PROMPT},
                {"role": "user", "content": PAIRWISE_DUEL_USER_PROMPT.format(
                    investor_profile=investor_profile, duel_text=duel_text)},
            ],
            response_format={"type": "json_object"},
        )
        try:
            result = loads_lenient(response.choices[0].message.content)
        except ValueError:
            continue
        winner = str(result.get("winner", "")).strip().upper()
        if winner in ("A", "B"):
            return {
                "new_wins": winner == new_label,
                "comparison_reasoning": result.get("comparison_reasoning", ""),
                "new_label": new_label,
            }
    raise RuntimeError(f"duel reply unreadable {MAX_PARSE_ATTEMPTS} times (model={JUDGE_MODEL})")
