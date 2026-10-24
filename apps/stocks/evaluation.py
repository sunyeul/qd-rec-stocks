"""
The number check, then the Jev judge: the thesis score, the novelty check and the blind pairwise duel.
"""

import random

from . import jev
from .config import JUDGE_MODEL
from .items import item_text
from .verifier import verify

# Rubric parts as Jev Score questions, worst level first (level i = i points). They are
# diagnostics and a gate; the score itself is the probability of beating the market.
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

OUTPERFORM_QUESTION = {
    "type": "noul",
    "instructions": "Weighing the thesis against the skeptic's critique and using only the card, will this stock "
                    "beat the S&P 500 over the next 3 months?",
    "criteria": {"true": "It beats the S&P 500 over the next 3 months.",
                 "false": "It matches or trails the S&P 500 over the next 3 months."},
}


async def judge_score_async(investor_profile: str, stock: dict, hypothesis: str, fields: dict,
                            critique: str = "") -> dict:
    """Score a thesis 0-10 as 10 x P(beats the S&P 500 over 3 months), so the quality
    the loop optimises is the PoC's goal.

    A thesis citing numbers that are on neither the card nor the profile scores 0
    without a judge call, and one whose evidence Jev judges missing
    (evidence_match < 1) scores 0.
    """
    card = item_text(stock)
    check = verify(fields, card, investor_profile)
    if not check["passed"]:
        return {"score": 0, "verifier": check,
                "reasoning": f"Number check failed: {', '.join(check['unmatched'])} not on the card"}
    questions = {k: {"type": "score", "instructions": q, "criteria": levels} for k, (q, levels) in JEV_RUBRIC.items()}
    questions["outperform"] = OUTPERFORM_QUESTION
    answers = await jev.decide(JUDGE_MODEL, {"investor": investor_profile, "stock_card": card, "thesis": hypothesis,
                                             "skeptic_critique": critique or "(none)"}, questions)
    result = {k: round(jev.expected_level(answers[k]), 3) for k in JEV_RUBRIC}
    result["p_outperform"] = round(answers["outperform"]["noul"], 4)
    gated = result["evidence_match"] < 1
    result["score"] = 0.0 if gated else round(10 * result["p_outperform"], 3)
    result["reasoning"] = "Evidence judged missing" if gated else f"P(beats S&P 500 in 3 months)={result['p_outperform']:.2f}"
    result["verifier"] = check
    return result


def _format_entry_plain(entry: dict, header: str) -> str:
    """Format a thesis entry with a free-form header (no scores)."""
    critique = f"\n**Skeptic's critique:**\n{entry['critique']}\n" if entry.get("critique") else ""
    return f"""### {header}

**Stock card:**
{item_text(entry)}

**Thesis:**
{entry.get("hypothesis", "")}
{critique}"""


async def check_novelty_async(
    investor_profile: str,
    new_entry: dict,
    neighbors: list[tuple[float, dict]],
) -> dict:
    """Judge novelty against the nearest neighbors."""
    existing_text = "\n".join(
        _format_entry_plain(nb, f"Existing thesis {i+1}")
        for i, (_sim, nb) in enumerate(neighbors)
    )
    new_text = _format_entry_plain(new_entry, "New thesis")
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


async def pairwise_duel_async(
    investor_profile: str,
    new_entry: dict,
    nearest_entry: dict,
) -> dict:
    """Compare the new entry and its nearest neighbor under shuffled A/B labels,
    hiding scores and which entry is new."""
    pair = [(new_entry, True), (nearest_entry, False)]
    random.shuffle(pair)
    new_label = "A" if pair[0][1] else "B"
    duel_text = "\n".join(
        _format_entry_plain(entry, f"Thesis {label}")
        for label, (entry, _is_new) in zip("AB", pair)
    )
    answers = await jev.decide(JUDGE_MODEL, {"investor": investor_profile, "theses": duel_text}, {"winner": {
        "type": "choice",
        "instructions": "Which thesis is the better idea for this investor, judged on how well the card's evidence "
                        "supports it, fit with the investor, concreteness of catalyst and falsifier, and risk handling?",
        "criteria": {"A": "Thesis A is stronger.", "B": "Thesis B is stronger."}}})
    return {"new_wins": answers["winner"]["choice"] == new_label, "new_label": new_label,
            "comparison_reasoning": f"Jev P(A)={answers['winner']['probabilities'].get('A', 0):.2f}"}
