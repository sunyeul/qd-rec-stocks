"""
Prompt templates for the stock idea recommender: zero-shot and few-shot thesis
generation, the judge's rubric score, the novelty check and the blind pairwise duel.
"""

_THESIS_RULES = """Write an investment thesis for buying this stock now, for this investor.

Rules:
- Use only the information on the stock card. Every number you write in "thesis",
  "evidence", "catalyst" or "risks" must be copied from the card (rounding is fine).
  Theses that state numbers not on the card are rejected automatically.
- Do not use knowledge of events or prices after the card's snapshot date.
- "evidence": 2-4 bullets, each citing a specific line of the card.
- "catalyst": a concrete event or development expected within the horizon that would make the market re-price the stock.
- "risks": 1-3 bullets.
- "falsifier": an observable condition that would prove the thesis wrong (thresholds allowed here).
- "horizon": the expected holding period, within the investor's horizon.
- Make the strongest honest case; if the card gives no real case, say so in the thesis."""

_THESIS_FIELDS = """  "thesis": "2-4 sentences: why this stock should beat the market over the horizon",
  "evidence": ["card fact 1", "card fact 2"],
  "catalyst": "...",
  "risks": ["..."],
  "falsifier": "...",
  "horizon": "..." """


# Thesis generation, zero-shot (seed archive, Novelty)
HYPOTHESIS_SYSTEM_PROMPT = """You are an equity analyst who writes investment theses grounded in the data you are given."""

HYPOTHESIS_USER_PROMPT = """## Investor:
{investor_profile}

---

## Stock card:
{stock_card}

---

""" + _THESIS_RULES + """

Respond in this EXACT JSON format:

{{
""" + _THESIS_FIELDS + """
}}"""


# Thesis generation, few-shot (Novelty+Evo, QD-Rec)
HYPOTHESIS_FEWSHOT_SYSTEM_PROMPT = HYPOTHESIS_SYSTEM_PROMPT

HYPOTHESIS_FEWSHOT_USER_PROMPT = """## Investor:
{investor_profile}

---

## Existing Theses (for inspiration only; their cards are not the target):

{existing_theses}

---

## Stock card to evaluate:
{stock_card}

---

""" + _THESIS_RULES + """

Use the existing theses as inspiration. You may:
- **Mutate**: Take the investment angle of ONE thesis and adapt it to this stock
- **Crossover**: Combine elements from TWO OR MORE theses into a new angle
- **Explore**: Propose a different angle not yet covered by any thesis

Numbers from the existing theses' cards do not count as being on this card.
Report which operator you used and the thesis number(s) you built on, using the
"Thesis N" numbers shown above (leave the source list empty for Explore).
Do not copy phrases verbatim from the existing theses.

Respond in this EXACT JSON format:

{{
  "operation": "mutate" | "crossover" | "explore",
  "source_examples": [thesis numbers you built on, e.g. [2] for mutate or [1, 3] for crossover; [] for explore],
  "evolution_note": "ONE sentence on how the example thesis/theses shaped this one. This is the ONLY field that may reference the examples; leave empty for explore.",
""" + _THESIS_FIELDS + """
}}"""


# Judge (rubric, 0-10 in all). Runs only on theses that passed the number check.
JUDGE_SYSTEM_PROMPT = """You are a skeptical investment committee member scoring stock theses."""

JUDGE_USER_PROMPT = """Score the following investment thesis.

## Investor:
{investor_profile}

## Stock card:
{stock_card}

## Thesis:
{hypothesis}

---

Score each criterion separately, using only the card as the factual record:
- "evidence_match" (0-3): the evidence is on the card and actually supports the conclusion; no cherry-picking of contradicting lines.
- "profile_fit" (0-3): the idea fits the investor's horizon, objective, risk limit, constraints and holdings.
- "risk_falsifier" (0-2): the main risks are named and the falsifier is observable and specific.
- "catalyst" (0-2): the catalyst is concrete and plausible within the horizon.

A thesis that argues against buying, or whose case is generic enough to fit any stock, must score low on every criterion.

Respond in this EXACT JSON format:

{{
  "reasoning": "Few sentences summarizing your assessment",
  "evidence_match": <integer 0-3>,
  "profile_fit": <integer 0-3>,
  "risk_falsifier": <integer 0-2>,
  "catalyst": <integer 0-2>
}}"""


# Novelty check, and the blind pairwise duel run only when the new thesis is not novel
NOVELTY_CHECK_SYSTEM_PROMPT = """You are a portfolio manager who checks whether a new stock idea adds a genuinely different theme to a watchlist."""

NOVELTY_CHECK_USER_PROMPT = """## Investor:
{investor_profile}

---

## Existing Theses:

{existing_text}

---

## New Thesis to Assess:

{new_text}

---

Assess whether the NEW thesis is a meaningfully different investment idea from the existing ones:
- Compare the return driver (e.g. valuation re-rating, earnings momentum, turnaround, secular growth, defensive income, event or spin-off), the main risk exposure, and the stock or industry.
- Evaluate strictly: the same stock with the same driver is NOT novel, and a different stock riding the same theme with the same risks is NOT novel. Only a distinct driver or exposure counts.

Respond in the following JSON format:

{{
  "novelty_reasoning": "1-2 sentences explaining whether the new thesis is a genuinely different idea.",
  "is_novel": true or false
}}"""

PAIRWISE_DUEL_SYSTEM_PROMPT = """You are a skeptical investment committee member who compares two stock theses and decides which is stronger."""

PAIRWISE_DUEL_USER_PROMPT = """## Investor:
{investor_profile}

---

## Theses to Compare:

{duel_text}

---

Which thesis is the better idea for this investor? Judge on the case alone, by:
- How well the card's evidence supports it
- Fit with the investor's horizon, objective and constraints
- Concreteness of the catalyst and the falsifier
- How well the risks are handled

You must pick exactly one winner; ties are not allowed.

Respond in the following JSON format:

{{
  "comparison_reasoning": "2-3 sentences explaining the key differentiators and your decision.",
  "winner": "A" or "B"
}}"""
