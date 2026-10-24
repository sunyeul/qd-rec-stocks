"""
Prompt templates for the stock idea recommender: zero-shot and few-shot thesis
generation and the skeptic's critique. The judge's questions are in evaluation.py.
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
- Make the strongest honest case; if the card gives no real case, say so in the thesis.
- Open the thesis with this stock's specific return driver, not a stock-agnostic phrase such as
  "X presents a compelling opportunity to outperform"."""

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


# Skeptic: the strongest case against the thesis, written before the judge sees it
# (bull/bear debate as in TradingAgents). Not number-checked; the judge weighs it.
CRITIQUE_SYSTEM_PROMPT = """You are a skeptical short seller reviewing a long thesis. You argue against it, honestly."""

CRITIQUE_USER_PROMPT = """## Stock card:
{stock_card}

## Long thesis:
{hypothesis}

---

Write the strongest honest case against buying this stock now, using only the card.
Point to card lines the thesis ignores or misreads, crowded or priced-in arguments, and what could
go wrong within the horizon. Do not write numbers that are not on the card.

Respond in this EXACT JSON format:

{{
  "critique": "3-5 short bullet points in one string, each starting with '- '"
}}"""
