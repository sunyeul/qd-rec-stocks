"""Zero-shot and few-shot thesis generation, with lineage parsed by shared.lineage.

The proposer answers in fields (thesis, evidence, catalyst, risks, falsifier,
horizon). The archive keeps them as entry["thesis_fields"] and their rendered
text as entry["hypothesis"], the string the shared code expects.
"""

from shared.lineage import fewshot_schema, parse_evolution
from shared.llm import acreate, agenerate_payload, client, json_schema_format, loads_lenient

from .config import CRITIQUE_MODEL, PROPOSER_MODEL, PROPOSER_REASONING_EFFORT as REASONING_EFFORT
from .items import item_text
from .prompts import (
    CRITIQUE_SYSTEM_PROMPT,
    CRITIQUE_USER_PROMPT,
    HYPOTHESIS_FEWSHOT_SYSTEM_PROMPT,
    HYPOTHESIS_FEWSHOT_USER_PROMPT,
    HYPOTHESIS_SYSTEM_PROMPT,
    HYPOTHESIS_USER_PROMPT,
)

THESIS_PROPERTIES = {
    "thesis": {"type": "string"},
    "evidence": {"type": "array", "items": {"type": "string"}},
    "catalyst": {"type": "string"},
    "risks": {"type": "array", "items": {"type": "string"}},
    "falsifier": {"type": "string"},
    "horizon": {"type": "string"},
}
_SCHEMA = {
    "type": "object",
    "properties": THESIS_PROPERTIES,
    "required": list(THESIS_PROPERTIES),
    "additionalProperties": False,
}
_FEWSHOT_SCHEMA = fewshot_schema("thesis")
_FEWSHOT_SCHEMA["properties"].update(THESIS_PROPERTIES)
_FEWSHOT_SCHEMA["required"] = list(_FEWSHOT_SCHEMA["properties"])


def thesis_fields(result: dict) -> dict:
    """The thesis fields of a proposer reply, lists as lists of non-empty strings."""
    out = {}
    for key, spec in THESIS_PROPERTIES.items():
        value = result.get(key)
        if spec["type"] == "array":
            value = value if isinstance(value, list) else [value] if value else []
            out[key] = [str(v).strip() for v in value if str(v).strip()]
        else:
            out[key] = str(value or "").strip()
    return out


def render_hypothesis(fields: dict) -> str:
    bullets = lambda xs: "\n".join(f"- {x}" for x in xs) or "- (none)"
    return (f"Thesis: {fields['thesis']}\nEvidence:\n{bullets(fields['evidence'])}\n"
            f"Catalyst: {fields['catalyst']}\nRisks:\n{bullets(fields['risks'])}\n"
            f"Falsifier: {fields['falsifier']}\nHorizon: {fields['horizon']}")


async def _generate_once(investor_profile: str, stock: dict) -> dict:
    response = await acreate(client(),
        model=PROPOSER_MODEL,
        reasoning_effort=REASONING_EFFORT,
        messages=[
            {"role": "system", "content": HYPOTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": HYPOTHESIS_USER_PROMPT.format(
                investor_profile=investor_profile, stock_card=item_text(stock))},
        ],
        response_format=json_schema_format("stock_thesis", _SCHEMA),
    )
    return loads_lenient(response.choices[0].message.content)


async def generate_hypothesis_async(investor_profile: str, stock: dict) -> tuple[str, dict]:
    """Return (hypothesis text, thesis fields) for a stock (zero-shot)."""
    result, _thesis = await agenerate_payload(lambda: _generate_once(investor_profile, stock), "thesis")
    fields = thesis_fields(result)
    return render_hypothesis(fields), fields


async def critique_async(stock: dict, hypothesis: str) -> str:
    """The skeptic's case against the thesis, written by CRITIQUE_MODEL."""
    result, critique = await agenerate_payload(lambda: _critique_once(stock, hypothesis), "critique")
    return critique


async def _critique_once(stock: dict, hypothesis: str) -> dict:
    response = await acreate(client(),
        model=CRITIQUE_MODEL,
        reasoning_effort=REASONING_EFFORT,
        messages=[
            {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
            {"role": "user", "content": CRITIQUE_USER_PROMPT.format(stock_card=item_text(stock), hypothesis=hypothesis)},
        ],
        response_format={"type": "json_object"},
    )
    return loads_lenient(response.choices[0].message.content)


def _format_context(entries: list[dict]) -> str:
    if not entries:
        return "(No existing theses yet)"
    return "\n".join(
        f"### Thesis {i}\n**Stock card:**\n{item_text(e)}\n\n**Thesis:**\n{e.get('hypothesis', '')}\n"
        for i, e in enumerate(entries, 1)
    )


async def _generate_fewshot_once(investor_profile: str, stock: dict, context_entries: list[dict]) -> dict:
    response = await acreate(client(),
        model=PROPOSER_MODEL,
        reasoning_effort=REASONING_EFFORT,
        messages=[
            {"role": "system", "content": HYPOTHESIS_FEWSHOT_SYSTEM_PROMPT},
            {"role": "user", "content": HYPOTHESIS_FEWSHOT_USER_PROMPT.format(
                investor_profile=investor_profile,
                existing_theses=_format_context(context_entries),
                stock_card=item_text(stock))},
        ],
        response_format=json_schema_format("stock_thesis_fewshot", _FEWSHOT_SCHEMA),
    )
    return loads_lenient(response.choices[0].message.content)


async def generate_hypothesis_fewshot_async(
    investor_profile: str, stock: dict, context_entries: list[dict]
) -> tuple[str, dict, dict]:
    """Return (hypothesis text, thesis fields, evolution) with parent entry IDs."""
    result, _thesis = await agenerate_payload(
        lambda: _generate_fewshot_once(investor_profile, stock, context_entries), "thesis"
    )
    fields = thesis_fields(result)
    return render_hypothesis(fields), fields, parse_evolution(result, context_entries)
