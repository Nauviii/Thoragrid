"""Heuristic guardrails: prompt injection defense and clinical output validation."""

import re

MAX_QUERY_LENGTH = 1000

_INJECTION_PATTERNS = re.compile(
    r"ignore (all |any |previous |prior )?instructions"
    r"|disregard (the |your )?(system|above)"
    r"|you are now"
    r"|act as (if|a)"
    r"|reveal (your |the )?(system prompt|instructions)"
    r"|repeat (your |the )?(system prompt|instructions)"
    r"|new instructions\s*:"
    r"|\[system\]"
    r"|<\|system\|>",
    re.IGNORECASE,
)

_DEFINITIVE_DIAGNOSIS_PATTERNS = re.compile(
    r"\byou have\b"
    r"|\banda menderita\b|\banda terkena\b"
    r"|\bdiagnos[ia]s?\s*:\s*\w"
    r"|\bpasti (menderita|terkena)\b"
    r"|\bdefinitely (has|have|indicates)\b",
    re.IGNORECASE,
)


def check_prompt_injection(text: str) -> bool:
    """Return True if user input matches a known prompt injection pattern."""
    return bool(_INJECTION_PATTERNS.search(text))


def sanitize_user_input(text: str) -> str:
    """Trim whitespace and cap length before user text enters any prompt."""
    return text.strip()[:MAX_QUERY_LENGTH]


def _has_definitive_diagnosis(text: str | None) -> bool:
    """Return True if text contains definitive diagnostic-certainty language."""
    return bool(text) and bool(_DEFINITIVE_DIAGNOSIS_PATTERNS.search(text))


def validate_llm2_output(parsed: dict) -> bool:
    """Return True if LLM Call 2 output (image path) avoids definitive diagnostic-certainty language.

    This checks calibrated uncertainty, not audience literacy — expert users rely on
    the AI hedging appropriately rather than overclaiming, same as any clinical decision
    support tool. cross_specialty_notes is validated for absence of forbidden phrasing
    only when present; it is legitimately null when no cross-specialty correlation applies.
    """
    if any(_has_definitive_diagnosis(c.get("explanation", "")) for c in parsed.get("conditions", [])):
        return False
    return not _has_definitive_diagnosis(parsed.get("cross_specialty_notes"))


def validate_text_qa_output(parsed: dict) -> bool:
    """Return True if text Q&A output avoids definitive diagnostic-certainty language.

    Only the answer is checked. The text path no longer carries cross_specialty_notes — that
    field belongs to image analysis, where a finding can implicate another specialty.
    """
    return not _has_definitive_diagnosis(parsed.get("answer", ""))

def _canonical_key(name: str) -> str:
    """Reduce a condition name to a comparable form: lowercase, letters and digits only."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def normalise_condition_names(parsed: dict, above_threshold: list[str]) -> dict:
    """Map the names LLM Call 2 returned back onto the exact names the CNN reported.

    The interface joins a written explanation to its heat map and zones by condition name. If
    the model writes "Pulmonary Edema" where the CNN reported "Edema", that join silently
    misses and the finding renders as prose with no image and no zones beside it — the reader
    loses the evidence and is given no sign that anything is absent.

    Matching is confined to the conditions that actually crossed threshold, which keeps the
    candidate set to a handful and makes a substring match safe. A name that matches nothing,
    or matches more than one candidate, is left exactly as the model wrote it: a wrong join is
    worse than a missing one, because the explanation would then sit beside another finding's
    heat map.
    """
    keys = {_canonical_key(c): c for c in above_threshold}

    for condition in parsed.get("conditions", []):
        written = condition.get("name", "")
        key = _canonical_key(written)
        if key in keys:
            condition["name"] = keys[key]
            continue
        # The observed drift is expansion, never abbreviation: the model writes "Pulmonary
        # Edema" for "Edema", not "Pleural" for "Pleural_Thickening". So a candidate is
        # accepted only when the canonical name is contained in what the model wrote, and
        # only when exactly one candidate fits. Matching the other direction would resolve a
        # bare "Pleural" onto whichever pleural finding happened to be present.
        hits = [canon for k, canon in keys.items() if k and k in key]
        if len(hits) == 1:
            condition["name"] = hits[0]

    return parsed