"""Decide whether a question is about the reading record or about disease."""

import re

RECORDS = "records"
CLINICAL = "clinical"
UNCERTAIN = "uncertain"

# Phrases that only make sense asked of a stored record. Kept specific: a bare "how many"
# also opens "how many types of pneumothorax are there", which is a clinical question.
_RECORD_PATTERNS = [
    # counting and aggregating over cases
    r"\bhow many (cases|studies|scans|images|findings|readings|interactions|x-?rays)\b",
    r"\b(berapa|brp) (banyak )?(kasus|studi|gambar|temuan|pemeriksaan|interaksi)\b",
    r"\b(total|count|number) of (cases|studies|findings|readings)\b",
    r"\b(rata-?rata|average|mean|median) (confidence|score|skor|latency|latensi)\b",
    r"\b(most|least) (common|frequent) (condition|finding|diagnosis)\b",
    r"\b(kondisi|temuan) (paling|ter)(sering|banyak)\b",
    r"\b(per|by|berdasarkan) (condition|kondisi|doctor|dokter|zone|zona|month|bulan)\b",
    r"\b(distribution|distribusi|breakdown|trend|tren) (of|per|by)\b",
    # the reader asking about their own activity
    r"\b(my|our) (cases|studies|record|history|readings|feedback)\b",
    r"\b(kasus|riwayat|rekam|catatan) (saya|kami)\b",
    r"\b(did|have) i\b", r"\bi (marked|flagged|reported|reviewed|submitted)\b",
    r"\bsaya (tandai|menandai|laporkan|melaporkan|review|tinjau)\b",
    # the record's own vocabulary
    r"\b(flagged|marked) as (incorrect|wrong|misaligned)\b",
    r"\b(false[_ ]positive|missed[_ ]finding|wrong[_ ]location|wrong[_ ]severity)\b",
    r"\bmisaligned\b", r"\btidak selaras\b",
    r"\b(how often|seberapa sering)\b",
]

# Phrases that place the question on disease rather than on the record.
_CLINICAL_PATTERNS = [
    # "what is the average ..." is a record question; the aggregate openings are excluded
    # rather than left to fight the record patterns for the same sentence.
    r"\bwhat (causes|is|are) (?!the (average|mean|median|total|count|number|distribution|breakdown))",
    r"\bapa (penyebab|itu|saja)\b",
    r"\b(symptoms?|gejala|tanda) (of|dari|pada)\b",
    r"\b(treatment|terapi|tatalaksana|management|penanganan) (of|for|untuk|pada)\b",
    r"\b(pathophysiology|patofisiologi|prognosis|etiology|etiologi)\b",
    r"\b(differential|diagnosis banding|diagnosa banding)\b",
    r"\b(why|how) (does|do|is|are|can)\b",
    # Indonesian puts the subject between the question word and the verb, so the two are
    # matched across the gap rather than adjacently.
    r"\b(mengapa|kenapa|bagaimana)\b.{0,40}?\b(bisa|dapat|terjadi|menyebabkan|muncul)\b",
    r"\b(explain|jelaskan|describe|uraikan)\b",
    r"\b(risk factors?|faktor risiko|complications?|komplikasi)\b",
    r"\b(present|manifest|appear) (as|with|on)\b",
]

_RECORD_RE = [re.compile(p, re.I) for p in _RECORD_PATTERNS]
_CLINICAL_RE = [re.compile(p, re.I) for p in _CLINICAL_PATTERNS]

# A margin, not a bare comparison. One record phrase against zero clinical phrases is a weak
# signal; two, or one with nothing pulling the other way, is not.
_DECIDE_MARGIN = 1


def _score(question: str) -> tuple[int, int]:
    """Count how many record-shaped and clinical-shaped phrases a question contains."""
    return (sum(1 for r in _RECORD_RE if r.search(question)),
            sum(1 for r in _CLINICAL_RE if r.search(question)))


def route(question: str) -> tuple[str, dict]:
    """Classify a question as records, clinical, or uncertain.

    Returns the route and the evidence behind it, so a surprising decision can be explained
    from a log rather than reproduced by hand.
    """
    rec, clin = _score(question)
    detail = {"record_hits": rec, "clinical_hits": clin, "by": "pattern"}

    if rec and not clin:
        return RECORDS, detail
    if clin and not rec:
        return CLINICAL, detail
    if rec - clin >= _DECIDE_MARGIN + 1:
        return RECORDS, detail
    if clin - rec >= _DECIDE_MARGIN + 1:
        return CLINICAL, detail
    return UNCERTAIN, detail


# --- arbitration for the uncertain band ---------------------------------------------------

ROUTER_SYSTEM = """You route a question to one of two systems for a chest radiograph assistant.

records  - the question asks about the reader's own stored reading record: counts, averages,
           totals, distributions, what they marked or flagged, activity over time.
clinical - the question asks about disease, imaging appearance, causes, treatment, or how to
           interpret a finding in general.

Answer with the route only. If the question could plausibly be either, choose clinical: an
unnecessary clinical answer is a mild disappointment, while an unnecessary database query
returns a table to someone who wanted an explanation.

Output ONLY valid JSON: {"route": "records"} or {"route": "clinical"}"""

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {"route": {"type": "string", "enum": [RECORDS, CLINICAL]}},
    "required": ["route"],
    "additionalProperties": False,
}


def arbitrate(question: str) -> tuple[str, dict]:
    """Resolve an uncertain question with a model call, falling back to clinical on failure.

    The fallback direction is deliberate. If arbitration cannot run, answering clinically is
    the lesser failure: the reader gets prose instead of a table, rather than a table instead
    of prose.
    """
    import json

    from core.llm.client import call_groq

    try:
        raw = call_groq(ROUTER_SYSTEM, f"Question: {question}",
                        schema=ROUTER_SCHEMA, schema_name="question_route",
                        max_tokens=256)
        chosen = json.loads(raw)["route"]
        if chosen in (RECORDS, CLINICAL):
            return chosen, {"by": "arbitration"}
    except Exception as exc:
        return CLINICAL, {"by": "arbitration_failed", "error": str(exc)[:120]}
    return CLINICAL, {"by": "arbitration_unrecognised"}


def resolve(question: str) -> tuple[str, dict]:
    """Route a question, arbitrating only when the patterns cannot decide."""
    decided, detail = route(question)
    if decided != UNCERTAIN:
        return decided, detail
    chosen, arb = arbitrate(question)
    return chosen, {**detail, **arb}