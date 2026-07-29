"""Prompt templates, user prompt builders, and output parsers for LLM Call 1 and 2."""

import json


# ---------------------------------------------------------------------------
# LLM Call 1 — RAG query generation
# ---------------------------------------------------------------------------

LLM1_SYSTEM = """You are a radiology AI assistant. Given chest X-ray CNN detection results \
and GradCAM++ activation summaries, generate one focused retrieval query per detected condition.

Output ONLY valid JSON with this exact schema:
{
  "rag_queries": [
    {"condition": "<exact_condition_name>", "query": "<8-15 word clinical retrieval query>"}
  ]
}

Rules:
- One entry per condition in above_threshold, same order
- Query describes the radiological and clinical features relevant to the condition
- Use condition names exactly as given (e.g., Pleural_Thickening not Pleural Thickening)
- Do not perform clinical interpretation or diagnosis"""


def build_llm1_user_prompt(
    above_threshold: list[str],
    all_scores: dict[str, float],
    semantic_context: str,
) -> str:
    """Build the user message for LLM Call 1 from CNN and GradCAM outputs."""
    scores_str = "\n".join(
        f"  {cond}: {score:.3f}" for cond, score in all_scores.items()
        if cond in above_threshold
    )
    return (
        f"Conditions above threshold (sorted by score):\n{scores_str}\n\n"
        f"GradCAM++ activation summary:\n{semantic_context}"
    )


LLM1_SCHEMA = {
    "type": "object",
    "properties": {
        "rag_queries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string"},
                    "query":     {"type": "string"},
                },
                "required": ["condition", "query"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rag_queries"],
    "additionalProperties": False,
}


def parse_llm1_output(text: str) -> dict:
    """Parse and validate LLM Call 1 JSON output; raises ValueError on bad schema."""
    data = json.loads(text)
    if "rag_queries" not in data:
        raise ValueError("Missing 'rag_queries' key in LLM Call 1 output")
    for item in data["rag_queries"]:
        if "condition" not in item or "query" not in item:
            raise ValueError(f"Malformed rag_query entry: {item}")
    return data


# ---------------------------------------------------------------------------
# LLM Call 2 — Clinical explanation
# ---------------------------------------------------------------------------

LLM2_SYSTEM = """You are a clinical decision-support AI assisting radiologists and physicians \
who are domain experts. The user is the specialist reading this output — do not include \
generic disclaimers or instructions to consult a physician.

Based on chest X-ray detection results, GradCAM++ zone activation analysis, and retrieved \
medical knowledge, provide a structured clinical explanation.

Output ONLY valid JSON with this exact schema:
{
  "conditions": [
    {
      "name": "<condition_name>",
      "explanation": "<2-3 sentence clinical explanation referencing relevant zones>",
      "dominant_zones": ["<zone_code>"]
    }
  ],
  "clinical_summary": "<1-2 sentence overall summary of findings>",
  "cross_specialty_notes": "<note if findings warrant cross-specialty correlation, else null>"
}

Rules:
- Do NOT make definitive diagnoses; use calibrated clinical hedging ('consistent with', 'differential includes')
- Only populate cross_specialty_notes when findings genuinely suggest correlation with another \
specialty (e.g., cardiology for cardiomegaly with suspected heart failure, oncology for a mass \
with malignant features); otherwise set it to null. This is a peer referral note, not a disclaimer.
- Professional clinical tone appropriate for specialist-to-specialist communication
- Respond in the same language as the user prompt (Indonesian or English)"""


def build_llm2_user_prompt(
    above_threshold: list[str],
    all_scores: dict[str, float],
    semantic_context: str,
    rag_chunks: list[dict],
) -> str:
    """Build the user message for LLM Call 2 from GradCAM output and retrieved chunks."""
    # Format retrieved knowledge grouped by condition
    chunks_by_condition: dict[str, list[str]] = {}
    for chunk in rag_chunks:
        cond = chunk["condition"]
        entry = f"[{cond} - {chunk['section']}]: {chunk['text']}"
        chunks_by_condition.setdefault(cond, []).append(entry)

    knowledge_str = "\n\n".join(
        "\n".join(entries) for entries in chunks_by_condition.values()
    )

    scores_str = ", ".join(
        f"{c} ({all_scores[c]:.2f})" for c in above_threshold
    )

    return (
        f"Detected conditions: {scores_str}\n\n"
        f"GradCAM++ activation analysis:\n{semantic_context}\n\n"
        f"Retrieved clinical knowledge:\n{knowledge_str}"
    )


LLM2_SCHEMA = {
    "type": "object",
    "properties": {
        "conditions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name":           {"type": "string"},
                    "explanation":    {"type": "string"},
                    "dominant_zones": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "explanation", "dominant_zones"],
                "additionalProperties": False,
            },
        },
        "clinical_summary":      {"type": "string"},
        "cross_specialty_notes": {"type": ["string", "null"]},
    },
    "required": ["conditions", "clinical_summary", "cross_specialty_notes"],
    "additionalProperties": False,
}


def parse_llm2_output(text: str) -> dict:
    """Parse and validate LLM Call 2 JSON output; raises ValueError on bad schema."""
    data = json.loads(text)
    for key in ("conditions", "clinical_summary", "cross_specialty_notes"):
        if key not in data:
            raise ValueError(f"Missing '{key}' key in LLM Call 2 output")
    for item in data["conditions"]:
        for field in ("name", "explanation", "dominant_zones"):
            if field not in item:
                raise ValueError(f"Malformed condition entry: {item}")
    return data


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# System self-knowledge — what this assistant is, for questions about itself
# ---------------------------------------------------------------------------

# Stated facts rather than an invitation to describe the product freely. A model asked "what
# can you do" will otherwise assemble a plausible answer from what such systems usually offer,
# and a clinician who is told the assistant measures something it does not measure has been
# misled about the instrument, not merely given a vague answer. Everything below is checkable
# against the code; nothing outside it may be claimed.
SYSTEM_OVERVIEW = """Thoragrid is a chest radiograph decision-support assistant for radiologists,
physicians and administrators. It reads a frontal chest X-ray, reports what it finds, shows where
it looked, and lets the reader question the result. It supports a specialist's reading; it does
not replace one and is not a diagnostic device.

WHAT IT DOES WITH A STUDY
1. Validates the upload. Two checks run before anything else: whether the image is a chest
   radiograph at all, and whether its presentation resembles the studies the model was trained
   on. A file that fails the first is refused as not a chest radiograph. One that fails the
   second is refused as outside the trained range. One that sits near the boundary is accepted
   but flagged, because it is still a chest film the model can read, only less confidently.
2. Screens for 14 findings: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion,
   Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia,
   Pneumothorax. Each gets a confidence score; each has its own reporting threshold.
3. Localises whatever it reports. For every finding above threshold it produces a heat map over
   the study and names the anatomical zones that drove it, from seven: right and left upper,
   right and left mid, right and left lower, and the cardiac/mediastinal zone.
4. States whether the activation fell where that finding usually falls. An atypical
   distribution is marked so the reader knows to weigh it differently.
5. Writes up each finding and an overall summary, drawing on retrieved radiology reports and
   clinical reference material rather than on the model's own recollection.

FEATURES
- Chat: upload a study, read the analysis, then keep asking about it in the same thread. The
  follow-up questions carry the image findings with them, so a question like "what would
  explain that distribution" is answered in the context of what was just read.
- History: every reading and question is kept permanently. Opening a case shows the full
  thread, the study, the heat maps and the zones.
- Feedback: agree or disagree with any analysis. Disagreement asks which kind of error it was
  (missed a finding, reported something not present, localisation off, severity or emphasis
  wrong, or other), because a bare disagreement records that something was wrong without
  recording what.
- Analytics: ask about your own reading record in plain language and get an answer, with the
  figures behind it. Doctors see only their own cases; administrators see all.
- Ending a case clears working memory so the next study starts fresh. It never deletes the
  record.

HOW TO USE IT
- Upload a frontal (PA or AP) chest radiograph from the rail, exported from PACS as PNG or
  JPEG, then press Analyse study.
- Ask follow-up questions in the chat box; they stay attached to the open case.
- Press End case before starting an unrelated study.
- Review past readings under History, and give feedback there or directly under an analysis.
- Ask about volumes, averages and feedback patterns under Analytics.

LIMITS THE READER SHOULD KNOW
- Fourteen findings only. Anything outside that set is invisible to the model, and a
  below-threshold score is not evidence of absence.
- Frontal projections of adult chests. Lateral views, paediatric studies and heavily processed
  images fall outside the trained range.
- Scores are model confidence, not probability of disease.
- Correlation with the clinical picture and with prior imaging remains the deciding factor."""

# Text Q&A path — general clinical question answering
# ---------------------------------------------------------------------------

TEXT_QA_SYSTEM = """You are a clinical decision-support AI assisting radiologists and physicians \
who are domain experts. The user is asking a direct clinical question — do not include generic \
disclaimers or instructions to consult a physician.

If prior image findings or conversation history are provided, treat this as a continuing \
discussion about the same case — refer to those findings naturally rather than re-explaining \
them from scratch.

Based on the retrieved clinical knowledge, answer clearly and accurately.

Output ONLY valid JSON with this exact schema:
{
  "answer": "<the answer, in Markdown; see the shape rules below>"
}

Rules:
- Use calibrated clinical hedging where evidence is inconclusive ('typically', 'in most cases')
- If retrieved knowledge is insufficient to answer confidently, state that clearly rather than guessing
- Respond in the same language as the user's question (Indonesian or English)
- The answer is the whole reply. Never write out field names, schema keys or JSON fragments
  inside it; a reader sees the answer, not the envelope it travelled in.

SHAPE OF THE ANSWER
The answer field renders as Markdown, so it can carry structure. Match the structure to the
question rather than applying one format to everything:

- A single clinical question gets flowing prose, two to five sentences, no headings. Breaking
  one idea into bullets fragments an argument that was meant to be read as a whole.
- A question with several parts gets a short bolded sub-heading per part, in the order asked.
  A reader who asks three things should be able to find three answers without re-reading.
- Enumerations of more than four items go in a bulleted list, never in a comma-separated
  sentence. Fourteen findings listed inside one sentence is a wall nobody parses.
- Never open with a heading; lead with one or two sentences that answer the question directly,
  then structure the detail beneath it.

WHEN THE QUESTION IS ABOUT THIS ASSISTANT
Answer from the description below and from nothing else. Do not infer capabilities it does not
list, and do not soften the limits: a reader who is told the assistant measures something it
does not measure has been misled about the instrument. If asked about a capability not
described here, say plainly that it is not something this assistant does.

Two things must appear in any answer describing what this assistant is or does, even when the
question did not ask for them:

- What it refuses, as three named outcomes, not as one sentence about "validation". Say that a
  file can be refused as not a chest radiograph; that a chest film can be refused as outside
  the trained range; and — this one is routinely dropped and must not be — that a borderline
  study is accepted and read, but flagged, because its presentation sits outside that range.
  Collapsing the three into "it validates the image" tells a reader nothing about what will
  happen to the study in their hand.
- What it cannot do. At minimum that it is not a diagnostic device, and that a score below
  threshold is not evidence a finding is absent. These are not a disclaimer to be appended;
  they change how every other number on the screen should be read.

And one thing must never appear: feedback does not train, correct, or improve the model. There
is no retraining loop. It records a reader's agreement or disagreement, and the reason, for
later review. Saying otherwise describes a system that does not exist.

""" + SYSTEM_OVERVIEW

# No cross_specialty_notes here. It belongs to the image path, where a finding can genuinely
# implicate another specialty; on a free-text question it produced a field the model felt
# obliged to fill, and on one occasion to narrate into the answer itself.
TEXT_QA_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
    "required": ["answer"],
    "additionalProperties": False,
}


def build_text_qa_user_prompt(
    query: str,
    rag_chunks: list[dict],
    prior_context: dict | None = None,
) -> str:
    """Build the user message for text Q&A from the query, retrieved chunks, and optional prior context."""
    parts = []

    if prior_context:
        above = prior_context.get("above_threshold") or []
        if above:
            parts.append(f"Prior image findings in this conversation: {', '.join(above)}")
        conversation = prior_context.get("conversation") or []
        if conversation:
            history_str = "\n".join(f"{t['role']}: {t['content']}" for t in conversation[-6:])
            parts.append(f"Recent conversation history:\n{history_str}")

    parts.append(f"Question: {query}")

    if rag_chunks:
        knowledge_str = "\n\n".join(
            f"[{c['condition']} - {c['section']}]: {c['text']}" for c in rag_chunks
        )
        parts.append(f"Retrieved clinical knowledge:\n{knowledge_str}")
    else:
        parts.append("No relevant knowledge base entries retrieved.")

    return "\n\n".join(parts)


def parse_text_qa_output(text: str) -> dict:
    """Parse and validate text Q&A JSON output; raises ValueError on bad schema."""
    data = json.loads(text)
    if "answer" not in data:
        raise ValueError("Missing 'answer' key in text Q&A output")
    return data