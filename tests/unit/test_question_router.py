"""Routing accuracy as a measured number rather than a hope.

The router decides whether a question is answered from the reading record or from clinical
literature. Getting it wrong is a quality failure, not a security one — the SQL agent runs
read-only, AST-validated and scoped to the caller's own rows however it is reached — so these
tests are about whether readers get the answer they asked for.

Both languages are covered on purpose: the assistant answers in English and Indonesian, and a
router that only read English would send every Indonesian analytics question to the clinical
path without anything failing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.routing.question_router import CLINICAL, RECORDS, UNCERTAIN, route

RECORD_QUESTIONS = [
    "How many cases per condition?",
    "What is the average confidence score for pneumothorax?",
    "How many findings did I mark as incorrect because the localization was off?",
    "How many Grad-CAM findings are misaligned with anatomical zones?",
    "Which condition is most common in my record?",
    "What is the total number of studies I have read?",
    "How often did I disagree with the model?",
    "Berapa banyak kasus per kondisi?",
    "Berapa banyak temuan yang saya tandai salah?",
    "Berapa rata-rata skor untuk efusi?",
]

CLINICAL_QUESTIONS = [
    "What causes pneumothorax?",
    "What are the symptoms of pulmonary edema?",
    "Explain the pathophysiology of emphysema",
    "Why does cardiomegaly appear on a frontal film?",
    "What is the differential for a solitary pulmonary nodule?",
    "Apa penyebab efusi pleura?",
    "Jelaskan tatalaksana untuk pneumonia",
    "Bagaimana emfisema bisa terjadi?",
    "Apa saja faktor risiko pneumothorax?",
    "Kenapa efusi pleura muncul pada gagal jantung?",
]

# Questions a person could reasonably read either way. Arbitration is the correct outcome —
# a confident answer here would mean the patterns are claiming more than they can support.
AMBIGUOUS_QUESTIONS = [
    "pneumothorax",
    "tell me about effusion",
    "How many types of pneumothorax are there?",
]


def test_record_questions_are_routed_to_the_record():
    """Every unambiguous record question must be decided by pattern, without a model call."""
    wrong = [q for q in RECORD_QUESTIONS if route(q)[0] != RECORDS]
    assert not wrong, f"routed away from the record: {wrong}"


def test_clinical_questions_are_routed_to_literature():
    """A clinical question must never reach the SQL agent when the patterns can tell."""
    wrong = [q for q in CLINICAL_QUESTIONS if route(q)[0] != CLINICAL]
    assert not wrong, f"routed away from clinical: {wrong}"


def test_indonesian_is_routed_as_well_as_english():
    """Half of each set is Indonesian; both halves must behave the same.

    Asserted separately because an English-only router passes the combined test whenever the
    English examples outnumber the Indonesian ones.
    """
    indonesian_record = [q for q in RECORD_QUESTIONS if "Berapa" in q]
    indonesian_clinical = [q for q in CLINICAL_QUESTIONS
                           if q.split()[0] in {"Apa", "Jelaskan", "Bagaimana", "Kenapa"}]
    assert indonesian_record and indonesian_clinical, "the Indonesian samples went missing"
    assert all(route(q)[0] == RECORDS for q in indonesian_record)
    assert all(route(q)[0] == CLINICAL for q in indonesian_clinical)


def test_ambiguous_questions_are_sent_to_arbitration():
    """The patterns must decline rather than guess when a question is genuinely open."""
    decided = [q for q in AMBIGUOUS_QUESTIONS if route(q)[0] != UNCERTAIN]
    assert not decided, f"decided without enough evidence: {decided}"


def test_aggregate_openings_are_not_mistaken_for_clinical():
    """'What is the average ...' asks about the record, not about disease.

    This is a regression test. The clinical pattern for 'what is' claimed the sentence, and
    the tie sent a plainly-answerable record question to arbitration.
    """
    for q in ["What is the average confidence score for pneumothorax?",
              "What is the total number of studies I have read?"]:
        assert route(q)[0] == RECORDS


def test_routing_reports_its_evidence():
    """A surprising decision has to be explainable from a log, not reproduced by hand."""
    _, detail = route("How many cases per condition?")
    assert detail["by"] == "pattern"
    assert detail["record_hits"] >= 1
    assert "clinical_hits" in detail