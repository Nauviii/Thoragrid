"""Integration tests for POST /feedback and GET /history, against the live stack.

These cover the one-feedback-per-interaction constraint, the verdict now surfaced on history
items so the UI can reflect it after a reload, and per-doctor scoping of the history list.

A text interaction is used as the subject rather than an image analysis: feedback and history
treat both identically, and /query is far cheaper than the full CNN + GradCAM + RAG pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest


def _create_interaction(client, headers) -> dict:
    """Run one cheap text interaction and return its response body."""
    response = client.post(
        "/query", json={"query": "Apa tanda radiologis utama pneumothorax?"}, headers=headers
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def interaction(client, doctor, cleanup_conversation):
    """One recorded interaction belonging to the throwaway doctor."""
    body = _create_interaction(client, doctor["headers"])
    cleanup_conversation(body["conversation_id"])
    return body


def test_feedback_is_recorded_and_echoed_back(client, doctor, interaction):
    """A submitted verdict comes back on the response with the interaction it belongs to."""
    response = client.post(
        "/feedback",
        json={"interaction_id": interaction["interaction_id"], "is_correct": True},
        headers=doctor["headers"],
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["interaction_id"] == interaction["interaction_id"]
    assert body["is_correct"] is True


def test_disagreement_keeps_the_reason(client, doctor, interaction):
    """The comment must survive the round trip; a disagreement without it has little review value."""
    response = client.post(
        "/feedback",
        json={
            "interaction_id": interaction["interaction_id"],
            "is_correct": False,
            "comment": "Missed the apical component.",
        },
        headers=doctor["headers"],
    )
    assert response.status_code == 200, response.text
    assert response.json()["comment"] == "Missed the apical component."


def test_second_feedback_on_same_interaction_is_rejected(client, doctor, interaction):
    """One feedback per interaction: the duplicate must 409 rather than overwrite."""
    first = client.post(
        "/feedback",
        json={"interaction_id": interaction["interaction_id"], "is_correct": True},
        headers=doctor["headers"],
    )
    assert first.status_code == 200

    second = client.post(
        "/feedback",
        json={"interaction_id": interaction["interaction_id"], "is_correct": False},
        headers=doctor["headers"],
    )
    assert second.status_code == 409


def test_feedback_on_unknown_interaction_is_404(client, doctor):
    """Feedback must not be attachable to an interaction that does not exist."""
    response = client.post(
        "/feedback",
        json={"interaction_id": "no-such-interaction", "is_correct": True},
        headers=doctor["headers"],
    )
    assert response.status_code == 404


def test_history_reports_no_feedback_before_any_is_given(client, doctor, interaction):
    """An untouched interaction must report None, so the UI offers the controls."""
    response = client.get("/history", headers=doctor["headers"])
    assert response.status_code == 200
    item = next(i for i in response.json()["items"] if i["id"] == interaction["interaction_id"])
    assert item["feedback"] is None


def test_history_reports_the_recorded_verdict(client, doctor, interaction):
    """After submitting, history must carry the verdict so a reload still shows it."""
    client.post(
        "/feedback",
        json={"interaction_id": interaction["interaction_id"], "is_correct": False},
        headers=doctor["headers"],
    )

    response = client.get("/history", headers=doctor["headers"])
    item = next(i for i in response.json()["items"] if i["id"] == interaction["interaction_id"])
    assert item["feedback"] is False


def test_history_is_scoped_to_the_owning_doctor(client, doctor, second_doctor, interaction):
    """One doctor's interactions must never appear in another doctor's history."""
    response = client.get("/history", headers=second_doctor["headers"])
    assert response.status_code == 200
    ids = [i["id"] for i in response.json()["items"]]
    assert interaction["interaction_id"] not in ids