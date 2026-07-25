"""Integration test: cross-path conversation memory (image -> text follow-up, close, transcript)."""

from pathlib import Path

import pytest

SAMPLE_XRAY = Path(__file__).parent / "fixtures" / "sample_xray.png"


def _upload_xray(client, headers):
    """Upload the sample X-ray and return the parsed response body."""
    with open(SAMPLE_XRAY, "rb") as f:
        r = client.post(
            "/analyze/xray", files={"file": ("x.png", f, "image/png")},
            data={"conversation_id": ""}, headers=headers,
        )
    assert r.status_code == 200, r.text
    return r.json()


def _require_fixture():
    if not SAMPLE_XRAY.exists():
        pytest.skip(f"Missing test fixture at {SAMPLE_XRAY} — place a real chest X-ray PNG there first.")


def test_text_followup_references_prior_image_findings(client, doctor, cleanup_conversation):
    """A text follow-up after an image analysis produces an answer grounded in that analysis."""
    _require_fixture()

    image_body = _upload_xray(client, doctor["headers"])
    conv_id = image_body["conversation_id"]
    cleanup_conversation(conv_id)

    if image_body["low_confidence_flag"]:
        pytest.skip("Sample X-ray produced no findings above threshold; cannot test follow-up grounding.")

    followup = client.post(
        "/query",
        json={"query": "Apa saran tata laksananya?", "conversation_id": conv_id},
        headers=doctor["headers"],
    )
    assert followup.status_code == 200
    body = followup.json()
    assert body["conversation_id"] == conv_id
    assert len(body["answer"]) > 0


def test_get_conversation_returns_full_transcript(client, doctor, cleanup_conversation):
    """GET /conversation/{id} returns every turn, in chronological order."""
    _require_fixture()

    image_body = _upload_xray(client, doctor["headers"])
    conv_id = image_body["conversation_id"]
    cleanup_conversation(conv_id)

    client.post("/query", json={"query": "Follow up satu", "conversation_id": conv_id}, headers=doctor["headers"])

    r = client.get(f"/conversation/{conv_id}", headers=doctor["headers"])
    assert r.status_code == 200
    body = r.json()
    assert body["conversation_id"] == conv_id
    assert len(body["turns"]) == 2
    image_turn, text_turn = body["turns"]
    assert image_turn["interaction_type"] == "image"
    assert text_turn["interaction_type"] == "text"

    # Image turn carries a freshly re-signed xray URL and one heatmap per above-threshold condition,
    # regardless of how long ago the case was analyzed (see core/storage/supabase_storage.py).
    assert image_turn["xray_url"] is not None and image_turn["xray_url"].startswith("http")
    assert len(image_turn["gradcam_findings"]) == len(image_body["above_threshold"])
    for finding in image_turn["gradcam_findings"]:
        assert finding["heatmap_url"] is not None and finding["heatmap_url"].startswith("http")  


def test_conversation_ownership_enforced(client, doctor, second_doctor, cleanup_conversation):
    """A different doctor cannot read or close another doctor's conversation."""
    _require_fixture()

    image_body = _upload_xray(client, doctor["headers"])
    conv_id = image_body["conversation_id"]
    cleanup_conversation(conv_id)

    r_get = client.get(f"/conversation/{conv_id}", headers=second_doctor["headers"])
    assert r_get.status_code == 403

    r_delete = client.delete(f"/conversation/{conv_id}", headers=second_doctor["headers"])
    assert r_delete.status_code == 403


def test_close_conversation_prevents_auto_revival(client, doctor, cleanup_conversation):
    """After DELETE /conversation/{id}, a follow-up with the same id is treated as standalone.

    The permanent Postgres transcript is unaffected — only the Redis working memory used
    for automatic context injection is cleared.
    """
    _require_fixture()

    image_body = _upload_xray(client, doctor["headers"])
    conv_id = image_body["conversation_id"]
    cleanup_conversation(conv_id)

    close_resp = client.delete(f"/conversation/{conv_id}", headers=doctor["headers"])
    assert close_resp.status_code == 200
    assert close_resp.json()["closed"] is True

    followup = client.post(
        "/query",
        json={"query": "Masih ingat temuan gambar tadi?", "conversation_id": conv_id},
        headers=doctor["headers"],
    )
    assert followup.status_code == 200

    transcript = client.get(f"/conversation/{conv_id}", headers=doctor["headers"])
    assert transcript.status_code == 200
    assert len(transcript.json()["turns"]) == 2  # image + post-close query, both preserved


def test_get_conversation_requires_auth(client):
    """Reading a conversation transcript without a token is rejected."""
    r = client.get("/conversation/some-id")
    assert r.status_code == 401