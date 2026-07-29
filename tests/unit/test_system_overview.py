"""Keep the assistant's description of itself tied to what the system actually is.

SYSTEM_OVERVIEW is the only source the model may use when asked what this assistant does.
Prose drifts from code silently: a condition gets added, a feedback category renamed, a
validation outcome introduced, and the description keeps confidently reporting the old shape
with nothing anywhere to contradict it. A clinician told the assistant measures something it
does not measure has been misled about the instrument, which is worse than a vague answer.

These tests fail when the two diverge.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.feedback_reasons import FEEDBACK_REASONS
from config.settings import settings
from core.llm.prompts import SYSTEM_OVERVIEW, TEXT_QA_SYSTEM


def test_every_covered_condition_is_named():
    """All 14 findings must be listed; a missing one reads as a capability the system lacks."""
    for condition in settings.gradcam_condition_colors:
        assert condition.replace("_", " ") in SYSTEM_OVERVIEW, condition


def test_no_condition_is_claimed_that_the_model_does_not_cover():
    """The count stated in prose must match the count the model actually classifies."""
    assert len(settings.gradcam_condition_colors) == 14
    assert "14 findings" in SYSTEM_OVERVIEW


def test_zone_count_matches_the_region_map():
    """The described zone count must match what GradCAM actually reports against."""
    assert len(settings.gradcam_zones) == 7
    assert "seven" in SYSTEM_OVERVIEW.lower()


def test_every_feedback_reason_is_described():
    """A reader offered five reasons should have been told those five exist."""
    lowered = SYSTEM_OVERVIEW.lower()
    for label in FEEDBACK_REASONS.values():
        head = label.lower().split(" or ")[0]
        assert head in lowered, label


def test_all_three_validation_outcomes_are_described():
    """Refuse, refuse-as-out-of-range, and accept-with-flag are distinct and all user-visible."""
    lowered = SYSTEM_OVERVIEW.lower()
    assert "not a chest radiograph" in lowered
    assert "outside the trained range" in lowered
    assert "flagged" in lowered


def test_limits_are_stated_not_softened():
    """The two limits most likely to be misread must appear explicitly."""
    lowered = SYSTEM_OVERVIEW.lower()
    assert "not evidence of absence" in lowered
    assert "not a diagnostic device" in lowered


def test_overview_is_embedded_in_the_prompt():
    """The description is useless unless the model actually receives it, and is told to rely on it.

    The grounding clause is matched on collapsed whitespace: the wording is what matters, and
    an assertion that breaks whenever a paragraph is re-wrapped teaches people to delete it.
    """
    assert SYSTEM_OVERVIEW in TEXT_QA_SYSTEM
    collapsed = " ".join(TEXT_QA_SYSTEM.split())
    assert "from the description below and from nothing else" in collapsed


def test_answer_shape_is_specified_rather_than_capped():
    """A length cap in the schema comment is what produced a four-sentence wall of prose.

    The model followed `<direct clinical answer, 2-5 sentences>` exactly, which left no room
    for the sub-headings a multi-part question needs. Shape rules replace the cap; if the cap
    ever returns, structure quietly disappears again with nothing failing.
    """
    assert "2-5 sentences" not in TEXT_QA_SYSTEM
    assert "SHAPE OF THE ANSWER" in TEXT_QA_SYSTEM
    assert "Markdown" in TEXT_QA_SYSTEM


def test_refusal_behaviour_is_required_in_self_description():
    """All three outcomes must be demanded, not just the idea that validation happens.

    The first attempt asked for "the three outcomes" as a list and got back a single clause —
    "validates that the image is a chest film and within the trained range" — which merges the
    two refusals and drops the flagged band entirely. The flagged band is the one a reader is
    most likely to meet, so it is asserted separately.
    """
    lowered = TEXT_QA_SYSTEM.lower()
    assert "refused as not a chest radiograph" in lowered
    assert "refused as outside the trained range" in lowered
    assert "flagged" in lowered and "accepted" in lowered
    assert "must not be" in lowered   # the instruction not to drop the third band


def test_limits_are_required_in_self_description():
    """The two limits most likely to be misread must be mandatory, not left to the question."""
    lowered = TEXT_QA_SYSTEM.lower()
    assert "not a diagnostic device" in lowered
    assert "not evidence a finding is absent" in lowered


def test_feedback_must_not_be_described_as_training_the_model():
    """There is no retraining loop; an answer claiming one describes a system that does not exist.

    The first generated description said feedback exists "to correct the model". Nothing in
    SYSTEM_OVERVIEW says that — the model inferred it, which is exactly the failure mode the
    grounding instruction is meant to prevent.
    """
    lowered = TEXT_QA_SYSTEM.lower()
    assert "does not train, correct, or improve the model" in lowered
    assert "no retraining loop" in lowered
    # The overview itself must not give the model the idea in the first place.
    assert "correct the model" not in SYSTEM_OVERVIEW.lower()