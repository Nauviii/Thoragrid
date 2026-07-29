"""Real-weight tests for CLIP-based chest X-ray validator (two-layer)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import pytest

from config.settings import settings
from core.clip.validator import validate


@pytest.fixture(scope="module")
def prototype():
    """Real calibrated CLIP prototype (centroid + threshold) from clip_prototype.json."""
    with open(settings.clip_prototype) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def result(sample_xray_image):
    """Single validation run reused across assertions."""
    return validate(sample_xray_image)


def test_cosine_scores_within_valid_range(result):
    """Cosine similarity scores (Layer 1) must lie in [-1, 1]."""
    assert -1.0 <= result.valid_score <= 1.0
    assert -1.0 <= result.invalid_score <= 1.0


def test_distance_is_non_negative(result):
    """Euclidean distance to centroid (Layer 2) cannot be negative."""
    assert result.distance >= 0.0


def test_thresholds_come_from_settings_not_the_artifact(result):
    """The operating point is configuration, not a property of the centroid.

    The artifact still carries the mean + 1.5 sigma value it was calibrated with. That cut
    refused 10.4% of genuine chest films when measured on held-out studies, so it is
    deliberately ignored; reading it back here would re-couple retuning to recalibration.
    """
    assert result.warn_threshold == settings.clip_warn_threshold
    assert result.reject_threshold == settings.clip_reject_threshold
    assert result.warn_threshold < result.reject_threshold


def test_centroid_dimension_matches_clip_embedding_size(prototype):
    """Centroid vector length must equal CLIP ViT-B/32 embedding dimension."""
    assert len(prototype["centroid"]) == 512


def test_is_valid_iff_both_layers_passed(result):
    """is_valid must be the logical AND of layer1_passed and layer2_passed."""
    assert result.is_valid == (result.layer1_passed and result.layer2_passed)


def test_layer1_fail_reason_mentions_chest_xray(result):
    """If Layer 1 fails, reason must explain non-resemblance to a chest X-ray."""
    if not result.layer1_passed:
        assert "chest X-ray" in result.reason


def test_layer2_fail_reason_mentions_distribution(result):
    """If Layer 1 passes but Layer 2 fails, reason must reference distribution mismatch."""
    if result.layer1_passed and not result.layer2_passed:
        assert "distribution" in result.reason

def test_is_valid_covers_both_accepted_bands(result):
    """A flagged study is accepted, not refused — that is the whole point of the middle band."""
    if result.quality_flagged:
        assert result.is_valid
        assert result.layer2_passed
        assert result.code == "quality_warning"


def test_code_and_flag_agree(result):
    """`code` and `quality_flagged` must never disagree; the UI branches on both."""
    assert (result.code == "quality_warning") == result.quality_flagged


def test_flag_band_matches_the_distance(result):
    """The flag must follow from where the distance actually fell, not be set independently."""
    if result.layer1_passed:
        in_band = result.warn_threshold < result.distance <= result.reject_threshold
        assert result.quality_flagged == in_band


def test_refusal_requires_exceeding_the_reject_threshold(result):
    """Layer 2 must only refuse past the reject boundary, never at the warn boundary."""
    if result.layer1_passed and not result.layer2_passed:
        assert result.distance > result.reject_threshold
        assert result.code == "outside_training_distribution"