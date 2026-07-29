"""CLIP-based chest X-ray input validator with two-layer validation.

Layer 1 — Zero-shot prompt scoring (Radford et al., 2021):
    Computes cosine similarity between image embedding and valid/invalid
    text prompts. Rejects if the image scores higher on invalid prompts.

Layer 2 — Prototype-based anomaly detection (Snell et al., 2017):
    Computes Euclidean distance from image embedding to NIH ChestX-ray14
    prototype centroid. Rejects if distance exceeds calibrated threshold.
    Euclidean distance is used per Snell et al. recommendation over cosine.

References:
  - Radford et al. (2021): Learning Transferable Visual Models From Natural
    Language Supervision, ICML 2021.
  - Snell et al. (2017): Prototypical Networks for Few-shot Learning,
    NeurIPS 2017.
"""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

import torch
import clip

from config.settings import settings


@dataclass
class ValidationResult:
    """Output of the CLIP validation pipeline.

    `is_valid` covers both accepted outcomes: a study inside the trained range and one that is
    readable but presented outside it. The second is distinguished by `quality_flagged` rather
    than by refusing it, because such an image is still a chest radiograph and the CNN still
    reads the right anatomy — blocking it costs a real study, while flagging it costs nothing.
    """
    is_valid:        bool
    layer1_passed:   bool
    layer2_passed:   bool   # within the reject boundary; a flagged image still passes
    quality_flagged: bool   # accepted, but outside the range the CNN was trained on
    code:            str    # valid | quality_warning | not_a_chest_xray | outside_training_distribution
    reason:          str
    valid_score:     float  # mean cosine sim vs valid prompts
    invalid_score:   float  # mean cosine sim vs invalid prompts
    distance:        float  # Euclidean distance to NIH centroid
    warn_threshold:  float
    reject_threshold: float


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_clip_model = None
_preprocess = None
_prototype = None


def _load_clip() -> tuple:
    """Load CLIP model once and cache in module scope."""
    global _clip_model, _preprocess
    if _clip_model is None:
        _clip_model, _preprocess = clip.load(
            settings.clip_model_name, device=_DEVICE
        )
        _clip_model.eval()
    return _clip_model, _preprocess


def _load_prototype() -> dict:
    """Load the NIH prototype centroid from the JSON artifact.

    The artifact's own `threshold` field is deliberately ignored. It was derived as
    mean + 1.5 sigma over the calibration draw, which fixed how many calibration images fell
    outside and said nothing about anything else; measured on held-out studies the same cut
    refused 10.4% of genuine chest films. The operating point now lives in settings, where it
    can be retuned from measurement without recomputing the centroid.
    """
    global _prototype
    if _prototype is None:
        with open(settings.clip_prototype) as f:
            data = json.load(f)
        _prototype = {"centroid": np.array(data["centroid"], dtype=np.float32)}
    return _prototype


def _encode_image(image: Image.Image) -> np.ndarray:
    """Encode PIL image to L2-normalized CLIP embedding (512,)."""
    model, preprocess = _load_clip()
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten().astype(np.float32)


def _encode_texts(prompts: list[str]) -> np.ndarray:
    """Encode text prompts to L2-normalized CLIP embeddings (N, 512)."""
    model, _ = _load_clip()
    tokens = clip.tokenize(prompts).to(_DEVICE)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype(np.float32)


def _layer1_prompt_check(image_emb: np.ndarray) -> tuple[bool, float, float]:
    """Layer 1: zero-shot prompt scoring via cosine similarity.

    Returns (passed, valid_score, invalid_score).
    Radford et al. (2021): cosine similarity in joint embedding space
    is the standard zero-shot classification metric for CLIP.
    """
    valid_embs = _encode_texts(settings.clip_valid_prompts)
    invalid_embs = _encode_texts(settings.clip_invalid_prompts)

    valid_score = float(np.dot(valid_embs, image_emb).mean())
    invalid_score = float(np.dot(invalid_embs, image_emb).mean())

    passed = valid_score > invalid_score
    return passed, valid_score, invalid_score


def _layer2_distance(image_emb: np.ndarray) -> float:
    """Layer 2: distance from the image to the NIH ChestX-ray14 prototype centroid.

    Note that with L2-normalized embeddings this ranks identically to cosine similarity:
    ||q - c||^2 = 1 - 2(q.c) + ||c||^2 is strictly decreasing in q.c, so the choice between
    the two metrics changes nothing here. Measured separation is nonetheless near-perfect
    against non-chest radiographs (AUROC 1.000) and non-medical images (AUROC 1.000).

    Banding is left to validate(); this returns the raw distance so the two thresholds are
    applied in one place.
    """
    centroid = _load_prototype()["centroid"]
    return float(np.linalg.norm(image_emb - centroid))


def validate(image: Image.Image) -> ValidationResult:
    """Run two-layer CLIP validation on a PIL image.

    Layer 1 asks whether this is the right kind of image; layer 2 asks whether it resembles
    what the CNN was trained on. They are orthogonal, and only the second can catch a chest
    film that is genuinely a chest film but unreadable — an inverted greyscale study passes
    any prompt classifier and still breaks the model.

    Args:
        image: Input PIL image from user upload.

    Returns:
        ValidationResult with layer-wise decisions, scores, and the band it fell into.
    """
    image_emb = _encode_image(image)

    layer1_passed, valid_score, invalid_score = _layer1_prompt_check(image_emb)
    distance = _layer2_distance(image_emb)

    warn = settings.clip_warn_threshold
    reject = settings.clip_reject_threshold

    layer2_passed = distance <= reject
    quality_flagged = layer2_passed and distance > warn
    is_valid = layer1_passed and layer2_passed

    if not layer1_passed:
        code = "not_a_chest_xray"
        reason = (
            f"Image does not resemble a chest X-ray "
            f"(valid_score={valid_score:.3f}, invalid_score={invalid_score:.3f})"
        )
    elif not layer2_passed:
        code = "outside_training_distribution"
        reason = (
            f"Image distribution inconsistent with NIH ChestX-ray14 "
            f"(distance={distance:.3f}, reject_threshold={reject:.3f})"
        )
    elif quality_flagged:
        code = "quality_warning"
        reason = (
            f"Chest X-ray accepted with reduced confidence in presentation "
            f"(distance={distance:.3f}, warn_threshold={warn:.3f})"
        )
    else:
        code = "valid"
        reason = "Valid chest X-ray"

    return ValidationResult(
        is_valid = is_valid,
        layer1_passed = layer1_passed,
        layer2_passed = layer2_passed,
        quality_flagged = quality_flagged,
        code = code,
        reason = reason,
        valid_score = round(valid_score, 4),
        invalid_score = round(invalid_score, 4),
        distance = round(distance, 4),
        warn_threshold = warn,
        reject_threshold = reject,
    )