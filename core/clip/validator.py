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
    """Output of CLIP validation pipeline."""
    is_valid:       bool
    layer1_passed:  bool
    layer2_passed:  bool
    code:           str     # machine-readable: valid | not_a_chest_xray | outside_training_distribution
    reason:         str
    valid_score:    float   # mean cosine sim vs valid prompts
    invalid_score:  float   # mean cosine sim vs invalid prompts
    distance:       float   # Euclidean distance to NIH centroid
    threshold:      float   # calibrated rejection threshold


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
    """Load NIH prototype centroid and threshold from JSON artifact."""
    global _prototype
    if _prototype is None:
        proto_path = settings.clip_prototype
        with open(proto_path) as f:
            data = json.load(f)
        _prototype = {
            "centroid":  np.array(data["centroid"], dtype=np.float32),
            "threshold": float(data["threshold"]),
        }
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


def _layer2_prototype_check(image_emb: np.ndarray) -> tuple[bool, float, float]:
    """Layer 2: Euclidean distance to NIH ChestX-ray14 prototype centroid.

    Returns (passed, distance, threshold).
    Uses Euclidean distance per Snell et al. (2017) recommendation —
    shown to outperform cosine for prototype-based classification.
    Threshold τ = mean_dist + 1.5 × std_dist, calibrated on 500 NIH samples.
    """
    prototype = _load_prototype()
    centroid = prototype["centroid"]
    threshold = prototype["threshold"]

    distance = float(np.linalg.norm(image_emb - centroid))
    passed = distance <= threshold

    return passed, distance, threshold


def validate(image: Image.Image) -> ValidationResult:
    """Run two-layer CLIP validation on a PIL image.

    Args:
        image: Input PIL image from user upload.

    Returns:
        ValidationResult with layer-wise decisions and scores.
    """
    image_emb = _encode_image(image)

    layer1_passed, valid_score, invalid_score = _layer1_prompt_check(image_emb)
    layer2_passed, distance, threshold = _layer2_prototype_check(image_emb)

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
            f"(distance={distance:.3f}, threshold={threshold:.3f})"
        )
    else:
        code = "valid"
        reason = "Valid chest X-ray"

    return ValidationResult(
        is_valid = is_valid,
        layer1_passed = layer1_passed,
        layer2_passed = layer2_passed,
        code = code,
        reason = reason,
        valid_score = round(valid_score, 4),
        invalid_score = round(invalid_score, 4),
        distance = round(distance, 4),
        threshold = round(threshold, 4),
    )