"""Measure an input validator against a labelled set of positives and negatives.

This is the instrument every later decision rests on. The shipped threshold was calibrated on
positives alone — mean + 1.5 sigma over 500 NIH samples — which fixes how many genuine chest
X-rays are rejected but says nothing about how many non-chest images get through. That number
can only be measured, and until it is, no backbone change can be called an improvement.

The validator is resolved by module path rather than imported directly, so the same harness
measures the CLIP version today and a DINOv2 version later without being rewritten. Any module
exposing `validate(PIL.Image) -> result` works, where the result carries `is_valid` and `code`;
`distance` is used for the threshold-free metrics when the validator exposes one.

Attribution is read from `code` rather than from layer flags, because the codes survive a
change of backbone and the layer numbering does not.

Expects:
    data/ood_eval/positive/                  chest X-rays held out of calibration
    data/ood_eval/negative/<stratum>/        one directory per failure mode

Usage:
    python scripts/eval_validator.py
    python scripts/eval_validator.py --validator core.ood.dinov2_validator --csv runs/dinov2.csv
"""

import argparse
import csv
import importlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def _images(directory: Path) -> list[Path]:
    """Return every image directly inside a directory, sorted for reproducibility."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in _SUFFIXES)


def _auroc(positive_scores: list[float], negative_scores: list[float]) -> float:
    """Return AUROC via the Mann-Whitney U relation, ties counted as half.

    Scores are distances, so a negative should score higher; 1.0 means every negative sits
    further from the reference distribution than every positive.
    """
    if not positive_scores or not negative_scores:
        return float("nan")
    wins = sum(
        1.0 if n > p else 0.5 if n == p else 0.0
        for n in negative_scores for p in positive_scores
    )
    return wins / (len(positive_scores) * len(negative_scores))


def _fpr_at_tpr(positive_scores: list[float], negative_scores: list[float],
                target_tpr: float = 0.95) -> tuple[float, float]:
    """Return (fpr, cut-off) at the distance threshold that accepts `target_tpr` of positives."""
    if not positive_scores or not negative_scores:
        return float("nan"), float("nan")
    ordered = sorted(positive_scores)
    index = max(min(int(round(target_tpr * len(ordered))) - 1, len(ordered) - 1), 0)
    cutoff = ordered[index]
    accepted = sum(1 for s in negative_scores if s <= cutoff)
    return accepted / len(negative_scores), cutoff


def _load_validator(module_path: str):
    """Resolve the validate() callable from a module path."""
    module = importlib.import_module(module_path)
    if not hasattr(module, "validate"):
        raise SystemExit(f"{module_path} exposes no validate() function")
    return module.validate


def _run(validate, paths: list[Path]) -> list:
    """Validate every image, reporting progress so a long run doesn't look hung."""
    results = []
    for i, path in enumerate(paths, 1):
        with Image.open(path) as image:
            results.append(validate(image))
        if i % 25 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)}", end="\r", flush=True)
    print(" " * 24, end="\r")
    return results


def _score_of(result) -> float | None:
    """Return the continuous out-of-distribution score, if this validator exposes one."""
    return getattr(result, "distance", None)


def evaluate(root: Path, module_path: str, csv_path: Path | None) -> None:
    """Run the validator over the labelled set and print the report."""
    validate = _load_validator(module_path)

    positives = _images(root / "positive")
    if not positives:
        raise SystemExit(f"No positives found in {root / 'positive'}")

    negative_root = root / "negative"
    strata = {}
    if negative_root.is_dir():
        for directory in sorted(p for p in negative_root.iterdir() if p.is_dir()):
            found = _images(directory)
            if found:
                strata[directory.name] = found
    if not strata:
        raise SystemExit(f"No negative strata found under {negative_root}")

    print(f"Validator : {module_path}")
    print(f"Positives : {len(positives)}")
    positive_results = _run(validate, positives)
    negative_results = {}
    for name, paths in strata.items():
        print(f"{name:<22}: {len(paths)}")
        negative_results[name] = _run(validate, paths)

    accepted_pos = sum(1 for r in positive_results if r.is_valid)
    tpr = accepted_pos / len(positives)

    print()
    print(f"Positives accepted (TPR) : {accepted_pos}/{len(positives)} ({tpr:.1%})")

    pos_scores = [s for s in map(_score_of, positive_results) if s is not None]
    neg_scores = [s for rs in negative_results.values() for s in map(_score_of, rs)
                  if s is not None]
    if pos_scores and neg_scores:
        fpr, cutoff = _fpr_at_tpr(pos_scores, neg_scores)
        print(f"AUROC (distance)         : {_auroc(pos_scores, neg_scores):.3f}")
        print(f"FPR@95TPR                : {fpr:.1%}  (cut-off {cutoff:.4f})")
    else:
        print("AUROC / FPR@95TPR        : not available (validator exposes no distance)")

    # Attribution by rejection code, so the report reads the same whatever the backbone is.
    codes = sorted({r.code for rs in negative_results.values() for r in rs} - {"valid"})
    header = f"{'stratum':<22}{'n':>5}{'let through':>13}" + "".join(f"{c[:18]:>20}" for c in codes)
    print()
    print(header)
    print("-" * len(header))

    totals = defaultdict(int)
    for name, results in negative_results.items():
        counts = Counter(r.code for r in results)
        through = counts["valid"]
        totals["n"] += len(results)
        totals["through"] += through
        row = f"{name:<22}{len(results):>5}{through:>13}"
        for code in codes:
            totals[code] += counts[code]
            row += f"{counts[code]:>20}"
        print(row)

    print("-" * len(header))
    summary = f"{'all negatives':<22}{totals['n']:>5}{totals['through']:>13}"
    for code in codes:
        summary += f"{totals[code]:>20}"
    print(summary)
    print()
    print(f"False positives let through : {totals['through']}/{totals['n']} "
          f"({totals['through'] / totals['n']:.1%})")

    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["stratum", "path", "is_valid", "code", "distance"])
            for path, result in zip(positives, positive_results):
                writer.writerow(["positive", path.name, result.is_valid, result.code,
                                 _score_of(result)])
            for name, results in negative_results.items():
                for path, result in zip(strata[name], results):
                    writer.writerow([name, path.name, result.is_valid, result.code,
                                     _score_of(result)])
        print(f"Per-image results written to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/ood_eval"))
    parser.add_argument("--validator", default="core.clip.validator",
                        help="module path exposing validate(image)")
    parser.add_argument("--csv", type=Path, default=None, help="write per-image results here")
    args = parser.parse_args()
    evaluate(args.root, args.validator, args.csv)