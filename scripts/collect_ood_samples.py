"""Sample images out of a downloaded dataset into one stratum of the evaluation set.

Public radiograph datasets arrive in whatever folder layout their author chose, usually split
by train/val and by class, and usually far larger than an evaluation set needs. This walks a
source tree, takes a reproducible random sample, and writes it flat into one stratum.

Sampling matters more than it looks. Taking the first N files in directory order tends to pull
a single patient, a single scanner, or a single class, and a validator that clears that sample
has cleared almost nothing. A seeded random draw across the whole tree is the cheapest defence
against measuring one narrow slice and calling it a distribution.

Usage:
    python scripts/collect_ood_samples.py \
        --source ~/Downloads/fracture-multi-region-x-ray-data \
        --stratum other_radiograph --n 80
"""

import argparse
import random
import shutil
from pathlib import Path

_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
_MIN_PIXELS = 128  # below this an image is a thumbnail or an icon, not a study


def _candidates(source: Path) -> list[Path]:
    """Walk the source tree for usable images."""
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in _SUFFIXES and p.is_file())


def collect(source: Path, out_root: Path, stratum: str, n: int, seed: int) -> None:
    """Copy a seeded random sample of images into data/ood_eval/negative/<stratum>/."""
    if not source.is_dir():
        raise SystemExit(f"Source not found: {source}")

    found = _candidates(source)
    if not found:
        raise SystemExit(f"No images under {source}")

    destination = out_root / "negative" / stratum
    destination.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    picks = rng.sample(found, min(n, len(found)))

    from PIL import Image

    written, skipped = 0, 0
    for path in picks:
        try:
            with Image.open(path) as image:
                if min(image.size) < _MIN_PIXELS:
                    skipped += 1
                    continue
        except Exception:
            skipped += 1
            continue
        # Keep a couple of parent directory names: dataset layouts encode the body part there,
        # and it makes a surprising rejection traceable to its source without re-deriving it.
        tag = "_".join(part for part in path.parts[-3:-1] if part)[:40]
        shutil.copy2(path, destination / f"{tag}__{path.name}")
        written += 1

    print(f"Found {len(found)} images under {source}")
    print(f"Wrote {written} to {destination}" + (f" ({skipped} skipped as too small)" if skipped else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="downloaded dataset root")
    parser.add_argument("--stratum", required=True,
                        help="other_radiograph | other_modality | non_medical")
    parser.add_argument("--out-root", type=Path, default=Path("data/ood_eval"))
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    collect(args.source, args.out_root, args.stratum, args.n, args.seed)