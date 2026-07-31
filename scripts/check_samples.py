"""Print the raw validator verdict for one or more images.

The interface deliberately shows a reader the outcome and what to do about it, not the number
behind it — which is right for a clinician and unhelpful when the number is the thing you need.
This bypasses the interface and prints what validate() actually returned.

Usage:
    python scripts/check_samples.py path/to/image.png [more.jpg ...]
    python scripts/check_samples.py data/samples/*.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from config.settings import settings
from core.clip.validator import validate

_BAND = {
    "valid": "accepted",
    "quality_warning": "accepted, flagged",
    "outside_training_distribution": "refused - outside trained range",
    "not_a_chest_xray": "refused - not a chest radiograph",
}


def main(paths: list[str]) -> None:
    """Validate each image and print distance, band and the thresholds in force."""
    warn, reject = settings.clip_warn_threshold, settings.clip_reject_threshold
    print(f"thresholds in force:  warn {warn}   reject {reject}\n")
    print(f"{'file':<44}{'distance':>10}{'L1':>5}{'L2':>5}  verdict")
    print("-" * 96)

    for p in paths:
        path = Path(p)
        try:
            with Image.open(path) as image:
                r = validate(image)
        except Exception as exc:
            print(f"{path.name[:42]:<44}{'-':>10}{'-':>5}{'-':>5}  unreadable: {exc}")
            continue

        # Where the distance sits relative to each band edge, which is the part worth quoting.
        print(f"{path.name[:42]:<44}{r.distance:>10.4f}"
              f"{'ok' if r.layer1_passed else 'no':>5}{'ok' if r.layer2_passed else 'no':>5}"
              f"  {_BAND.get(r.code, r.code)}")

    print()
    print("For the slide, quote the distance column. The band follows from it:")
    print(f"  d <= {warn}          accepted")
    print(f"  {warn} < d <= {reject}   accepted, flagged")
    print(f"  d > {reject}          refused")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])