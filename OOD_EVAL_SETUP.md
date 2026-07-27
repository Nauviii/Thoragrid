# Evaluation environment for input validation

Paths follow the repo as it stands: scripts live in `scripts/`, data under `data/ood_eval/`.
`data/` is already in `.gitignore`, so none of these images enter the repository.

## What changed since the first draft

`eval_validator.py` was rewritten. The first version read `layer1_passed` / `layer2_passed`,
which are specific to the CLIP validator — it would have needed rewriting the moment a DINOv2
version appeared. It now resolves the validator by module path and attributes rejections by
the `code` field instead:

```bash
python scripts/eval_validator.py                                  # CLIP, today
python scripts/eval_validator.py --validator core.ood.dinov2_validator   # later, unchanged
```

Any module exposing `validate(PIL.Image) -> result` works, where the result carries `is_valid`
and `code`. If it also exposes `distance`, the threshold-free metrics are reported; if not,
the acceptance table still is. Both codes you already ship — `not_a_chest_xray` and
`outside_training_distribution` — survive the backbone change, which is why they make a better
axis for the report than layer numbers.

`collect_ood_samples.py` is new. `build_degraded_negatives.py` is unchanged.

---

## Layout

```
data/ood_eval/
├── positive/                      chest X-rays held out of calibration
└── negative/
    ├── other_radiograph/
    ├── other_modality/
    ├── non_medical/
    └── degraded_chest/            generated, not downloaded
```

---

## Which datasets

**`other_radiograph` — the stratum that matters most, because it is the case that failed.**

Use `bmadushanirodrigo/fracture-multi-region-x-ray-data` on Kaggle. It covers radiographs
across multiple anatomical regions, needs no access agreement, and downloads through the
Kaggle API you already use. Eighty images is plenty.

Request **MURA** (Stanford AIMI) in parallel — 40,561 images across elbow, finger, forearm,
hand, humerus, shoulder and wrist, from one hospital's PACS, so its acquisition characteristics
are realistic rather than scraped. It is the better set, but it is gated behind a Research Use
Agreement that takes time to come back. Start the request now and use the Kaggle set meanwhile;
if MURA arrives, re-run the harness with it and report that number instead.

**`other_modality`.** Search Kaggle for brain tumour MRI or head CT collections — several are
small, ungated, and already exported as PNG. What matters is not which one, but that the
modality is genuinely different: CT, MRI or ultrasound rather than another radiograph. Forty
images is enough.

Avoid "Medical MNIST" for the number you report. It bundles AbdomenCT, ChestCT, Hand, HeadCT
and BreastMRI in one convenient download, but its images are 64×64. Upscaled to 224 they carry
a blur signature any detector separates trivially, so it will flatter your results. Fine as a
smoke test, never as evidence.

**`non_medical`.** Supply these yourself; no download needed. Thirty photographs, screenshots
and scanned documents. Include at least five phone photos of a monitor displaying a chest
X-ray — that is the realistic failure in a reading room, and it is far harder than a photo of
a dog.

**`positive`.** Chest X-rays held out of the 500 used to build `clip_prototype.json`. Sampling
from those 500 would bias the acceptance rate upward. 150–250 images.

Verify each licence at source before publishing. You said commercial use is not a concern, so
this is documentation, not a blocker.

---

## Running it

```bash
# 1. Held-out positives into data/ood_eval/positive/, then the free stratum:
python scripts/build_degraded_negatives.py \
    --source data/ood_eval/positive \
    --out    data/ood_eval/negative/degraded_chest \
    --per-transform 20

# 2. Sample each downloaded dataset into its stratum:
python scripts/collect_ood_samples.py --source ~/Downloads/fracture-multi-region-x-ray-data \
                                      --stratum other_radiograph --n 80
python scripts/collect_ood_samples.py --source ~/Downloads/head-ct-dataset \
                                      --stratum other_modality --n 40

# 3. Baseline the validator that ships today. Keep the CSV — it is the before number.
python scripts/eval_validator.py --csv runs/clip_baseline.csv
```

Even 30 negatives per stratum gives a usable first reading. Precision improves with more, but
the shape of the answer — which stratum leaks and which code catches it — shows up early.

---

## Reading the report

**AUROC** — threshold-free separation quality. Below ~0.75 the representation is the problem
and no threshold tuning will rescue it. This is the number that decides whether a new backbone
is warranted.

**FPR@95TPR** — the operating point to quote: share of negatives accepted at the cut-off that
admits 95% of genuine chest films.

**Per-stratum, per-code table** — where the leak is and which mechanism caught what.

The prediction worth checking first: `other_radiograph` should show almost nothing under
`not_a_chest_xray`. Every negative prompt currently in `settings.py` names a non-radiograph
category, so a knee film beats them trivially and layer one cannot reject it by construction.
If the table confirms that, the cheapest real improvement is the prompt set and its aggregation
rule — not a new model.

---

## Target

Set the operating point explicitly instead of by sigma multiplier:

> reject ≥ 95% of non-chest radiographs while accepting ≥ 98% of genuine chest X-rays

Whether that is reachable in CLIP space is what the baseline run answers, and it decides
whether the next step is a prompt fix or DINOv2.