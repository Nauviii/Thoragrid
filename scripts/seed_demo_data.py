"""Seed synthetic reading history so the Analytics page has something to plot.

The database currently holds whatever was uploaded while testing — the same handful of
studies, repeated. That is enough to prove the query path works and not enough to show what
the page is for: a chart of two identical bars says nothing about a reading record.

Everything written here is generated, and the point of the generation is that it stays
internally consistent with the rest of the system rather than being plausible-looking noise:

  * Condition frequencies follow the label distribution of NIH ChestX-ray14 itself, so the
    long tail is the real one. Infiltration leads and Hernia is rare because that is how the
    dataset is shaped, not because those numbers looked good on a bar chart.
  * A condition only appears in `above_threshold` when its score clears that condition's own
    calibrated threshold, read from multilabel_thresholds.json.
  * Dominant zones are drawn from CONDITION_TO_ZONES, and `aligned` is computed by the same
    rule the pipeline uses — so a query for misaligned findings returns rows that really are
    misaligned under the system's own definition.
  * Feedback reasons come from config.feedback_reasons, so the analytics agent can filter on
    them exactly as it would on real feedback.

Rows are tagged in `raw_query` so they can be found and removed again; see --purge.

Usage:
    python scripts/seed_demo_data.py --doctor doctor --findings 400
    python scripts/seed_demo_data.py --purge
"""

import argparse
import json
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.feedback_reasons import FEEDBACK_REASONS, OTHER_REASON
from config.settings import settings
from core.gradcam.region_map import CONDITION_TO_ZONES
from scripts.db_models import (
    CNNResult, GradCAMFinding, Interaction, LLMOutput, Session as UserSession,
    User, UserFeedback,
)

# Label frequencies in NIH ChestX-ray14. Kept as the raw counts rather than as percentages so
# the source stays obvious and the numbers stay checkable against Data_Entry_2017.csv.
NIH_LABEL_COUNTS: dict[str, int] = {
    "Atelectasis": 11559, "Cardiomegaly": 2776, "Consolidation": 4667, "Edema": 2303,
    "Effusion": 13317, "Emphysema": 2516, "Fibrosis": 1686, "Hernia": 227,
    "Infiltration": 19894, "Mass": 5782, "Nodule": 6331, "Pleural_Thickening": 3385,
    "Pneumonia": 1431, "Pneumothorax": 5302,
}

ALL_ZONES = ["RUZ", "LUZ", "RMZ", "LMZ", "CAR", "RLZ", "LLZ"]

# Every seeded row carries this in raw_query so --purge can find them without guessing.
SEED_TAG = "[seed:demo]"

_FALLBACK_THRESHOLD = 0.5
_MISALIGNED_RATE = 0.18   # roughly the share the pipeline flags as atypical in practice
_FEEDBACK_RATE = 0.35     # share of studies a reader actually comments on
_DISAGREE_RATE = 0.30     # of those, the share that disagree


def _load_thresholds() -> dict[str, float]:
    """Read per-class thresholds, falling back to a flat value if the artifact is absent."""
    path = Path(settings.thresholds_path)
    if not path.exists():
        print(f"warning: {path} not found, using a flat {_FALLBACK_THRESHOLD} threshold")
        return {c: _FALLBACK_THRESHOLD for c in NIH_LABEL_COUNTS}
    data = json.loads(path.read_text())
    thresholds = data.get("thresholds", data)
    return {c: float(thresholds.get(c, _FALLBACK_THRESHOLD)) for c in NIH_LABEL_COUNTS}


def _allocate(total_findings: int, floor: int = 2) -> dict[str, int]:
    """Split a finding budget across conditions in NIH proportion, with a floor per condition.

    The floor exists so the rarest conditions still appear at all. Hernia is 0.3% of NIH
    labels; at any demo-sized total it would round to zero and the chart would quietly show
    thirteen conditions while claiming fourteen.
    """
    grand_total = sum(NIH_LABEL_COUNTS.values())
    allocated = {
        condition: max(floor, round(total_findings * count / grand_total))
        for condition, count in NIH_LABEL_COUNTS.items()
    }
    return allocated


def _scores_for(present: list[str], thresholds: dict[str, float],
                rng: random.Random) -> dict[str, float]:
    """Build a full 14-condition score vector consistent with which conditions are present."""
    scores = {}
    for condition, threshold in thresholds.items():
        if condition in present:
            # Comfortably over the line, but not implausibly certain.
            scores[condition] = round(min(0.99, rng.uniform(threshold + 0.05, threshold + 0.42)), 4)
        else:
            scores[condition] = round(max(0.01, rng.uniform(0.02, threshold - 0.03)), 4)
    return scores


def _zones_for(condition: str, aligned: bool, rng: random.Random) -> list[str]:
    """Pick dominant zones that make `aligned` true or false under the pipeline's own rule."""
    expected = CONDITION_TO_ZONES.get(condition, [])
    if aligned and expected:
        return rng.sample(expected, k=min(len(expected), rng.choice([1, 1, 2])))
    off = [z for z in ALL_ZONES if z not in expected] or ALL_ZONES
    return rng.sample(off, k=min(len(off), rng.choice([1, 2])))


def _purge(session) -> None:
    """Delete every seeded row, leaving real interactions untouched."""
    ids = [
        row[0] for row in
        session.query(Interaction.id).filter(Interaction.raw_query.like(f"%{SEED_TAG}%")).all()
    ]
    if not ids:
        print("nothing to purge")
        return
    for model in (UserFeedback, GradCAMFinding, LLMOutput, CNNResult):
        session.query(model).filter(model.interaction_id.in_(ids)).delete(synchronize_session=False)
    session.query(Interaction).filter(Interaction.id.in_(ids)).delete(synchronize_session=False)
    session.commit()
    print(f"purged {len(ids)} seeded interactions and their child rows")


def seed(username: str, total_findings: int, days: int, seed_value: int) -> None:
    """Generate studies, findings and feedback for one doctor account."""
    engine = create_engine(settings.database_url)
    db = sessionmaker(bind=engine)()

    user = db.query(User).filter_by(username=username).first()
    if user is None:
        raise SystemExit(f"No user named {username!r}. Run scripts/seed_users.py first.")

    rng = random.Random(seed_value)
    thresholds = _load_thresholds()
    budget = _allocate(total_findings)

    # A pool of findings to distribute over studies, shuffled so multi-finding studies mix
    # conditions rather than pairing the same two every time.
    pool = [c for condition, n in budget.items() for c in [condition] * n]
    rng.shuffle(pool)

    user_session = UserSession(user_id=user.id, role=user.role,
                               started_at=datetime.now(timezone.utc) - timedelta(days=days))
    db.add(user_session)
    db.flush()

    studies = 0
    reason_codes = list(FEEDBACK_REASONS)
    while pool:
        # NIH studies carry one label far more often than several; this mirrors that shape.
        take = min(rng.choices([1, 2, 3], weights=[62, 28, 10])[0], len(pool))
        present = list(dict.fromkeys(pool[:take]))
        pool = pool[take:]

        when = datetime.now(timezone.utc) - timedelta(
            days=rng.uniform(0, days), hours=rng.uniform(0, 24)
        )
        # The id is generated here rather than left to the column default: conversation_id is
        # a self-referential foreign key, so it has to hold a real interaction id at INSERT
        # time — there is no window after the flush in which to patch it up.
        interaction_id = str(uuid.uuid4())
        interaction = Interaction(
            id=interaction_id,
            session_id=user_session.id,
            conversation_id=interaction_id,   # a study opens its own conversation
            interaction_type="image",
            raw_query=SEED_TAG,
            image_hash=f"seed{rng.getrandbits(48):012x}",
            xray_storage_url=None,
            timestamp=when,
            latency_ms=rng.randint(7200, 15800),
        )
        db.add(interaction)
        db.flush()

        db.add(CNNResult(
            interaction_id=interaction.id,
            all_scores=_scores_for(present, thresholds, rng),
            above_threshold=present,
            low_confidence_flag=False,
        ))

        for condition in present:
            aligned = rng.random() > _MISALIGNED_RATE
            zones = _zones_for(condition, aligned, rng)
            db.add(GradCAMFinding(
                interaction_id=interaction.id,
                condition=condition,
                heatmap_storage_url="",
                dominant_zones=zones,
                aligned=any(z in CONDITION_TO_ZONES.get(condition, []) for z in zones),
                zone_stats={z: round(rng.uniform(0.02, 0.4), 3) for z in ALL_ZONES},
            ))

        db.add(LLMOutput(
            interaction_id=interaction.id,
            call2_output={
                "conditions": [],
                "clinical_summary": f"{SEED_TAG} Generated record for demonstration.",
                "cross_specialty_notes": None,
            },
        ))

        if rng.random() < _FEEDBACK_RATE:
            agreed = rng.random() > _DISAGREE_RATE
            comment = None
            if not agreed:
                code = rng.choices(reason_codes, weights=[30, 22, 26, 16, 6])[0]
                comment = f"{code}: presentation atypical" if code == OTHER_REASON else code
            db.add(UserFeedback(interaction_id=interaction.id, is_correct=agreed,
                                comment=comment, submitted_at=when + timedelta(minutes=3)))

        studies += 1

    db.commit()

    print(f"seeded {studies} studies for {username}, {sum(budget.values())} findings")
    print()
    print(f"{'condition':<22}{'findings':>10}")
    for condition, n in sorted(budget.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{condition:<22}{n:>10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doctor", default="doctor", help="username to attribute records to")
    parser.add_argument("--findings", type=int, default=400, help="total findings to generate")
    parser.add_argument("--days", type=int, default=45, help="spread records over this many days")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--purge", action="store_true", help="remove seeded rows and exit")
    args = parser.parse_args()

    if args.purge:
        engine = create_engine(settings.database_url)
        _purge(sessionmaker(bind=engine)())
    else:
        seed(args.doctor, args.findings, args.days, args.seed)