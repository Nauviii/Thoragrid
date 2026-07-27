"""Canonical disagreement reasons, shared by the feedback UI and the SQL agent's schema prompt.

Stored as stable codes rather than prose so the data stays aggregatable: the analytics agent
filters on these exact values, and renaming a display label must never silently break a query.
Each code names the pipeline stage most likely at fault — a missed finding points at the CNN
thresholds, a localisation error at GradCAM, a severity error at retrieval or the LLM — which
is what makes the feedback actionable rather than just a score.
"""

OTHER_REASON = "other"

# Insertion order is the order shown in the UI.
FEEDBACK_REASONS: dict[str, str] = {
    "missed_finding": "Missed a finding",
    "false_positive": "Reported something not present",
    "wrong_location": "Localisation off",
    "wrong_severity": "Severity or emphasis wrong",
    OTHER_REASON: "Other",
}