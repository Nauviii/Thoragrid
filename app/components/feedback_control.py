"""Agree/disagree feedback controls for a single interaction, shared by Chat and History."""

import streamlit as st

from app import api_client
from config.feedback_reasons import FEEDBACK_REASONS, OTHER_REASON

# Verdicts submitted during this run, so the control updates without refetching history.
_STATE_KEY = "feedback_given"


def _picker_key(interaction_id: str) -> str:
    """Session-state key marking that a disagreement is awaiting its reason."""
    return f"fb_picker_{interaction_id}"


def _note_key(interaction_id: str) -> str:
    """Session-state key marking that an 'other' reason is awaiting its free text."""
    return f"fb_note_{interaction_id}"


def _clear_pending(interaction_id: str) -> None:
    """Drop any in-progress disagreement state for this interaction."""
    st.session_state.pop(_picker_key(interaction_id), None)
    st.session_state.pop(_note_key(interaction_id), None)


def _submit(interaction_id: str, is_correct: bool, comment: str | None) -> None:
    """Send feedback, treating a 409 as already recorded rather than as an error."""
    try:
        api_client.submit_feedback(st.session_state.token, interaction_id, is_correct, comment)
    except api_client.ApiError as exc:
        if exc.status_code != 409:
            st.warning(str(exc))
            return

    st.session_state[_STATE_KEY][interaction_id] = is_correct
    _clear_pending(interaction_id)
    st.rerun()


def _render_note_form(interaction_id: str) -> None:
    """Collect free text for the one reason that cannot be pre-categorised."""
    note = st.text_area(
        "What was wrong?",
        key=f"fb_note_text_{interaction_id}",
        placeholder="Anything the categories above don't cover.",
    )
    col_submit, col_cancel, _ = st.columns([1, 1, 4])
    with col_submit:
        if st.button("Submit", key=f"fb_note_submit_{interaction_id}", use_container_width=True):
            # Keep the code as the prefix so 'other' stays filterable even with a note attached.
            comment = f"{OTHER_REASON}: {note}" if note.strip() else OTHER_REASON
            _submit(interaction_id, False, comment)
    with col_cancel:
        if st.button("Cancel", key=f"fb_note_cancel_{interaction_id}", use_container_width=True):
            _clear_pending(interaction_id)
            st.rerun()


def _render_reason_picker(interaction_id: str) -> None:
    """Offer one-click reasons for a disagreement, two per row."""
    st.markdown('<span class="ma-caption">What was wrong?</span>', unsafe_allow_html=True)

    codes = list(FEEDBACK_REASONS)
    for start in range(0, len(codes), 2):
        columns = st.columns(2)
        for column, code in zip(columns, codes[start:start + 2]):
            with column:
                if st.button(
                    FEEDBACK_REASONS[code],
                    key=f"fb_reason_{code}_{interaction_id}",
                    use_container_width=True,
                ):
                    if code == OTHER_REASON:
                        st.session_state[_note_key(interaction_id)] = True
                        st.rerun()
                    else:
                        _submit(interaction_id, False, code)

    if st.button("Cancel", key=f"fb_picker_cancel_{interaction_id}", use_container_width=True):
        _clear_pending(interaction_id)
        st.rerun()


def render_feedback(interaction_id: str, recorded: bool | None = None) -> None:
    """Render feedback controls for one interaction, reflecting any verdict already on record."""
    given = st.session_state.setdefault(_STATE_KEY, {})
    verdict = given.get(interaction_id, recorded)

    if verdict is not None:
        label = "You agreed with this analysis." if verdict else "You flagged this analysis as incorrect."
        st.markdown(f'<span class="ma-caption">{label}</span>', unsafe_allow_html=True)
        return

    if st.session_state.get(_note_key(interaction_id)):
        _render_note_form(interaction_id)
        return

    if st.session_state.get(_picker_key(interaction_id)):
        _render_reason_picker(interaction_id)
        return

    col_agree, col_disagree, _ = st.columns([1, 1, 4])
    with col_agree:
        if st.button("Agree", key=f"fb_agree_{interaction_id}", use_container_width=True):
            _submit(interaction_id, True, None)
    with col_disagree:
        if st.button("Disagree", key=f"fb_disagree_{interaction_id}", use_container_width=True):
            st.session_state[_picker_key(interaction_id)] = True
            st.rerun()