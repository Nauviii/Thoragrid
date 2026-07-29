"""Thoragrid entry point: the app shell, sign-in, session resume, and page routing.

The shell is a dark rail holding the mark, navigation, whatever the current page needs, and
the signed-in reader. Streamlit's own navigation is suppressed (position="hidden") so the
rail can be built from st.page_link and styled as one piece rather than fought with.

Run with: streamlit run app/main.py
"""
import sys
from pathlib import Path
import streamlit as st
import extra_streamlit_components as stx
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import api_client
from app.theme import inject_css, brand_lockup, brand_mark, zone_watermark

st.set_page_config(
    page_title="Thoragrid", page_icon="◫", layout="wide",
    initial_sidebar_state="expanded",
)

# Only the opaque key lives in the browser; the JWT stays server-side in Redis.
_SESSION_COOKIE = "medassist_session"

# Exactly one instance per script run, at module scope. It cannot be cached, because the
# constructor issues a keyed widget command and Streamlit forbids those inside cached
# functions. It cannot be stashed in session_state either: a widget that isn't re-rendered
# on every run goes stale on the frontend. And it must not be constructed twice in one run,
# or the two calls collide on the same widget key.
_cookies = stx.CookieManager(key="medassist_cookie_manager")


def _forget_cookie(widget_key: str) -> None:
    """Delete the session cookie, tolerating it already being absent."""
    try:
        _cookies.delete(_SESSION_COOKIE, key=widget_key)
    except KeyError:
        # CookieManager.delete() also drops the name from its local dict, which raises if
        # the cookie was never loaded in this run. The browser-side delete still went out.
        pass


def _restore_session() -> bool:
    """Resume a login from the session cookie; return whether a session is now active."""
    if "token" in st.session_state:
        return True

    session_key = _cookies.get(_SESSION_COOKIE)
    if not session_key:
        return False

    try:
        session = api_client.resume_session(session_key)
    except api_client.ApiError:
        _forget_cookie("delete_stale_session")
        return False

    _store_session(session)
    return True


def _store_session(session: dict) -> None:
    """Put an authenticated session into session state and the browser cookie."""
    st.session_state.token = session["token"]
    st.session_state.role = session["role"]
    st.session_state.username = session["username"]
    st.session_state.session_key = session["session_key"]
    # The cookie's own expiry is deliberately loose; Redis holds the authoritative TTL and
    # a cookie outliving it simply fails to resume and is cleaned up above.
    _cookies.set(_SESSION_COOKIE, session["session_key"], key="set_session")


def _sign_in_screen() -> None:
    """Render the sign-in screen: the mark, one line of orientation, two fields."""
    st.markdown(
        f'<div style="height:9vh"></div>'
        f'<div style="display:flex;justify-content:center;margin-bottom:1.6rem">'
        f"{brand_mark(56)}</div>",
        unsafe_allow_html=True,
    )
    _, center, _ = st.columns([1, 1.15, 1])
    with center:
        st.markdown(
            '<div style="text-align:center">'
            '<div style="font-family:var(--serif);font-size:2.7rem;line-height:1.05;'
            'letter-spacing:-0.015em">Thoragrid</div>'
            '<div class="ma-caption" style="margin-top:0.5rem">'
            "Chest radiograph decision support. Fourteen findings, localised and explained, "
            "for the specialist reading the study.</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown('<hr class="ma-divider">', unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="doctor")
        password = st.text_input("Password", type="password", placeholder="••••••••")

        if st.button("Sign in", width="stretch", type="primary"):
            try:
                _store_session(api_client.login(username, password))
                st.rerun()
            except api_client.ApiError:
                st.error("That username and password don't match an account.")

        st.markdown(
            '<div class="ma-caption" style="text-align:center;margin-top:1.4rem">'
            "Access is limited to registered clinicians and administrators.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="display:flex;justify-content:center;opacity:0.5;margin-top:2rem">'
        f"{zone_watermark(120)}</div>",
        unsafe_allow_html=True,
    )


def _sign_out() -> None:
    """Close any open case, revoke the browser session server-side, and return to sign-in."""
    conversation_id = st.session_state.get("conversation_id")
    if conversation_id:
        try:
            api_client.close_conversation(st.session_state.token, conversation_id)
        except api_client.ApiError:
            pass  # best-effort; the Postgres transcript is preserved regardless

    session_key = st.session_state.get("session_key")
    if session_key:
        api_client.end_browser_session(session_key)
    _forget_cookie("delete_session")

    st.session_state.clear()
    st.rerun()


def _rail_top(pages: dict) -> None:
    """Render the mark and the navigation rail."""
    st.markdown(brand_lockup(), unsafe_allow_html=True)
    for page in pages.values():
        st.page_link(page, width="stretch")


def _rail_bottom() -> None:
    """Render the signed-in reader and the sign-out control at the foot of the rail."""
    name = st.session_state.username
    st.markdown('<hr class="ma-rail-rule">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="ma-user"><div class="ma-user-dot">{name[:2]}</div>'
        f'<div><div class="ma-user-name">{name}</div>'
        f'<div class="ma-user-role">{st.session_state.role}</div></div></div>',
        unsafe_allow_html=True,
    )
    if st.button("Sign out", width="stretch"):
        _sign_out()


def main() -> None:
    """Route to sign-in or the authenticated app."""
    inject_css()

    if not _restore_session():
        _sign_in_screen()
        return

    from app.views import chat, history, sql_agent

    pages = {
        "chat": st.Page(chat.render, title="Chat", url_path="chat",
                        icon=":material/stethoscope:", default=True),
        "history": st.Page(history.render, title="History", url_path="history",
                           icon=":material/history:"),
        "analytics": st.Page(sql_agent.render, title="Analytics", url_path="analytics",
                             icon=":material/monitoring:"),
    }
    current = st.navigation(list(pages.values()), position="hidden")

    with st.sidebar:
        _rail_top(pages)

    # The page runs between the two rail halves so a page can contribute its own controls
    # (the Chat case panel) into the middle of the rail.
    current.run()

    with st.sidebar:
        _rail_bottom()


main()