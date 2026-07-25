"""MedAssist frontend entry point: auth gate, custom navigation, page routing.

Run with: streamlit run app/main.py
"""
import sys
from pathlib import Path
import streamlit as st
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import api_client
from app.theme import inject_css

st.set_page_config(
    page_title="MedAssist", page_icon="◫", layout="wide",
    initial_sidebar_state="expanded",
)


def _login_screen() -> None:
    """Render the login form; sets session token on success."""
    _, center, _ = st.columns([1, 1.1, 1])
    with center:
        st.markdown('<div style="height:12vh"></div>', unsafe_allow_html=True)
        st.markdown("# MedAssist")
        st.markdown(
            '<span class="ma-caption">Chest X-ray decision support for radiologists '
            "and physicians</span>",
            unsafe_allow_html=True,
        )
        st.markdown('<hr class="ma-divider">', unsafe_allow_html=True)

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Sign in", use_container_width=True):
            try:
                st.session_state.token = api_client.login(username, password)
                st.session_state.username = username
                st.rerun()
            except api_client.ApiError:
                st.error("Incorrect username or password.")


def _header() -> None:
    """Render the app header with the signed-in user."""
    st.markdown(
        f'<div class="ma-header">'
        f'<span class="ma-header-title">MedAssist</span>'
        f'<span class="ma-header-meta">{st.session_state.username}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    """Route to the login screen or the authenticated app."""
    inject_css()

    if "token" not in st.session_state:
        _login_screen()
        return

    _header()

    from app.pages import chat, history, sql_agent

    page = st.navigation([
        st.Page(chat.render, title="Chat", url_path="chat", default=True),
        st.Page(history.render, title="History", url_path="history"),
        st.Page(sql_agent.render, title="Analytics", url_path="analytics"),
    ], position="top")
    page.run()


main()