
from __future__ import annotations

import json
import re

import markdown as md
import streamlit as st
from st_click_detector import click_detector

from app.contracts import DocxLocator, PdfLocator
from frontend.api import (
    get_session_messages,
    get_sessions,
    post_chat,
    post_session,
)
from frontend.pages_upload import render_documents_section


_ABSTENTION_MSG = (
    "No answer could be grounded in your documents. "
    "try rephrasing, or check the relevant files are indexed."
)


def _locator_label(loc: dict) -> str:
    if loc["kind"] == "pdf":
        return PdfLocator(**loc).label()
    return DocxLocator(**loc).label()


@st.dialog("Source")
def _show_source(cite: dict) -> None:
    st.caption(f"{cite['file_name']} · {cite['locator_label']}")
    st.write(cite["chunk_text"])


def _cite_display(c: dict) -> dict:
    return {
        "file_name": c["file_name"],
        "locator_label": _locator_label(c["locator"]),
        "chunk_text": c["chunk_text"],
    }


def _answer_html(text: str, cite_nums: set) -> str:
    html = md.markdown(text)

    def repl(match: re.Match) -> str:
        n = match.group(1)
        if n in cite_nums:
            return (
                f"<a href='#' id='{n}' "
                f"style='text-decoration:none;font-weight:600;'>[{n}]</a>"
            )
        return match.group(0)

    return re.sub(r"\[(\d+)\]", repl, html)


def _render_message(m_idx: int, msg: dict) -> None:
    if msg["role"] == "user":
        st.write(msg["text"])
        return

    ans: dict = msg["answer"]

    if ans.get("abstained") or not ans.get("answer", "").strip():
        st.info(_ABSTENTION_MSG)
        return

    cites: list = ans.get("citations", [])
    cite_by_num = {c["id"].strip("[]"): c for c in cites}
    html = _answer_html(ans["answer"], set(cite_by_num))

    clicked = click_detector(html, key=f"cite-detector-{m_idx}")

    seen_key = f"_cite_seen_{m_idx}"
    if clicked and clicked != st.session_state.get(seen_key):
        st.session_state[seen_key] = clicked
        c = cite_by_num.get(clicked)
        if c is not None:
            _show_source(_cite_display(c))

    if cites:
        with st.expander(f"Sources ({len(cites)})", expanded=False):
            for i, c in enumerate(cites):
                label = f"{c['id']} {c['file_name']} · {_locator_label(c['locator'])}"
                if st.button(label, key=f"src-{m_idx}-{i}", use_container_width=True):
                    _show_source(_cite_display(c))


def _deserialize_messages(raw_msgs: list) -> list:
    out = []
    for m in raw_msgs:
        if m["role"] == "user":
            out.append({"role": "user", "text": m["content"]})
        else:
            citations = json.loads(m.get("citations") or "[]")
            out.append({
                "role": "assistant",
                "answer": {
                    "answer": m["content"],
                    "reasoning": m.get("reasoning") or "",
                    "citations": citations,
                    "abstained": False,
                },
            })
    return out


def render_chat() -> None:
    st.title("Ask Bob")

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = ""

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.markdown(
            """
            <style>
            section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
                overflow-y: hidden !important;
            }
            section[data-testid="stSidebar"] [class*="st-key-del-file-"] button {
                background-color: #d64545;
                border-color: #d64545;
                color: #ffffff;
            }
            section[data-testid="stSidebar"] [class*="st-key-del-file-"] button:hover {
                background-color: #b83b3b;
                border-color: #b83b3b;
                color: #ffffff;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.header("Chat History")
        with st.container(height=200, border=True, key="pane_chat_history"):
            if st.button("+ New Chat", use_container_width=True):
                st.session_state["session_id"] = ""
                st.session_state["messages"] = []
                st.rerun()

            try:
                all_sessions = get_sessions()
            except Exception:
                all_sessions = []

            active_id = st.session_state["session_id"]
            for s in all_sessions:
                label = s["name"] or "New Chat"
                is_active = s["id"] == active_id
                sid = s["id"]

                if st.button(
                    label,
                    key=f"sess-{sid}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state["session_id"] = sid
                    try:
                        raw_msgs = get_session_messages(sid)
                        st.session_state["messages"] = _deserialize_messages(raw_msgs)
                    except Exception:
                        st.session_state["messages"] = []
                    st.rerun()

        st.header("Manage Documents")
        with st.container(border=True, key="pane_manage_docs"):
            render_documents_section()

    has_docs = len(st.session_state.get("files", [])) > 0

    if not has_docs:
        st.info(
            "No documents yet. Use the **Manage Documents** pane in the sidebar to "
            "upload files first. Chat is disabled until your collection has "
            "something to answer from."
        )

    for m_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            _render_message(m_idx, msg)

    chat_placeholder = (
        "Ask about your documents" if has_docs else "Upload documents to start chatting"
    )
    if prompt := st.chat_input(chat_placeholder, disabled=not has_docs):
        st.session_state.messages.append({"role": "user", "text": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                try:
                    if not st.session_state["session_id"]:
                        st.session_state["session_id"] = post_session()["id"]
                    answer = post_chat(prompt, st.session_state["session_id"])
                except Exception:
                    answer = {
                        "answer": "",
                        "reasoning": "The service was unable to process this request.",
                        "citations": [],
                        "abstained": True,
                    }

            assistant_msg = {"role": "assistant", "answer": answer}
            st.session_state.messages.append(assistant_msg)
            _render_message(len(st.session_state.messages) - 1, assistant_msg)
