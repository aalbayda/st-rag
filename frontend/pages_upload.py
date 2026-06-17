from __future__ import annotations

import time

import httpx
import streamlit as st

from app.config import get_settings
from frontend.api import delete_file, get_files, post_files

def _fetch_files_with_retry(attempts: int = 12, delay: float = 0.6) -> tuple[list, bool]:
    for attempt in range(attempts):
        try:
            return get_files(), True
        except httpx.RequestError:
            if attempt < attempts - 1:
                time.sleep(delay)
        except Exception:
            return [], False
    return [], False


def _stage_icon(stage: str | None) -> str:
    return {
        "queued": "⏳",
        "parsing": "📄",
        "embedding": "🔢",
        "indexed": "✅",
        "failed": "❌",
    }.get(stage or "", "❓")


def _render_file_list(rows: list[dict]) -> None:
    st.subheader("Indexed documents")
    if not rows:
        st.markdown("*No documents yet*")
        return

    list_height = min(60 + 64 * len(rows), 300)
    with st.container(height=list_height, border=False):
        for row in rows:
            stage = row.get("stage") or ""
            name = row.get("name") or "(unknown)"
            error = row.get("error")
            file_id = row.get("id")

            icon = _stage_icon(stage)
            label = f"{icon} **{name}**"
            detail = f"  |  stage: `{stage}`" if stage else ""

            st.markdown(label + detail)
            if error:
                st.caption(f":red[{error}]")

            deletable = stage not in ("queued", "parsing", "embedding")
            if deletable and st.button(
                "Delete document",
                key=f"del-file-{file_id}",
                help="Remove this document",
                use_container_width=True,
            ):
                try:
                    delete_file(file_id)
                except Exception as exc:
                    st.error(f"Could not delete {name}: {exc}")
                else:
                    try:
                        st.session_state.files = get_files()
                    except Exception:
                        st.session_state.files = []
                    st.rerun()


def render_documents_section() -> None:
    s = get_settings()

    if "files" not in st.session_state:
        with st.spinner("Loading documents..."):
            files, reachable = _fetch_files_with_retry
        if reachable:
            st.session_state.files = files
        else:
            st.warning(
                "Couldn’t reach the backend yet. It may still be starting. "
                "This clears on its own; reload if it persists."
            )

    uploaded = st.file_uploader(
        "Select PDF or DOCX files",
        accept_multiple_files=True,
        type=["pdf", "docx"],
        help=f"Up to {s.max_files} files, {s.max_file_bytes // (1024 * 1024)} MB each.",
    )

    too_many = bool(uploaded) and len(uploaded) > s.max_files
    too_big = bool(uploaded) and any(f.size > s.max_file_bytes for f in uploaded)

    if too_many:
        st.warning(
            f"Too many files selected ({len(uploaded)}/{s.max_files}). "
            "Please remove some before uploading."
        )
    if too_big:
        over = [
            f.name
            for f in uploaded
            if f.size > s.max_file_bytes
        ]
        limit_mb = s.max_file_bytes // (1024 * 1024)
        st.warning(
            f"File(s) exceed the {limit_mb} MB limit: {', '.join(over)}. "
            "Please remove them before uploading."
        )

    upload_disabled = not uploaded or too_many or too_big
    if st.button("Upload", disabled=upload_disabled, type="primary"):
        with st.spinner("Uploading..."):
            try:
                result = post_files(uploaded)
            except Exception as exc:
                st.error(f"Upload failed: {exc}")
            else:
                results = result.get("results", [])
                accepted = [r for r in results if r.get("accepted")]
                rejected = [r for r in results if not r.get("accepted")]

                if accepted:
                    names = ", ".join(r.get("name", "") for r in accepted)
                    st.success(f"Accepted {len(accepted)} file(s): {names}")
                if rejected:
                    for r in rejected:
                        reason = r.get("reason") or "rejected by server"
                        st.warning(f"{r.get('name', '?')}: {reason}")

                try:
                    st.session_state.files = get_files()
                except Exception as exc:
                    st.error(f"Could not refresh file list: {exc}")

    st.divider

    _render_file_list(st.session_state.get("files", []))
