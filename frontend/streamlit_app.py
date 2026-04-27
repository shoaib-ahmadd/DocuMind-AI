from __future__ import annotations
# app.py  ── must be the very first two executable lines
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)          # reads .env from the working directory

from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from chatbot.chat_engine_new import ChatConfig, ChatEngine
from core.file_handler import is_vector_store_ready, process_uploaded_files


BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_db"


# ---------------------------------------------------------------------------
# Custom CSS — clean, minimal, professional dark-neutral theme
# ---------------------------------------------------------------------------

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── App background ── */
.stApp {
    background: #0f1117;
}

/* ── Main content column ── */
section.main > div {
    padding-top: 1.5rem;
    padding-bottom: 4rem;
    max-width: 820px;
    margin: 0 auto;
}

/* ── Title area ── */
h1 {
    font-weight: 600;
    font-size: 1.75rem !important;
    letter-spacing: -0.5px;
    color: #f0f0f0 !important;
}

/* ── Chat bubbles ── */
.stChatMessage {
    border-radius: 14px !important;
    padding: 0.1rem 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* User bubble */
[data-testid="stChatMessageContent"]:has(~ [data-testid="stChatMessageAvatarUser"]),
div[data-testid="stChatMessage"][data-message-author-role="user"] [data-testid="stChatMessageContent"] {
    background: #1e2130 !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 0.8rem 1.1rem !important;
    color: #e8e8e8 !important;
    font-size: 0.96rem !important;
    line-height: 1.6 !important;
    border: 1px solid #2a2f45 !important;
}

/* Assistant bubble */
div[data-testid="stChatMessage"][data-message-author-role="assistant"] [data-testid="stChatMessageContent"] {
    background: #161b2e !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 0.9rem 1.2rem !important;
    color: #d4d8e8 !important;
    font-size: 0.96rem !important;
    line-height: 1.65 !important;
    border: 1px solid #1f2640 !important;
}

/* ── Confidence badge ── */
.confidence-badge {
    display: inline-block;
    margin-top: 0.35rem;
    padding: 2px 10px;
    border-radius: 20px;
    background: #1a2035;
    border: 1px solid #252d48;
    color: #6b7db3;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.3px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0c0f1a !important;
    border-right: 1px solid #1a1f35 !important;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stCaption {
    color: #8892b0 !important;
    font-size: 0.85rem !important;
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ccd6f6 !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    margin-bottom: 0.4rem !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0f1420 !important;
    border: 1px dashed #2a3050 !important;
    border-radius: 10px !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: #3d5afe !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 1.4rem !important;
    transition: background 0.2s ease, transform 0.1s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: #536dfe !important;
    transform: translateY(-1px);
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0);
}

/* ── Secondary button ── */
.stButton > button:not([kind="primary"]) {
    background: transparent !important;
    border: 1px solid #2a3050 !important;
    border-radius: 10px !important;
    color: #8892b0 !important;
    font-size: 0.88rem !important;
}

/* ── Text input ── */
.stTextInput > div > div > input {
    background: #131722 !important;
    border: 1px solid #252d48 !important;
    border-radius: 12px !important;
    color: #e0e4f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1rem !important;
    transition: border-color 0.2s ease !important;
}
.stTextInput > div > div > input:focus {
    border-color: #3d5afe !important;
    box-shadow: 0 0 0 2px rgba(61,90,254,0.15) !important;
}
.stTextInput > div > div > input::placeholder {
    color: #3d4a6a !important;
}

/* ── Divider ── */
hr {
    border-color: #1a1f35 !important;
    margin: 1.2rem 0 !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #3d5afe !important;
}

/* ── Info / warning boxes ── */
.stAlert {
    border-radius: 10px !important;
    font-size: 0.88rem !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: #3d5afe !important;
}

/* ── Caption ── */
.stCaption {
    color: #4a5570 !important;
    font-size: 0.82rem !important;
}

/* ── Empty-state card ── */
.empty-state {
    text-align: center;
    padding: 3.5rem 2rem;
    background: #0c101d;
    border: 1px dashed #1e2540;
    border-radius: 16px;
    color: #3d4a6a;
    font-size: 0.94rem;
    line-height: 1.7;
    margin-top: 1.5rem;
}
.empty-state .icon {
    font-size: 2.4rem;
    margin-bottom: 0.8rem;
}
.empty-state strong {
    display: block;
    color: #5a6a96;
    font-size: 1.05rem;
    margin-bottom: 0.4rem;
}

/* ── File list in sidebar ── */
.file-pill {
    background: #0f1420;
    border: 1px solid #1e2540;
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.82rem;
    color: #6b7db3;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
</style>
"""


# ---------------------------------------------------------------------------
# Engine cache
# ---------------------------------------------------------------------------

@st.cache_resource
def get_chat_engine(vector_store_dir: str) -> ChatEngine:
    """Cache engine creation across reruns."""
    return ChatEngine(config=ChatConfig(vector_store_dir=vector_store_dir))


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    for d in (UPLOAD_DIR, VECTOR_STORE_DIR):
        if d.exists() and not d.is_dir():
            d.unlink()
        d.mkdir(parents=True, exist_ok=True)


def _save_uploaded_file(uploaded_file: Any) -> str:
    _ensure_dirs()
    target_path = UPLOAD_DIR / Path(uploaded_file.name).name
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(target_path)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "uploaded_file_paths": [],
        "processed": is_vector_store_ready(str(VECTOR_STORE_DIR)),
        "chat_history": [],
        "last_processed_file_paths": [],
        "force_reprocess": False,
        "question_input": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar() -> None:
    with st.sidebar:
        st.markdown("## 📂 Documents")

        uploaded_files = st.file_uploader(
            "Upload PDF or image files",
            type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            new_paths: List[str] = [_save_uploaded_file(f) for f in uploaded_files]
            st.session_state.uploaded_file_paths = new_paths

            st.markdown("**Queued files**")
            for p in st.session_state.uploaded_file_paths:
                st.markdown(
                    f'<div class="file-pill">📄 {Path(p).name}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No files uploaded yet.")

        st.divider()

        # ── Process button ───────────────────────────────────────────
        if st.session_state.uploaded_file_paths:
            if st.button("⚙️ Process Documents", type="primary", use_container_width=True):
                st.session_state.force_reprocess = True
        else:
            st.button(
                "⚙️ Process Documents",
                type="primary",
                use_container_width=True,
                disabled=True,
            )

        # ── Status indicator ─────────────────────────────────────
        if st.session_state.processed:
            st.success("✅ Ready to answer questions", icon=None)
        else:
            st.caption("Process your documents to enable Q&A.")

        # ── Clear chat ────────────────────────────────────────────────
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------

def _maybe_process_documents() -> None:
    if not st.session_state.uploaded_file_paths:
        return

    current_paths = list(st.session_state.uploaded_file_paths)
    should_process = (
        st.session_state.force_reprocess
        or (not st.session_state.processed)
        or (current_paths != st.session_state.last_processed_file_paths)
    )

    if not should_process:
        return

    with st.sidebar:
        progress_bar = st.progress(0)
        status = st.empty()
        status.info("Processing documents…")

        def _on_progress(frac: float, msg: str) -> None:
            progress_bar.progress(int(frac * 100))
            status.info(msg)

        with st.spinner("Building index…"):
            summaries = process_uploaded_files(
                current_paths,
                vector_store_dir=str(VECTOR_STORE_DIR),
                chunk_size=400,
                overlap=50,
                ocr_lang="en",
                ocr_fallback_for_pdfs=True,
                embed_model_name="all-MiniLM-L6-v2",
                recreate_store=False,
                progress_callback=_on_progress,
            )

        progress_bar.progress(100)
        status.success("Done!")

    st.session_state.processed = True
    st.session_state.last_processed_file_paths = current_paths
    st.session_state.force_reprocess = False

    with st.sidebar:
        st.caption("Chunk summary:")
        for s in summaries:
            st.caption(f"• {Path(s.file_path).name} — {s.num_chunks} chunks")


# ---------------------------------------------------------------------------
# Chat rendering
# ---------------------------------------------------------------------------

def _render_chat() -> None:
    history: List[Dict[str, Any]] = st.session_state.chat_history

    if not history:
        st.markdown(
            """
            <div class="empty-state">
                <div class="icon">🤖</div>
                <strong>DocuMind AI is ready</strong>
                Upload your documents in the sidebar, process them,<br>
                then ask anything about their content.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for item in history:
        # ── User ──────────────────────────────────────────────────────
        with st.chat_message("user"):
            st.write(item["query"])

        # ── Assistant ─────────────────────────────────────────────────
        with st.chat_message("assistant"):
            st.write(item["answer"])
            confidence = item["confidence_score"]
            bar_pct = int(confidence * 100)
            st.markdown(
                f'<span class="confidence-badge">confidence {bar_pct}%</span>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Answer callback
# ---------------------------------------------------------------------------

def _on_get_answer() -> None:
    query = (st.session_state.get("question_input") or "").strip()
    if not query:
        return

    engine = get_chat_engine(str(VECTOR_STORE_DIR))
    with st.spinner("Thinking…"):
        resp = engine.answer(query)

    st.session_state.chat_history.append(
        {
            "query": query,
            "answer": resp.answer,
            "confidence_score": float(resp.confidence_score),
        }
    )

    st.session_state["question_input"] = ""
    

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="DocuMind AI",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom styles
    st.markdown(_CSS, unsafe_allow_html=True)

    _ensure_dirs()
    _init_session_state()

    # Sidebar (uploads + processing)
    _sidebar()
    _maybe_process_documents()

    # ── Header ────────────────────────────────────────────────────────
    st.title("DocuMind AI 🤖")
    st.caption("Ask questions about your uploaded documents — powered by Groq + RAG.")
    st.divider()

    # ── Chat history ──────────────────────────────────────────────────
    _render_chat()

    st.divider()

    # ── Input row ────────────────────────────────────────────────────
    col_input, col_btn = st.columns([5, 1], gap="small")

    with col_input:
        st.text_input(
            "question",
            key="question_input",
            placeholder="Ask anything about your documents…",
            label_visibility="collapsed",
        )

    with col_btn:
        st.button(
            "Send ➤",
            type="primary",
            disabled=not st.session_state.processed,
            on_click=_on_get_answer,
            use_container_width=True,
        )

    # Keyboard-submit hint
    if not st.session_state.processed:
        st.caption("⚠️ Process your documents first to enable Q&A.")


if __name__ == "__main__":
    main()
