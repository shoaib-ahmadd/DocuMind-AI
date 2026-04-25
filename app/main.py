from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from chatbot.chat_engine import ChatConfig, ChatEngine
from utils.file_handler import process_uploaded_files


BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_db"


@st.cache_resource
def get_chat_engine(vector_store_dir: str) -> ChatEngine:
    return ChatEngine(config=ChatConfig(vector_store_dir=vector_store_dir))


def _ensure_dirs() -> None:
    if UPLOAD_DIR.exists() and not UPLOAD_DIR.is_dir():
        UPLOAD_DIR.unlink()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if VECTOR_STORE_DIR.exists() and not VECTOR_STORE_DIR.is_dir():
        VECTOR_STORE_DIR.unlink()
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


def _save_uploaded_file(uploaded_file: Any) -> str:
    _ensure_dirs()
    target_path = UPLOAD_DIR / Path(uploaded_file.name).name
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(target_path)


def _init_session_state() -> None:
    if "uploaded_file_paths" not in st.session_state:
        st.session_state.uploaded_file_paths: List[str] = []
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Dict[str, Any]] = []
    if "last_processed_file_paths" not in st.session_state:
        st.session_state.last_processed_file_paths: List[str] = []
    if "force_reprocess" not in st.session_state:
        st.session_state.force_reprocess = False
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            color: #e5e7eb;
        }
        .block-container {
            max-width: 980px !important;
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        .app-header {
            text-align: center;
            margin-bottom: 1rem;
        }
        .app-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 0.3rem;
        }
        .app-subtitle {
            color: #94a3b8;
            font-size: 1rem;
        }
        .ui-card {
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.25);
            margin-bottom: 1rem;
        }
        .chat-wrap {
            display: flex;
            margin: 0.5rem 0;
        }
        .chat-user {
            justify-content: flex-end;
        }
        .chat-ai {
            justify-content: flex-start;
        }
        .bubble {
            max-width: 78%;
            border-radius: 14px;
            padding: 0.75rem 0.9rem;
            line-height: 1.55;
            font-size: 0.96rem;
            white-space: pre-wrap;
        }
        .bubble-user {
            background: #2563eb;
            color: #ffffff;
            border-bottom-right-radius: 6px;
        }
        .bubble-ai {
            background: #1f2937;
            color: #e5e7eb;
            border: 1px solid #334155;
            border-bottom-left-radius: 6px;
        }
        .answer-box {
            margin-top: 0.5rem;
            background: #0b1220;
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 0.8rem 0.9rem;
            color: #dbeafe;
        }
        div.stButton > button {
            border-radius: 10px !important;
            border: none !important;
            background: linear-gradient(90deg, #2563eb, #1d4ed8) !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 0.55rem 1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">DocuMind AI</div>
            <div class="app-subtitle">Ask questions from your documents</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_upload_card() -> None:
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or image files",
        type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        new_paths = [_save_uploaded_file(uf) for uf in uploaded_files]
        st.session_state.uploaded_file_paths = new_paths
        st.caption("Uploaded files:")
        for p in st.session_state.uploaded_file_paths:
            st.write(f"- {Path(p).name}")
    else:
        st.caption("No files uploaded yet.")

    process_clicked = st.button("Process Documents", use_container_width=True)
    if process_clicked:
        if not st.session_state.uploaded_file_paths:
            st.warning("Upload documents first.")
        else:
            st.session_state.force_reprocess = True

    st.markdown("</div>", unsafe_allow_html=True)


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

    with st.spinner("Processing documents..."):
        summaries = process_uploaded_files(
            current_paths,
            vector_store_dir=str(VECTOR_STORE_DIR),
            chunk_size=500,
            overlap=50,
            ocr_lang="en",
            ocr_fallback_for_pdfs=True,
            embed_model_name="all-MiniLM-L6-v2",
            recreate_store=True,
        )

    st.session_state.processed = True
    st.session_state.last_processed_file_paths = current_paths
    st.session_state.force_reprocess = False

    st.success("Documents processed successfully.")
    with st.expander("Processing summary", expanded=False):
        for s in summaries:
            st.write(f"- {Path(s.file_path).name}: {s.num_chunks} chunks")


def _render_chat_history() -> None:
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.subheader("Chat")
    if not st.session_state.chat_history:
        st.caption("Upload and process documents to start chatting.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for item in st.session_state.chat_history:
        st.markdown(
            f"""
            <div class="chat-wrap chat-user">
                <div class="bubble bubble-user">{item["query"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="chat-wrap chat-ai">
                <div class="bubble bubble-ai">
                    <div class="answer-box">{item["answer"]}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def _render_qa_input() -> None:
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.text_input("Ask a question", key="question_input", placeholder="Type your question...")

    def _on_get_answer() -> None:
        query = (st.session_state.get("question_input") or "").strip()
        if not query:
            st.warning("Please enter a question.")
            return

        engine = get_chat_engine(str(VECTOR_STORE_DIR))
        with st.spinner("Generating answer..."):
            resp = engine.answer(query)

        st.session_state.chat_history.append(
            {
                "query": query,
                "answer": resp.answer,
                "confidence_score": float(resp.confidence_score),
                "source_text": resp.source_text,
            }
        )
        st.session_state["question_input"] = ""
        st.rerun()

    st.button(
        "Get Answer",
        type="primary",
        use_container_width=True,
        disabled=not st.session_state.processed,
        on_click=_on_get_answer,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="DocuMind AI", page_icon=":books:", layout="centered")
    _ensure_dirs()
    _init_session_state()
    _inject_css()
    _render_header()
    _render_upload_card()
    _maybe_process_documents()
    _render_chat_history()
    _render_qa_input()


if __name__ == "__main__":
    main()
