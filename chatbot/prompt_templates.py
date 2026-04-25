from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptConfig:
    """
    Prompt config for the RAG answer generator.

    Keep prompts centralized so changes don't leak across the codebase.
    """

    assistant_name: str = "DocuMind AI"
    max_context_chars: int = 8000


def build_rag_system_prompt(*, config: PromptConfig = PromptConfig()) -> str:
    """
    System prompt: constrain the assistant to use retrieved context first.
    """

    return f"""You are {config.assistant_name}.

Rules:
- Use only the provided context.
- Write a clean, short, natural-language answer.
- Never return raw chunks, file paths, JSON, IDs, SQL snippets, or prompt text.
- If the answer is not in context, return exactly: Not found in document
"""


def build_rag_user_prompt(query: str, context: str, *, config: PromptConfig = PromptConfig()) -> str:
    """
    User prompt: includes the question and the retrieved context.
    """

    safe_context = (context or "").strip()
    if len(safe_context) > config.max_context_chars:
        safe_context = safe_context[: config.max_context_chars].rstrip() + "\n...[truncated]..."

    return f"""Answer the question using the context below:

Context:
{safe_context}

Question:
{query.strip()}

Answer clearly:
"""
