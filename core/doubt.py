"""
core/doubt.py
=============
Contextual Doubt Assistant for clarifying specific AI-generated responses.

This module provides a dedicated assistant that answers follow-up doubts
about an individual AI response without disturbing the main conversation,
LangGraph checkpoint thread, or state.
"""

from __future__ import annotations

import logging
from typing import Generator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from core.config import settings

logger = logging.getLogger(__name__)

DOUBT_SYSTEM_TEMPLATE = """You are the Contextual Doubt Assistant in an AI Research Assistant system.
The user is asking a specific doubt, clarification, or follow-up question regarding a particular response generated earlier in their conversation.

=== ORIGINAL USER PROMPT ===
{original_prompt}

=== ORIGINAL AI RESPONSE BEING EXPLAINED ===
{original_response}

=== YOUR GUIDELINES ===
1. Anchor your explanations directly to the concepts, claims, steps, code, or data in the original response above.
2. Break down complex points into easy-to-understand explanations.
3. If the user asks for examples, simplifications, step-by-step walkthroughs, or code explanations, provide them clearly.
4. Maintain continuity with earlier messages in this doubt conversation thread.
5. Use clean GitHub-flavored markdown with code blocks, bold highlights, or bullet points where helpful.
6. Keep your answer focused, accurate, friendly, and concise.
"""


def _build_doubt_messages(
    original_prompt: str,
    original_response: str,
    doubt_history: list[dict],
    new_query: str,
) -> list:
    """Construct the list of LangChain messages for the doubt assistant LLM."""
    prompt_excerpt = (original_prompt or "").strip() or "(No prior user prompt available)"
    response_excerpt = (original_response or "").strip() or "(No response content available)"

    system_text = DOUBT_SYSTEM_TEMPLATE.format(
        original_prompt=prompt_excerpt,
        original_response=response_excerpt,
    )

    messages = [SystemMessage(content=system_text)]

    # Append prior doubt thread messages
    for msg in doubt_history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Append the new user doubt
    messages.append(HumanMessage(content=new_query.strip()))
    return messages


def _get_doubt_llm() -> ChatGroq:
    """Return a configured ChatGroq instance for the doubt assistant."""
    return ChatGroq(
        model=settings.groq_model,
        temperature=0.2,
        groq_api_key=settings.groq_api_key,
        streaming=True,
    )


def answer_doubt(
    original_prompt: str,
    original_response: str,
    doubt_history: list[dict],
    new_query: str,
) -> str:
    """
    Generate a complete answer to a user doubt synchronously.

    Parameters
    ----------
    original_prompt : str
        The prompt that originally produced the response being questioned.
    original_response : str
        The text of the assistant response being questioned.
    doubt_history : list[dict]
        Chronological list of prior doubt messages `[{"role": "user"|"assistant", "content": str}]`.
    new_query : str
        The user's latest doubt question.

    Returns
    -------
    str
        The Doubt Assistant's answer.
    """
    try:
        llm = _get_doubt_llm()
        messages = _build_doubt_messages(
            original_prompt=original_prompt,
            original_response=original_response,
            doubt_history=doubt_history,
            new_query=new_query,
        )
        response = llm.invoke(messages)
        return str(response.content)
    except Exception as exc:
        logger.error("Doubt Assistant error: %s", exc)
        return f"⚠️ Unable to answer doubt: {exc}"


def stream_doubt_answer(
    original_prompt: str,
    original_response: str,
    doubt_history: list[dict],
    new_query: str,
) -> Generator[str, None, None]:
    """
    Stream tokens from the Doubt Assistant for real-time rendering in Streamlit.

    Yields
    ------
    str
        Incremental token chunks.
    """
    try:
        llm = _get_doubt_llm()
        messages = _build_doubt_messages(
            original_prompt=original_prompt,
            original_response=original_response,
            doubt_history=doubt_history,
            new_query=new_query,
        )
        for chunk in llm.stream(messages):
            if chunk.content:
                yield str(chunk.content)
    except Exception as exc:
        logger.error("Doubt Assistant streaming error: %s", exc)
        yield f"\n\n⚠️ Error generating doubt response: {exc}"
