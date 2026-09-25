"""OpenAI client for grounded NASA mission answers."""

from typing import Dict, List
from openai import OpenAI

SYSTEM_PROMPT = """You are a NASA mission operations expert specializing in Apollo 11,
Apollo 13, and Space Shuttle Challenger (STS-51L).

Answer the user's question using only the retrieved NASA context supplied for the
current turn. Cite the supplied sources inline using labels such as [Source 1] and
[Source 2]. Do not invent mission facts, quotations, dates, crew actions, or technical
details that are not supported by the retrieved context.

If the retrieved evidence is incomplete, conflicting, or insufficient, clearly say
what is uncertain and what additional evidence would be needed. Distinguish retrieved
facts from explanation.

Every factual claim about mission events, dates, crew members, spacecraft systems,
procedures, or actions must be directly supported by at least one retrieved source and
cited inline. If a detail is not explicitly supported by the retrieved context, omit it.
Do not use prior model knowledge to fill gaps, even when the detail is historically true.
When summarizing a sequence, preserve chronological order, times, measurements, units,
and technical terminology exactly as stated in the retrieved context; do not reinterpret
or normalize them.
Prefer a shorter fully supported answer over a more complete but weakly supported one.

Be concise but detailed enough to answer the question.
"""

def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-4o-mini",
    max_history_messages: int = 8,
) -> str:
    """Generate a source-grounded response using retrieved NASA context.

    Conversation history is retained as role/content turns, but only a bounded number
    of recent messages is sent so stale retrieved context is not carried forward.
    """
    if not openai_key:
        raise ValueError("An OpenAI API key is required.")
    if not user_message or not user_message.strip():
        raise ValueError("user_message must not be empty.")

    clean_context = (context or "").strip()
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for item in (conversation_history or [])[-max_history_messages:]:
        role = item.get("role")
        text = item.get("content")
        if role in {"user", "assistant"} and isinstance(text, str) and text.strip():
            messages.append({"role": role, "content": text.strip()})

    grounded_user_message = (
        "Retrieved NASA context for this turn:\n"
        "----------------------------------------\n"
        f"{clean_context if clean_context else '[No relevant context was retrieved.]'}\n"
        "----------------------------------------\n\n"
        f"User question: {user_message.strip()}\n\n"
        "Answer from the retrieved context. Cite source labels exactly as provided. "
        "If the context does not support an answer, say so explicitly."
    )
    messages.append({"role": "user", "content": grounded_user_message})

    base_url = "https://openai.vocareum.com/v1" if openai_key.startswith("voc") else None
    client = OpenAI(api_key=openai_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )

    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("OpenAI returned an empty response.")
    return answer.strip()
