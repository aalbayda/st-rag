
from __future__ import annotations

from app.clients.openai_client import get_openai_client
from app.config import get_settings


def generate_session_name(first_question: str) -> str | None:
    client = get_openai_client()
    settings = get_settings()

    try:
        completion = client.chat.completions.create(
            model=settings.naming_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a short title (3-6 words) for a chat session about "
                        "the user's question. Reply with only the title, no quotes, no punctuation."
                    ),
                },
                {"role": "user", "content": first_question[:500]},
            ],
            max_tokens=20,
            temperature=0.3,
        )
        name = completion.choices[0].message.content
        if name and name.strip():
            return name.strip()[:80]
        return None
    except Exception:
        return None
