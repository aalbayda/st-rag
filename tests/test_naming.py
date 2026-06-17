
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_completion(content: str | None) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _mock_client(content: str | None = "Capital of France") -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = _make_completion(content)
    return client


@patch("app.services.naming.get_openai_client")
def test_returns_name_for_normal_question(mock_get_client):
    mock_get_client.return_value = _mock_client("Capital of France")
    from app.services.naming import generate_session_name
    result = generate_session_name("What is the capital of France?")
    assert result == "Capital of France"
    assert len(result) <= 80


@patch("app.services.naming.get_openai_client")
def test_uses_naming_model(mock_get_client):
    client = _mock_client("Some Title")
    mock_get_client.return_value = client
    from app.config import get_settings
    from app.services.naming import generate_session_name
    generate_session_name("Tell me about AI")
    call_kwargs = client.chat.completions.create.call_args
    model_used = call_kwargs[1].get("model") or call_kwargs[0][0]
    assert model_used == get_settings().naming_model


@patch("app.services.naming.get_openai_client")
def test_does_not_construct_new_openai_client(mock_get_client):
    mock_get_client.return_value = _mock_client("Test Title")
    with patch("openai.OpenAI") as mock_raw:
        from app.services.naming import generate_session_name
        generate_session_name("Hello")
        mock_raw.assert_not_called()
    mock_get_client.assert_called()


@patch("app.services.naming.get_openai_client")
def test_exception_returns_none(mock_get_client):
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("network failure")
    mock_get_client.return_value = client
    from app.services.naming import generate_session_name
    result = generate_session_name("What is 2+2?")
    assert result is None


@patch("app.services.naming.get_openai_client")
def test_none_content_returns_none(mock_get_client):
    mock_get_client.return_value = _mock_client(None)
    from app.services.naming import generate_session_name
    result = generate_session_name("Some question")
    assert result is None


@patch("app.services.naming.get_openai_client")
def test_empty_string_content_returns_none(mock_get_client):
    mock_get_client.return_value = _mock_client("")
    from app.services.naming import generate_session_name
    result = generate_session_name("Some question")
    assert result is None


@patch("app.services.naming.get_openai_client")
def test_whitespace_only_content_returns_none(mock_get_client):
    mock_get_client.return_value = _mock_client("   ")
    from app.services.naming import generate_session_name
    result = generate_session_name("Some question")
    assert result is None


@patch("app.services.naming.get_openai_client")
def test_question_truncated_to_500_chars(mock_get_client):
    client = _mock_client("Long Question Title")
    mock_get_client.return_value = client
    from app.services.naming import generate_session_name
    long_question = "x" * 1000
    generate_session_name(long_question)
    call_messages = client.chat.completions.create.call_args[1]["messages"]
    user_content = call_messages[1]["content"]
    assert len(user_content) == 500


@patch("app.services.naming.get_openai_client")
def test_result_truncated_to_80_chars(mock_get_client):
    long_name = "A" * 200
    mock_get_client.return_value = _mock_client(long_name)
    from app.services.naming import generate_session_name
    result = generate_session_name("Some question")
    assert result is not None
    assert len(result) == 80


@patch("app.services.naming.get_openai_client")
def test_empty_question_still_calls_api(mock_get_client):
    client = _mock_client("Empty Question Chat")
    mock_get_client.return_value = client
    from app.services.naming import generate_session_name
    result = generate_session_name("")
    client.chat.completions.create.assert_called_once()
    assert result == "Empty Question Chat"
