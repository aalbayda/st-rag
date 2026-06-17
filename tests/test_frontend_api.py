
from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def clear_caches():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
    try:
        from frontend.api import get_client

        get_client.cache_clear()
    except (ImportError, AttributeError):
        pass


def _make_fake_uploaded(name: str, content: bytes, content_type: str = "application/pdf"):
    f = MagicMock()
    f.name = name
    f.type = content_type
    f.size = len(content)
    f.getvalue.return_value = content
    return f


class TestImport:
    def test_import_cleanly(self):
        from frontend.api import get_client, get_files, post_chat, post_files  # noqa: F401

    def test_no_vendor_sdk_imports(self):
        import ast
        import os

        worktree = os.path.dirname(os.path.dirname(__file__))
        src = open(os.path.join(worktree, "frontend", "api.py")).read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                else:
                    names = [node.module or ""]
                for name in names:
                    assert not name.startswith("openai"), f"openai import found: {name}"
                    assert not name.startswith("pinecone"), f"pinecone import found: {name}"


class TestGetClient:
    def test_get_client_returns_httpx_client(self):
        from frontend.api import get_client

        with patch("frontend.api.get_settings") as mock_settings:
            mock_settings.return_value.api_base_url = "http://localhost:9999"
            get_client.cache_clear()
            client = get_client()
            assert isinstance(client, httpx.Client)

    def test_get_client_is_cached(self):
        from frontend.api import get_client

        with patch("frontend.api.get_settings") as mock_settings:
            mock_settings.return_value.api_base_url = "http://localhost:9999"
            get_client.cache_clear()
            c1 = get_client()
            c2 = get_client()
            assert c1 is c2, "get_client() must be a cached singleton"

    def test_get_client_uses_api_base_url(self):
        from frontend.api import get_client

        with patch("frontend.api.get_settings") as mock_settings:
            mock_settings.return_value.api_base_url = "http://myserver:1234"
            get_client.cache_clear()
            client = get_client()
            assert "myserver" in str(client.base_url)


class TestPostChat:
    def _make_mock_response(self, json_data: dict, status_code: int = 200) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data
        resp.raise_for_status.return_value = None
        return resp

    def test_post_chat_sends_correct_path_and_body(self):
        from frontend.api import post_chat

        answer_data = {
            "answer": "Paris",
            "reasoning": "The document states...",
            "citations": [],
            "abstained": False,
        }
        mock_resp = self._make_mock_response(answer_data)
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp

        with patch("frontend.api.get_client", return_value=mock_client):
            result = post_chat("What is the capital of France?", "sess-123")

        mock_client.post.assert_called_once_with(
            "/chat",
            json={
                "question": "What is the capital of France?",
                "session_id": "sess-123",
            },
        )
        assert result == answer_data

    def test_post_chat_returns_json_dict(self):
        from frontend.api import post_chat

        expected = {"answer": "test", "reasoning": "r", "citations": [], "abstained": False}
        mock_resp = self._make_mock_response(expected)
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp

        with patch("frontend.api.get_client", return_value=mock_client):
            result = post_chat("test question", "sess-1")

        assert result == expected

    def test_post_chat_calls_raise_for_status(self):
        from frontend.api import post_chat

        mock_resp = self._make_mock_response({})
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp

        with patch("frontend.api.get_client", return_value=mock_client):
            post_chat("q", "sess-1")

        mock_resp.raise_for_status.assert_called_once()

    def test_post_chat_propagates_connect_error(self):
        from frontend.api import post_chat

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")

        with patch("frontend.api.get_client", return_value=mock_client):
            with pytest.raises(httpx.ConnectError):
                post_chat("test", "sess-1")

    def test_post_chat_propagates_http_status_error(self):
        from frontend.api import post_chat

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Internal Server Error",
            request=MagicMock,
            response=mock_resp,
        )
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp

        with patch("frontend.api.get_client", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                post_chat("test", "sess-1")


class TestPostFiles:
    def test_post_files_sends_correct_multipart_field(self):
        from frontend.api import post_files

        fake_file = _make_fake_uploaded("report.pdf", b"pdf-bytes", "application/pdf")
        expected_result = {"results": [{"name": "report.pdf", "accepted": True, "file_id": "abc"}]}

        mock_resp = MagicMock()
        mock_resp.status_code = 202
        mock_resp.json.return_value = expected_result
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp

        with patch("frontend.api.get_client", return_value=mock_client):
            result = post_files([fake_file])

        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/files"

        files_kwarg = call_args[1]["files"]
        assert len(files_kwarg) == 1
        field_name, file_tuple = files_kwarg[0]
        assert field_name == "files"
        assert file_tuple[0] == "report.pdf"
        assert file_tuple[1] == b"pdf-bytes"
        assert "pdf" in file_tuple[2]

        assert result == expected_result

    def test_post_files_sends_repeated_files_field_for_multiple(self):
        from frontend.api import post_files

        f1 = _make_fake_uploaded("a.pdf", b"aaa", "application/pdf")
        f2 = _make_fake_uploaded("b.docx", b"bbb", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

        mock_resp = MagicMock()
        mock_resp.status_code = 202
        mock_resp.json.return_value = {"results": []}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp

        with patch("frontend.api.get_client", return_value=mock_client):
            post_files([f1, f2])

        files_kwarg = mock_client.post.call_args[1]["files"]
        assert len(files_kwarg) == 2
        assert all(f[0] == "files" for f in files_kwarg)
        assert files_kwarg[0][1][0] == "a.pdf"
        assert files_kwarg[1][1][0] == "b.docx"

    def test_post_files_returns_json_dict(self):
        from frontend.api import post_files

        expected = {"results": [{"name": "x.pdf", "accepted": True, "file_id": "xyz"}]}
        fake_file = _make_fake_uploaded("x.pdf", b"data")

        mock_resp = MagicMock()
        mock_resp.status_code = 202
        mock_resp.json.return_value = expected
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp

        with patch("frontend.api.get_client", return_value=mock_client):
            result = post_files([fake_file])

        assert result == expected

    def test_post_files_uses_octet_stream_fallback_for_missing_type(self):
        from frontend.api import post_files

        fake_file = _make_fake_uploaded("data.bin", b"data", "")
        fake_file.type = None

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp

        with patch("frontend.api.get_client", return_value=mock_client):
            post_files([fake_file])

        files_kwarg = mock_client.post.call_args[1]["files"]
        assert files_kwarg[0][1][2] == "application/octet-stream"


class TestGetFiles:
    def test_get_files_sends_get_to_correct_path(self):
        from frontend.api import get_files

        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        with patch("frontend.api.get_client", return_value=mock_client):
            get_files()

        mock_client.get.assert_called_once_with("/files")

    def test_get_files_returns_list(self):
        from frontend.api import get_files

        expected = [
            {
                "id": "f1",
                "name": "report.pdf",
                "byte_size": 1024,
                "page_count": 5,
                "chunk_count": 20,
                "status": "indexed",
                "stage": "indexed",
                "error": None,
                "created_at": "2026-01-01T00:00:00",
            }
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = expected
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        with patch("frontend.api.get_client", return_value=mock_client):
            result = get_files()

        assert result == expected

    def test_get_files_returns_empty_list_when_no_files(self):
        from frontend.api import get_files

        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        with patch("frontend.api.get_client", return_value=mock_client):
            result = get_files()

        assert result == []
