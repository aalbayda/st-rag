
from __future__ import annotations

from app.config import get_settings


PDF_MAGIC: bytes = b"%PDF-"

ZIP_MAGIC: bytes = b"PK\x03\x04"


def sniff_kind(data: bytes, filename: str) -> str | None:
    lower = filename.lower()
    if data[:5] == PDF_MAGIC and lower.endswith(".pdf"):
        return "pdf"
    if data[:4] == ZIP_MAGIC and lower.endswith(".docx"):
        return "docx"
    return None


def validate_upload(files: list[tuple[bytes, str]]) -> list[dict]:
    if not files:
        return []

    settings = get_settings()
    max_files = settings.max_files
    max_file_bytes = settings.max_file_bytes

    if len(files) > max_files:
        return [
            {
                "name": filename,
                "accepted": False,
                "kind": None,
                "reason": f"exceeds {max_files}-file limit per upload",
            }
            for _, filename in files
        ]

    results: list[dict] = []
    for data, filename in files:
        if len(data) > max_file_bytes:
            results.append(
                {
                    "name": filename,
                    "accepted": False,
                    "kind": None,
                    "reason": f"file too large (>{max_file_bytes // (1024 * 1024)}MB)",
                }
            )
            continue

        kind = sniff_kind(data, filename)
        if kind is None:
            results.append(
                {
                    "name": filename,
                    "accepted": False,
                    "kind": None,
                    "reason": "unsupported type: PDF/DOCX only",
                }
            )
            continue

        results.append(
            {
                "name": filename,
                "accepted": True,
                "kind": kind,
                "reason": None,
            }
        )

    return results
