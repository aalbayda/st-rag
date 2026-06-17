
from __future__ import annotations

import hashlib
import re
import unicodedata


def make_file_id(filename: str) -> str:
    sha_hex = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16]

    basename = filename.replace("\\", "/").rstrip("/")
    if "/" in basename:
        basename = basename.rsplit("/", 1)[-1]

    stem = basename.rsplit(".", 1)[0] if "." in basename else basename

    nfkd = unicodedata.normalize("NFKD", stem)
    ascii_stem = "".join(c for c in nfkd if not unicodedata.combining(c))

    slug = ascii_stem.lower()
    slug = re.sub(r"[^a-z0-9_]", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    slug = slug.strip("-_")

    slug = slug[:48]

    if slug:
        file_id = f"{slug}-{sha_hex}"
    else:
        file_id = sha_hex

    return file_id
