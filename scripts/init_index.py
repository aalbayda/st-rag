
from __future__ import annotations

import sys

from app.clients.pinecone_client import ensure_index


def main() -> int:
    print("Bootstrapping Pinecone index (dimension=3072, metric=dotproduct)...")

    try:
        desc = ensure_index()
    except Exception as exc:
        print(f"ERROR: Failed to bootstrap index: {exc}", file=sys.stderr)
        print(
            "Check that PINECONE_API_KEY and PINECONE_INDEX_NAME env vars are set correctly.",
            file=sys.stderr,
        )
        return 1

    index_name = getattr(desc, "name", None) or (desc.get("name") if isinstance(desc, dict) else "?")
    dimension  = getattr(desc, "dimension", None) or (desc.get("dimension") if isinstance(desc, dict) else "?")
    metric     = getattr(desc, "metric", None) or (desc.get("metric") if isinstance(desc, dict) else "?")

    print(f"  index name : {index_name}")
    print(f"  dimension  : {dimension}")
    print(f"  metric     : {metric}")
    print("Index ready.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
