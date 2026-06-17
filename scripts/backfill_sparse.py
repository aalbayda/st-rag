
from __future__ import annotations

import sys

from sqlmodel import Session, select

from app.clients.pinecone_client import get_index, get_pinecone, sparse_encode_passages
from app.config import get_settings
from app.db.engine import get_engine
from app.db.models import File


def backfill() -> None:
    settings = get_settings()
    index = get_index()

    with Session(get_engine()) as session:
        rows = session.exec(select(File).where(File.status == "ready")).all()
        file_ids = [row.id for row in rows]

    if not file_ids:
        print("No ready files found in SQLite. Nothing to backfill.")
        return

    print(f"Backfilling {len(file_ids)} file(s)...")

    total_upserted = 0
    total_skipped = 0

    for file_id in file_ids:
        print(f"  [{file_id}] Listing vectors...")
        try:
            ids = [
                item.id
                for page in index.list(prefix=f"{file_id}#")
                for item in page.vectors
                if item.id is not None
            ]

            if not ids:
                print(f"  [{file_id}] No vectors found. Skipping.")
                continue

            print(f"  [{file_id}] Found {len(ids)} vector(s). Fetching in batches...")

            upsert_batch: list[dict] = []
            FETCH_BATCH = 100

            for batch_start in range(0, len(ids), FETCH_BATCH):
                batch_ids = ids[batch_start : batch_start + FETCH_BATCH]
                fetch_result = index.fetch(ids=batch_ids)

                fetched = fetch_result.vectors if hasattr(fetch_result, "vectors") else {}

                for vec_id, vec_obj in fetched.items():
                    orig_values = getattr(vec_obj, "values", None)
                    if not orig_values:
                        print(f"    WARNING: vector {vec_id} has no dense values. Skipping.")
                        total_skipped += 1
                        continue

                    orig_metadata = dict(getattr(vec_obj, "metadata", None) or {})
                    chunk_text = orig_metadata.get("chunk_text", "")

                    if not chunk_text:
                        print(f"    WARNING: vector {vec_id} has no chunk_text in metadata. Skipping.")
                        total_skipped += 1
                        continue

                    try:
                        sparse_vecs = sparse_encode_passages([chunk_text])
                        sparse_vec = sparse_vecs[0]
                    except Exception as enc_exc:
                        print(f"    WARNING: sparse encode failed for {vec_id}: {enc_exc}. Skipping.")
                        total_skipped += 1
                        continue

                    upsert_batch.append(
                        {
                            "id": vec_id,
                            "values": list(orig_values),
                            "sparse_values": sparse_vec,
                            "metadata": orig_metadata,
                        }
                    )

            if upsert_batch:
                UPSERT_BATCH = settings.upsert_batch_size
                for i in range(0, len(upsert_batch), UPSERT_BATCH):
                    chunk = upsert_batch[i : i + UPSERT_BATCH]
                    index.upsert(vectors=chunk, batch_size=UPSERT_BATCH)

                print(f"  [{file_id}] Re-upserted {len(upsert_batch)} vector(s).")
                total_upserted += len(upsert_batch)

        except Exception as file_exc:
            print(f"  [{file_id}] ERROR: {file_exc}. Continuing to next file.")

    print(f"\nBackfill complete: {total_upserted} vector(s) re-upserted, {total_skipped} skipped.")


if __name__ == "__main__":
    backfill
