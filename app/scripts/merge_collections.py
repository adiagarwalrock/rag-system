"""
Merge two or more Qdrant collections into a single new collection.

Run:
    uv run python -m app.scripts.merge_collections col_a col_b [col_c ...] [--output <name>]

The output collection name defaults to '_merged_<col_a>_<col_b>_...' if --output is not given.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.config import settings
from app.core.logging_config import configure_logging

warnings.filterwarnings(
    "ignore",
    message="Api key is used with an insecure connection.",
    category=UserWarning,
)

configure_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

BATCH_SIZE = 500


def emit(message: str, *, level: int = logging.INFO) -> None:
    print(message)
    logger.log(level, message)


def _client() -> QdrantClient:
    return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)


def _dense_summary(vectors_config) -> tuple[int | None, str | None]:
    """Return (size, distance_name) from a vectors_config dict or VectorParams."""
    if isinstance(vectors_config, dict):
        for params in vectors_config.values():
            if hasattr(params, "size"):
                return params.size, params.distance.name if params.distance else None
        return None, None
    if hasattr(vectors_config, "size"):
        return vectors_config.size, vectors_config.distance.name if vectors_config.distance else None
    return None, None


def _sparse_summary(sparse_config) -> str:
    if not sparse_config:
        return "none"
    names = list(sparse_config.keys())
    modifiers = []
    for v in sparse_config.values():
        mod = getattr(v, "modifier", None)
        modifiers.append(mod.name if mod else "none")
    return ", ".join(f"{n}({m})" for n, m in zip(names, modifiers))


def verify_collections(client: QdrantClient, names: list[str]) -> bool:
    all_ok = True
    for name in names:
        if client.collection_exists(name):
            emit(f"[PASS] Collection found: '{name}'")
        else:
            emit(f"[FAIL] Collection not found: '{name}'", level=logging.ERROR)
            all_ok = False
    return all_ok


def compare_collections(client: QdrantClient, names: list[str]) -> list[dict]:
    infos = []
    for name in names:
        info = client.get_collection(name)
        size, distance = _dense_summary(info.config.params.vectors)
        sparse = _sparse_summary(info.config.params.sparse_vectors)
        points = info.points_count or 0
        infos.append(
            {
                "name": name,
                "points": points,
                "dense_size": size,
                "distance": distance,
                "sparse": sparse,
                "raw": info,
            }
        )

    col_w = max(len(d["name"]) for d in infos) + 2
    emit("")
    emit(f"{'Collection':<{col_w}}  {'Points':>10}  {'Dense size':>10}  {'Distance':<10}  Sparse")
    emit("-" * (col_w + 50))
    for d in infos:
        emit(
            f"{d['name']:<{col_w}}  {d['points']:>10,}  {str(d['dense_size']):>10}  {str(d['distance']):<10}  {d['sparse']}"
        )
    emit("")

    mismatches = []
    ref = infos[0]
    for d in infos[1:]:
        if d["dense_size"] != ref["dense_size"]:
            mismatches.append(
                f"Dense size mismatch: '{ref['name']}' has {ref['dense_size']}, '{d['name']}' has {d['dense_size']}"
            )
        if d["distance"] != ref["distance"]:
            mismatches.append(
                f"Distance mismatch: '{ref['name']}' uses {ref['distance']}, '{d['name']}' uses {d['distance']}"
            )
        if d["sparse"] != ref["sparse"]:
            mismatches.append(
                f"Sparse config mismatch: '{ref['name']}' has {ref['sparse']}, '{d['name']}' has {d['sparse']}"
            )

    if mismatches:
        for m in mismatches:
            emit(f"[WARN] {m}", level=logging.WARNING)
        emit("[WARN] Collections have structural differences — merged result may be inconsistent.", level=logging.WARNING)
    else:
        emit("[PASS] All collections have matching vector configurations.")

    return infos


def confirm(prompt: str) -> bool:
    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("y", "yes")


def create_merged_collection(client: QdrantClient, output_name: str, template_info: dict) -> None:
    raw = template_info["raw"]
    params = raw.config.params

    client.create_collection(
        collection_name=output_name,
        vectors_config=params.vectors,
        sparse_vectors_config=params.sparse_vectors or {},
    )
    emit(f"[INFO] Created collection '{output_name}'.")

    # Recreate payload indexes from the source collection's schema if available,
    # falling back to the standard RAG indexes.
    payload_schema = getattr(raw, "payload_schema", None) or {}
    if payload_schema:
        for field_name, field_schema in payload_schema.items():
            data_type = getattr(field_schema, "data_type", None)
            if data_type is not None:
                client.create_payload_index(
                    collection_name=output_name,
                    field_name=field_name,
                    field_schema=data_type,
                )
    else:
        # Standard RAG collection indexes
        standard_indexes = (
            ("client_id", qdrant_models.PayloadSchemaType.KEYWORD),
            ("document_id", qdrant_models.PayloadSchemaType.KEYWORD),
            ("file_name", qdrant_models.PayloadSchemaType.KEYWORD),
            ("document_version_group", qdrant_models.PayloadSchemaType.KEYWORD),
            ("page_num", qdrant_models.PayloadSchemaType.INTEGER),
            ("slide_num", qdrant_models.PayloadSchemaType.INTEGER),
        )
        for field_name, schema_type in standard_indexes:
            client.create_payload_index(
                collection_name=output_name,
                field_name=field_name,
                field_schema=schema_type,
            )

    emit(f"[INFO] Payload indexes created on '{output_name}'.")


def copy_collection(client: QdrantClient, src: str, dst: str, total_points: int) -> int:
    offset = None
    copied = 0

    while True:
        records, offset = client.scroll(
            collection_name=src,
            limit=BATCH_SIZE,
            with_vectors=True,
            with_payload=True,
            offset=offset,
        )
        if not records:
            break

        points = [
            qdrant_models.PointStruct(
                id=r.id,
                vector=r.vector,  # type: ignore[arg-type]  # VectorStructOutput is runtime-compatible with VectorStruct
                payload=r.payload or {},
            )
            for r in records
        ]
        client.upsert(collection_name=dst, points=points, wait=True)
        copied += len(records)

        progress = f"{copied:,}/{total_points:,}" if total_points else f"{copied:,}"
        print(f"\r  Copying '{src}': {progress} points...", end="", flush=True)

        if offset is None:
            break

    print()  # newline after progress line
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge two or more Qdrant collections into a new one."
    )
    parser.add_argument(
        "collections",
        nargs="+",
        metavar="COLLECTION",
        help="Names of source collections to merge (at least 2).",
    )
    parser.add_argument(
        "--output",
        metavar="NAME",
        default=None,
        help="Name for the merged output collection. Defaults to _merged_<col1>_<col2>_...",
    )
    args = parser.parse_args()

    if len(args.collections) < 2:
        emit("[FAIL] Please provide at least 2 collection names to merge.", level=logging.ERROR)
        sys.exit(1)

    source_names: list[str] = args.collections
    output_name: str = args.output or ("_merged_" + "_".join(source_names))

    emit("Qdrant Collection Merge")
    emit("=" * 60)
    emit(f"Sources : {', '.join(source_names)}")
    emit(f"Output  : {output_name}")
    emit("=" * 60)

    client = _client()

    # Step 1: verify all source collections exist
    emit("\n[1/4] Verifying source collections...")
    if not verify_collections(client, source_names):
        sys.exit(1)

    # Step 2: fetch and compare properties
    emit("\n[2/4] Comparing collection properties...")
    infos = compare_collections(client, source_names)

    # Step 3: confirm
    total_points = sum(d["points"] for d in infos)
    emit(f"\n[3/4] Merge summary")
    emit(f"  Merging {len(source_names)} collections → '{output_name}'")
    emit(f"  Estimated total points: {total_points:,}")

    if client.collection_exists(output_name):
        emit(
            f"[WARN] Output collection '{output_name}' already exists.",
            level=logging.WARNING,
        )
        if not confirm(f"  Overwrite '{output_name}'? [y/N]: "):
            emit("[ABORT] Merge cancelled.")
            sys.exit(0)
        client.delete_collection(output_name)
        emit(f"[INFO] Deleted existing '{output_name}'.")

    if not confirm("\n  Proceed with merge? [y/N]: "):
        emit("[ABORT] Merge cancelled.")
        sys.exit(0)

    # Step 4: create and populate merged collection
    emit(f"\n[4/4] Creating and populating '{output_name}'...")
    create_merged_collection(client, output_name, infos[0])

    total_copied = 0
    try:
        for d in infos:
            copied = copy_collection(client, d["name"], output_name, d["points"])
            emit(f"  [PASS] '{d['name']}': {copied:,} points copied.")
            total_copied += copied
    except Exception:
        emit(
            f"[ERROR] Copy failed mid-merge. Cleaning up partial collection '{output_name}'...",
            level=logging.ERROR,
        )
        try:
            client.delete_collection(output_name)
            emit(f"[INFO] Partial collection '{output_name}' deleted.")
        except Exception:
            emit(
                f"[WARN] Could not delete partial collection '{output_name}' — please delete manually.",
                level=logging.WARNING,
            )
        raise

    emit("")
    emit("=" * 60)
    emit(f"[DONE] Collection '{output_name}' created with {total_copied:,} points.")


if __name__ == "__main__":
    main()
