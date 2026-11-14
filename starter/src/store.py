"""Store and search chunk vectors using PostgreSQL + pgvector."""
import json
import re
from typing import Any

import psycopg

from .models import Chunk

# Filter keys are whitelisted so user-supplied names can never become SQL.
_COLUMN_FILTERS = {"document_id", "source", "page"}
_METADATA_FILTERS = {
    "section",
    "part",
    "chapter",
    "subsection",
    "profile",
    "profile_type",
    "country",
    "region",
    "income_group",
    "year",
    "document_type",
    "language",
    "category",
    "audience",
    "version",
}
_FILTER_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _vector_literal(vector: list[float]) -> str:
    """Render a Python vector as a pgvector text literal."""
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        """Retain connection settings and validate the embedding dimension."""
        if not isinstance(database_url, str) or not database_url.strip():
            raise ValueError("database_url must be a non-empty connection string")
        if isinstance(dimensions, bool) or not isinstance(dimensions, int):
            raise TypeError("dimensions must be an integer")
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than 0")

        self.database_url = database_url.strip()
        self.dimensions = dimensions

    # ------------------------------------------------------------------ schema
    def create_schema(self) -> None:
        """Create the extension/table/indexes and reject dimension mismatches."""
        with psycopg.connect(self.database_url, connect_timeout=10) as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS chunks (
                        strategy    text        NOT NULL,
                        chunk_id    text        NOT NULL,
                        document_id text        NOT NULL,
                        source      text        NOT NULL,
                        page        integer     NOT NULL,
                        text        text        NOT NULL,
                        metadata    jsonb       NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding   vector({self.dimensions}) NOT NULL,
                        PRIMARY KEY (strategy, chunk_id)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS chunks_strategy_idx
                        ON chunks (strategy)
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS chunks_strategy_document_idx
                        ON chunks (strategy, document_id)
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS chunks_metadata_idx
                        ON chunks USING gin (metadata jsonb_path_ops)
                    """
                )
                cursor.execute(
                    """
                    SELECT format_type(atttypid, atttypmod)
                    FROM pg_attribute
                    WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'
                    """
                )
                row = cursor.fetchone()
                existing = row[0] if row else ""
                expected = f"vector({self.dimensions})"
                if existing != expected:
                    raise ValueError(
                        f"Existing chunks.embedding is {existing or 'unknown'}, "
                        f"but the embedding model needs {expected}. Use a new "
                        "database/table or drop the old vectors before re-ingesting."
                    )

    # ------------------------------------------------------------------ writes
    def upsert(self, strategy: str, chunks: list[Chunk],
               vectors: list[list[float]]) -> None:
        """Replace a strategy's chunks for the documents present in the batch.

        ``chunks`` must contain the complete, current chunk set for every
        ``document_id`` it mentions: obsolete rows for those documents are
        deleted in the same transaction before the new rows are inserted. This
        keeps re-ingestion idempotent and removes stale chunks after a source
        changes. Chunk IDs are unique within the batch.
        """
        if not isinstance(strategy, str) or not strategy.strip():
            raise ValueError("strategy must be a non-empty string")
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Got {len(chunks)} chunks but {len(vectors)} vectors; they must match"
            )
        if not chunks:
            return

        rows: list[tuple[str, str, str, str, int, str, str, str]] = []
        seen_ids: set[str] = set()
        document_ids: set[str] = set()
        for chunk, vector in zip(chunks, vectors):
            if chunk.chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id in batch: {chunk.chunk_id}")
            seen_ids.add(chunk.chunk_id)
            document_ids.add(chunk.document_id)

            if len(vector) != self.dimensions:
                raise ValueError(
                    f"Vector for {chunk.chunk_id} has dimension {len(vector)}, "
                    f"expected {self.dimensions}"
                )
            metadata = dict(chunk.metadata or {})
            rows.append(
                (
                    strategy,
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.source,
                    int(chunk.page),
                    chunk.text,
                    json.dumps(metadata, ensure_ascii=False),
                    _vector_literal(vector),
                )
            )

        with psycopg.connect(self.database_url, connect_timeout=10) as connection:
            with connection.cursor() as cursor:
                # Remove every old chunk of the documents being re-ingested.
                cursor.execute(
                    "DELETE FROM chunks WHERE strategy = %s AND document_id = ANY(%s)",
                    (strategy, sorted(document_ids)),
                )
                cursor.executemany(
                    """
                    INSERT INTO chunks (
                        strategy, chunk_id, document_id, source, page,
                        text, metadata, embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                    """,
                    rows,
                )

    # ------------------------------------------------------------------ reads
    def count(self, strategy: str) -> int:
        """Return the number of stored chunks for one strategy."""
        if not isinstance(strategy, str) or not strategy.strip():
            raise ValueError("strategy must be a non-empty string")
        with psycopg.connect(self.database_url, connect_timeout=10) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT count(*) FROM chunks WHERE strategy = %s", (strategy,)
                )
                row = cursor.fetchone()
                return int(row[0]) if row else 0

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Filter permitted metadata in SQL and return top-k cosine matches.

        ``filters`` may use ``document_id``, ``source``, ``page`` or any
        whitelisted metadata field (``section``, ``country``, ``year``, ...).
        Every value is passed as a query parameter.
        """
        if not isinstance(strategy, str) or not strategy.strip():
            raise ValueError("strategy must be a non-empty string")
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query vector has dimension {len(query_vector)}, "
                f"expected {self.dimensions}"
            )
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        conditions = ["strategy = %s"]
        parameters: list[Any] = [strategy]
        metadata_filter: dict[str, Any] = {}

        for key, value in (filters or {}).items():
            if value is None:
                continue
            if not isinstance(key, str) or not _FILTER_KEY_RE.match(key):
                raise ValueError(f"Invalid filter key: {key!r}")
            if key in _COLUMN_FILTERS:
                if key == "page":
                    try:
                        value = int(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"page filter must be an integer: {value!r}") from exc
                conditions.append(f"{key} = %s")
                parameters.append(value)
            elif key in _METADATA_FILTERS:
                metadata_filter[key] = value
            else:
                raise ValueError(
                    f"Unsupported filter key {key!r}; allowed keys are "
                    f"{sorted(_COLUMN_FILTERS | _METADATA_FILTERS)}"
                )

        if metadata_filter:
            conditions.append("metadata @> %s::jsonb")
            parameters.append(json.dumps(metadata_filter, ensure_ascii=False))

        vector = _vector_literal(query_vector)
        sql = f"""
            SELECT chunk_id, document_id, source, page, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
            WHERE {' AND '.join(conditions)}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        # Order: SELECT vector, WHERE values, ORDER BY vector, LIMIT.
        query_parameters = [vector, *parameters, vector, top_k]

        with psycopg.connect(self.database_url, connect_timeout=10) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, query_parameters)
                rows = cursor.fetchall()

        results: list[dict[str, Any]] = []
        for chunk_id, document_id, source, page, text, metadata, score in rows:
            payload = dict(metadata or {})
            payload.setdefault("document_id", document_id)
            payload.setdefault("source", source)
            payload.setdefault("page", page)
            results.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "source": source,
                    "page": page,
                    "text": text,
                    "metadata": payload,
                    "score": float(score),
                }
            )
        return results
