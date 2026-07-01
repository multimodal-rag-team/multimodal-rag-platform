"""pgvector cosine similarity search over text, table, and image chunk tables."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Iterator, Literal, TypedDict

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extensions import connection as PsycopgConnection

from indexer.pgvector_indexer import PGVectorIndexer

logger = logging.getLogger(__name__)

ChunkType = Literal["text", "table", "image"]

CHUNK_SOURCES: dict[ChunkType, tuple[str, ChunkType]] = {
    "text": ("text_chunks", "text"),
    "table": ("table_chunks", "table"),
    "image": ("image_chunks", "image"),
}


class RetrievalResult(TypedDict):
    doc_id: str
    page: int
    type: ChunkType
    similarity: float


class ChunkHit(TypedDict):
    doc_id: str
    page: int
    type: ChunkType
    similarity: float
    content: str
    metadata: dict[str, Any]


class PGVectorRetriever:
    """Similarity search against indexed multimodal chunk tables."""

    def __init__(
        self,
        database_url: str | None = None,
        *,
        indexer: PGVectorIndexer | None = None,
    ) -> None:
        self._indexer = indexer or PGVectorIndexer(database_url=database_url)
        self.database_url = self._indexer.database_url

    @contextmanager
    def _connection(self) -> Iterator[PsycopgConnection]:
        conn = psycopg2.connect(self.database_url)
        register_vector(conn)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def search(
        self,
        query: str,
        chunk_type: ChunkType,
        *,
        top_k: int = 5,
        doc_id: str | None = None,
        with_content: bool = False,
    ) -> list[RetrievalResult] | list[ChunkHit]:
        """Run cosine similarity search on a single chunk table."""
        table_name, result_type = CHUNK_SOURCES[chunk_type]
        embedding = self._indexer.embed_text(query)
        return self._search_embedding(
            embedding,
            table_name=table_name,
            result_type=result_type,
            top_k=top_k,
            doc_id=doc_id,
            with_content=with_content,
        )

    def search_hybrid(
        self,
        query: str,
        *,
        top_k: int = 5,
        per_source_k: int | None = None,
        doc_id: str | None = None,
        with_content: bool = False,
    ) -> list[RetrievalResult] | list[ChunkHit]:
        """Search all chunk tables, merge by similarity, return global top-k."""
        limit = per_source_k or top_k
        merged: list[RetrievalResult] | list[ChunkHit] = []
        for chunk_type in CHUNK_SOURCES:
            merged.extend(
                self.search(
                    query,
                    chunk_type,
                    top_k=limit,
                    doc_id=doc_id,
                    with_content=with_content,
                )
            )
        merged.sort(key=lambda row: row["similarity"], reverse=True)
        return merged[:top_k]

    def list_documents(self) -> list[dict[str, Any]]:
        """Return distinct indexed doc_ids with chunk counts per modality."""
        sql = """
            WITH docs AS (
                SELECT doc_id FROM text_chunks
                UNION
                SELECT doc_id FROM table_chunks
                UNION
                SELECT doc_id FROM image_chunks
            )
            SELECT
                d.doc_id,
                (SELECT COUNT(*) FROM text_chunks t WHERE t.doc_id = d.doc_id) AS text_chunks,
                (SELECT COUNT(*) FROM table_chunks tb WHERE tb.doc_id = d.doc_id) AS table_chunks,
                (SELECT COUNT(*) FROM image_chunks i WHERE i.doc_id = d.doc_id) AS image_chunks,
                GREATEST(
                    COALESCE((SELECT MAX(created_at) FROM text_chunks t WHERE t.doc_id = d.doc_id), 'epoch'),
                    COALESCE((SELECT MAX(created_at) FROM table_chunks tb WHERE tb.doc_id = d.doc_id), 'epoch'),
                    COALESCE((SELECT MAX(created_at) FROM image_chunks i WHERE i.doc_id = d.doc_id), 'epoch')
                ) AS last_indexed_at
            FROM docs d
            ORDER BY last_indexed_at DESC
        """
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()

        return [
            {
                "doc_id": row[0],
                "text_chunks": int(row[1]),
                "table_chunks": int(row[2]),
                "image_chunks": int(row[3]),
                "last_indexed_at": row[4].isoformat() if row[4] else None,
            }
            for row in rows
        ]

    def _search_embedding(
        self,
        embedding: list[float],
        *,
        table_name: str,
        result_type: ChunkType,
        top_k: int,
        doc_id: str | None,
        with_content: bool = False,
    ) -> list[RetrievalResult] | list[ChunkHit]:
        if with_content:
            sql = f"""
                SELECT
                    doc_id,
                    page_num,
                    content,
                    metadata,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM {table_name}
                WHERE (%s IS NULL OR doc_id = %s)
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
        else:
            sql = f"""
                SELECT
                    doc_id,
                    page_num,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM {table_name}
                WHERE (%s IS NULL OR doc_id = %s)
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
        params: tuple[Any, ...] = (
            embedding,
            doc_id,
            doc_id,
            embedding,
            top_k,
        )

        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        if with_content:
            return [
                {
                    "doc_id": row[0],
                    "page": int(row[1]),
                    "content": row[2],
                    "metadata": _coerce_metadata(row[3]),
                    "type": result_type,
                    "similarity": float(row[4]),
                }
                for row in rows
            ]

        return [
            {
                "doc_id": row[0],
                "page": int(row[1]),
                "type": result_type,
                "similarity": float(row[2]),
            }
            for row in rows
        ]


def _coerce_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}