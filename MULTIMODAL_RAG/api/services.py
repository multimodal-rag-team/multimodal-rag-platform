"""Business logic for indexing and RAG queries."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent.router_agent import RouterAgent
from api.config import DEFAULT_TOP_K, LANGFUSE_ENABLED, UPLOAD_DIR
from config import settings
from api.schemas import (
    DocumentInfo,
    ImageItem,
    IndexResponse,
    IndexResult,
    QueryResponse,
    SourceItem,
    TableItem,
)
from indexer.pgvector_indexer import PGVectorIndexer
from retriever.pgvector_retriever import ChunkHit

logger = logging.getLogger(__name__)

ANSWER_SYSTEM_PROMPT = """\
You are a helpful assistant answering questions from indexed PDF documents.
Use only the provided context. If the context is insufficient, say so clearly.
Reference document id and page numbers when citing facts.
"""


def _get_langfuse_client() -> Any:
    """Returns Langfuse client or None if disabled/failed."""
    if not LANGFUSE_ENABLED:
        return None
    try:
        from langfuse import Langfuse
        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        # Verify credentials are valid before returning
        client.auth_check()
        return client
    except Exception:
        logger.warning("Langfuse disabled: auth check failed. Check your API keys.")
        return None


class QueryService:
    def __init__(self, router: RouterAgent | None = None) -> None:
        self.router = router or RouterAgent(top_k=DEFAULT_TOP_K)
        self._openai = OpenAI(api_key=settings.openai_api_key)
        self._langfuse = _get_langfuse_client()

    def answer_query(
        self,
        query: str,
        *,
        doc_id: str | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> QueryResponse:
        if self._langfuse:
            trace = self._langfuse.trace(
                name="query-pipeline",
                input={"query": query, "doc_id": doc_id},
            )
            try:
                result = self._run_query(query, doc_id=doc_id, top_k=top_k)
                trace.update(output={"answer": result.answer})
                return result
            except Exception as e:
                trace.update(output={"error": str(e)})
                raise
        return self._run_query(query, doc_id=doc_id, top_k=top_k)

    def _run_query(self, query: str, *, doc_id: str | None, top_k: int) -> QueryResponse:
        decision = self.router.classify_query(query)
        hits = self.router.retrieve(
            query, route=decision.route, top_k=top_k,
            doc_id=doc_id, with_content=True,
        )
        typed_hits: list[ChunkHit] = [h for h in hits if _is_chunk_hit(h)]  # type: ignore
        sources, tables, images = _partition_hits(typed_hits)
        context = _build_context(typed_hits)
        answer = self._generate_answer(query, context)
        return QueryResponse(
            answer=answer,
            route=decision.route,
            reasoning=decision.reasoning,
            sources=sources,
            tables=tables,
            images=images,
        )

    def _generate_answer(self, query: str, context: str) -> str:
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        response = self._openai.chat.completions.create(
            model=settings.openai_answer_model,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        return (response.choices[0].message.content or "").strip()


class IndexService:
    def __init__(
        self,
        upload_dir: Path | None = None,
        indexer: PGVectorIndexer | None = None,
        parser: Any | None = None,
    ) -> None:
        self.upload_dir = upload_dir or UPLOAD_DIR
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.indexer = indexer or PGVectorIndexer()
        self._parser = parser
        self._langfuse = _get_langfuse_client()
        self.indexer.ensure_schema()

    def _get_parser(self) -> Any:
        if self._parser is None:
            from parser.pdf_parser import PDFParser
            self._parser = PDFParser()
        return self._parser

    def index_files(
        self,
        file_paths: list[tuple[str, Path]],
        *,
        replace: bool = True,
    ) -> IndexResponse:
        results: list[IndexResult] = []

        for filename, path in file_paths:
            doc_id = Path(filename).stem
            logger.info("Indexing %s as doc_id=%s", filename, doc_id)

            if self._langfuse:
                trace = self._langfuse.trace(
                    name=f"index:{doc_id}",
                    input={"filename": filename},
                )
                try:
                    counts = self._index_single(path, doc_id, replace)
                    trace.update(output={"counts": counts})
                except Exception as e:
                    trace.update(output={"error": str(e)})
                    raise
            else:
                counts = self._index_single(path, doc_id, replace)

            logger.info("Indexed %s: %s", doc_id, counts)
            results.append(IndexResult(doc_id=doc_id, filename=filename, counts=counts))

        return IndexResponse(indexed=results)

    def _index_single(self, path: Path, doc_id: str, replace: bool) -> dict:
        blocks = self._get_parser().parse(path, doc_id=doc_id)
        logger.info("Parsed %d blocks from %s", len(blocks), doc_id)
        return self.indexer.index_blocks(blocks, replace_document=replace)


class DocumentService:
    def __init__(self, router: RouterAgent | None = None) -> None:
        self.router = router or RouterAgent()

    def list_documents(self) -> list[DocumentInfo]:
        rows = self.router.retriever.list_documents()
        return [DocumentInfo(**row) for row in rows]


def _is_chunk_hit(hit: Any) -> bool:
    return isinstance(hit, dict) and "content" in hit


def _partition_hits(
    hits: list[ChunkHit],
) -> tuple[list[SourceItem], list[TableItem], list[ImageItem]]:
    sources: list[SourceItem] = []
    tables: list[TableItem] = []
    images: list[ImageItem] = []

    for hit in hits:
        meta = hit.get("metadata") or {}
        if hit["type"] == "table":
            tables.append(TableItem(
                doc_id=hit["doc_id"], page=hit["page"],
                similarity=hit["similarity"], description=hit["content"],
                raw_table=meta.get("raw_table"), metadata=meta,
            ))
        elif hit["type"] == "image":
            images.append(ImageItem(
                doc_id=hit["doc_id"], page=hit["page"],
                similarity=hit["similarity"], description=hit["content"],
                image_path=meta.get("image_path"), metadata=meta,
            ))
        else:
            sources.append(SourceItem(
                doc_id=hit["doc_id"], page=hit["page"], type=hit["type"],
                similarity=hit["similarity"], content=hit["content"], metadata=meta,
            ))
    return sources, tables, images


def _build_context(hits: list[ChunkHit]) -> str:
    parts: list[str] = []
    for i, hit in enumerate(hits, start=1):
        meta = hit.get("metadata") or {}
        header = (
            f"[{i}] doc_id={hit['doc_id']} page={hit['page']} "
            f"type={hit['type']} similarity={hit['similarity']:.3f}"
        )
        body = hit["content"]
        if hit["type"] == "table" and meta.get("raw_table"):
            body = f"{body}\nRaw table: {meta['raw_table']}"
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)