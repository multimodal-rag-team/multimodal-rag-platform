"""LangChain router agent: classify query modality and retrieve from pgvector."""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from config import settings
from retriever.pgvector_retriever import (
    ChunkHit,
    ChunkType,
    PGVectorRetriever,
    RetrievalResult,
)

RouteType = Literal["text", "table", "image", "hybrid"]

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """\
You are a query router for a multimodal document retrieval system indexed with:
- text chunks (narrative prose and headings)
- table chunks (tabular / numeric data described in natural language)
- image chunks (figures, charts, diagrams described in natural language)

Classify the user query into exactly one route:
- text: prose, definitions, explanations, policies, narrative facts
- table: numeric comparisons, metrics, rows/columns, spreadsheet-like facts
- image: charts, figures, diagrams, photos, visual layout, "what does the figure show"
- hybrid: clearly needs more than one modality (e.g. text + table, or overview spanning types)

Choose hybrid only when multiple modalities are required. Otherwise pick the single best route.
"""


class RouteDecision(BaseModel):
    """Structured routing label for a user query."""

    route: Literal["text", "table", "image", "hybrid"] = Field(
        description="Retrieval modality: text, table, image, or hybrid"
    )
    reasoning: str = Field(
        default="",
        description="Brief justification for the chosen route",
    )


class RouterAgent:
    """Classify queries with LangChain, then search pgvector chunk tables."""

    def __init__(
        self,
        retriever: PGVectorRetriever | None = None,
        *,
        database_url: str | None = None,
        router_model: str | None = None,
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever or PGVectorRetriever(
            database_url=database_url or settings.database_url
        )
        self.router_model = router_model or settings.openai_router_model
        self.default_top_k = top_k
        self._structured_llm = self._build_structured_llm()

    def _build_structured_llm(self) -> Any:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is required for RouterAgent. "
                "Install with: pip install langchain-openai"
            ) from exc

        llm = ChatOpenAI(
            model=self.router_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
        return llm.with_structured_output(RouteDecision)

    def classify_query(self, query: str) -> RouteDecision:
        """Classify a query with reasoning via structured LangChain output."""
        decision: RouteDecision = self._structured_llm.invoke(
            [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ]
        )
        logger.info("Routed query to %s: %s", decision.route, decision.reasoning)
        return decision

    def classify(self, query: str) -> RouteType:
        """Classify a query as text, table, image, or hybrid."""
        return self.classify_query(query).route

    def retrieve(
        self,
        query: str,
        route: RouteType | None = None,
        *,
        top_k: int | None = None,
        doc_id: str | None = None,
        with_content: bool = False,
    ) -> list[RetrievalResult] | list[ChunkHit]:
        """Classify (if needed), search pgvector, return ranked results."""
        k = top_k if top_k is not None else self.default_top_k
        chosen = route or self.classify(query)

        if chosen == "hybrid":
            return self.retriever.search_hybrid(
                query, top_k=k, doc_id=doc_id, with_content=with_content
            )

        return self.retriever.search(
            query,
            cast(ChunkType, chosen),
            top_k=k,
            doc_id=doc_id,
            with_content=with_content,
        )

    def run(
        self,
        query: str,
        *,
        top_k: int | None = None,
        doc_id: str | None = None,
    ) -> dict[str, Any]:
        """
        End-to-end: classify, retrieve, return route label and results.

        Each result has: doc_id, page, type, similarity.
        """
        decision = self.classify_query(query)
        results = self.retrieve(
            query,
            route=decision.route,
            top_k=top_k,
            doc_id=doc_id,
        )
        return {
            "route": decision.route,
            "reasoning": decision.reasoning,
            "results": results,
        }


def route_and_retrieve(
    query: str,
    *,
    database_url: str | None = None,
    top_k: int = 5,
    doc_id: str | None = None,
) -> dict[str, Any]:
    """Convenience wrapper around :class:`RouterAgent`."""
    return RouterAgent(database_url=database_url, top_k=top_k).run(
        query, top_k=top_k, doc_id=doc_id
    )


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO)
    cli = argparse.ArgumentParser(description="Route a query and retrieve pgvector chunks.")
    cli.add_argument("query")
    cli.add_argument("--top-k", type=int, default=5)
    cli.add_argument("--doc-id", default=None)
    args = cli.parse_args()

    output = route_and_retrieve(
        args.query,
        top_k=args.top_k,
        doc_id=args.doc_id,
    )
    print(json.dumps(output, indent=2))