"""Shared parser types (no heavy PDF dependencies)."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

BlockType = Literal["heading", "text", "table", "image"]


class ParsedBlock(TypedDict):
    doc_id: str
    page: int
    type: BlockType
    content: str
    metadata: dict[str, Any]