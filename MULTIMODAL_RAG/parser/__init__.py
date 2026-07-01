"""Document parsing: PDFs, tables, images, and unstructured content."""

from parser.types import BlockType, ParsedBlock

__all__ = ["BlockType", "ParsedBlock", "PDFParser", "blocks_to_json", "parse_pdf"]


def __getattr__(name: str):
    if name in {"PDFParser", "blocks_to_json", "parse_pdf"}:
        from parser.pdf_parser import PDFParser, blocks_to_json, parse_pdf

        return {"PDFParser": PDFParser, "blocks_to_json": blocks_to_json, "parse_pdf": parse_pdf}[
            name
        ]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")