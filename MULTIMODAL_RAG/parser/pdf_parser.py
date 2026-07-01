"""
PDF Parser — Multimodal RAG
===========================
Table extraction uses a Vision-first cascade:

  1. GPT-4o Vision  — renders each page as PNG → extracts tables as JSON
                      (handles LaTeX math, merged cells, multi-line headers)
  2. Camelot lattice — fallback for bordered tables in non-academic PDFs
  3. pdfplumber lines — fallback for ruled tables
  4. pdfplumber default — last resort

Rate limiting — guaranteed no page is skipped:
  - Batch pause every N pages to stay under OpenAI TPM limits
  - On 429 rate limit errors: retry with exponential backoff, NO page is
    ever permanently dropped due to a rate limit
  - On non-rate-limit errors: per-page fallback to Camelot/pdfplumber
    so that page still gets table extraction via a different method
  - Second pass: any page that failed vision gets retried at the end
    after a longer cooldown, before falling back
  - Zero quality changes to successful extractions: same model, same
    prompt, same DPI=250, same detail=high

Image extraction — two complementary strategies:
  - Raster images  : embedded image XObjects via PyMuPDF get_images()
                     (JPEG, PNG, TIFF, raw bitmap embedded in the PDF)
  - Vector figures : matplotlib plots, architecture diagrams, flowcharts,
                     and other content drawn with PDF path operators.
                     Detected by clustering drawing paths and anchoring to
                     "Figure N." captions; rendered to PNG via page crop.
                     Handles academic papers (arXiv), technical manuals,
                     product datasheets, and any PDF where figures are
                     generated programmatically rather than rasterised.
"""

from __future__ import annotations

import base64
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

import fitz
import pdfplumber
from openai import OpenAI
from PIL import Image

from config import settings
from parser.types import BlockType, ParsedBlock

if TYPE_CHECKING:
    from unstructured.documents.elements import Element

logger = logging.getLogger(__name__)

# ── Category sets ─────────────────────────────────────────────────
HEADING_CATEGORIES = frozenset(
    {"Title", "Header", "Subtitle", "Heading", "Subheading"}
)
TEXT_CATEGORIES = frozenset({
    "NarrativeText", "Text", "ListItem", "BulletedText",
    "NumberedListItem", "FigureCaption", "Footer", "Footnote",
    "UncategorizedText",
})
SKIP_CATEGORIES = frozenset({"Table", "Image", "PageBreak"})

# ── Raster image settings ─────────────────────────────────────────
DEFAULT_VISION_PROMPT = (
    "Describe this image in detail for document search and retrieval. "
    "Include visible text, charts, diagrams, and semantic meaning."
)
MIN_IMAGE_WIDTH  = 100   # pixels — applies to BOTH raster and vector renders
MIN_IMAGE_HEIGHT = 100   # pixels
JPEG_QUALITY     = 75

# ── Vector figure extraction settings ────────────────────────────
# Caption anchor pattern — matches "Figure 1", "Fig. 2", "FIGURE 3", etc.
# Intentionally broad to cover academic papers, technical manuals, and reports.
FIGURE_CAPTION_RE = re.compile(
    r"^(Figure|Fig\.?|FIGURE|FIG\.?)\s*[\d]+",
    re.IGNORECASE,
)
VECTOR_FIG_RENDER_DPI     = 150   # DPI for rendering vector figure crops
VECTOR_FIG_CLUSTER_GAP    = 30    # pt gap between path groups → new cluster
VECTOR_FIG_MIN_PATHS      = 3     # min drawing paths to qualify as a figure
VECTOR_FIG_MIN_PT_WIDTH   = 40    # pt — pre-render size gate (generous)
VECTOR_FIG_MIN_PT_HEIGHT  = 30    # pt — pre-render size gate (generous)
VECTOR_FIG_PADDING        = 10    # pt padding around detected figure bounds
VECTOR_FIG_CAPTION_MARGIN = 6     # pt below caption bottom to include
# Uncaptioned standalone figures (e.g. inline diagrams, product schematics)
VECTOR_FIG_UNCAP_MIN_W    = 100   # pt — larger gate for uncaptioned figures
VECTOR_FIG_UNCAP_MIN_H    = 80    # pt

# ── Table vision settings — UNCHANGED for quality ─────────────────
TABLE_RENDER_DPI = 200   # higher = better OCR accuracy for math symbols

# ── Rate limit settings ───────────────────────────────────────────
PAGES_PER_BATCH         = 10
BATCH_PAUSE_SECONDS     = 12.0
MAX_RETRIES_PER_PAGE    = 8
BASE_BACKOFF_SECONDS    = 1.0
MAX_BACKOFF_SECONDS     = 60.0
SECOND_PASS_COOLDOWN    = 30.0

TABLE_VISION_PROMPT = """You are a precise table extractor for scientific papers.

Look at this page carefully. Find ALL tables on this page.

For each table:
1. Extract EVERY row and EVERY column exactly as shown
2. Preserve mathematical notation (e.g. O(n²·d), O(log_k(n)))
3. Preserve merged/spanning headers by repeating the header text
4. Include the header row as the first row
5. Keep all numeric values exact

Return ONLY a JSON array. Each element is one table (array of rows). Each row is an array of cell strings.

Example format:
[
  [
    ["Layer Type", "Complexity per Layer", "Sequential Operations", "Maximum Path Length"],
    ["Self-Attention", "O(n²·d)", "O(1)", "O(1)"],
    ["Recurrent", "O(n·d²)", "O(n)", "O(n)"]
  ]
]

If NO tables exist on this page, return: []

Rules:
- Return ONLY the JSON array, no explanation, no markdown fences
- Use empty string "" for empty cells
- Preserve exact text including superscripts written as unicode or as text like n^2
"""


# ─────────────────────────────────────────────────────────────────
# NUL-BYTE SANITIZER  ← NEW
# ─────────────────────────────────────────────────────────────────

def _sanitize_text(text: str) -> str:
    """Remove NUL (0x00) bytes that PostgreSQL cannot store in text columns."""
    return text.replace("\x00", "")


# ─────────────────────────────────────────────────────────────────
# TABLE QUALITY FILTER
# ─────────────────────────────────────────────────────────────────

def _clean_cell(cell: Any) -> str:
    return str(cell or "").replace("\n", " ").replace("\r", " ").strip()


def _is_valid_table(rows: list[list[str]]) -> bool:
    if not rows or len(rows) < 2:
        return False
    cleaned = [[_clean_cell(c) for c in row] for row in rows]
    max_cols = max((len(r) for r in cleaned), default=0)
    if max_cols < 2:
        return False
    if not any(cleaned[0]):
        return False
    total  = sum(len(r) for r in cleaned)
    filled = sum(1 for r in cleaned for c in r if c)
    if total == 0 or filled / total < 0.35:
        return False
    # Reject 2-column prose layout artifacts
    if max_cols == 2:
        long_cells  = sum(1 for r in cleaned for c in r if len(c) > 60)
        total_cells = sum(len(r) for r in cleaned)
        if total_cells and long_cells / total_cells > 0.4:
            return False
    return True


# ─────────────────────────────────────────────────────────────────
# ELEMENT HELPERS
# ─────────────────────────────────────────────────────────────────

def _element_page(element: "Element") -> int:
    page = getattr(getattr(element, "metadata", None), "page_number", None)
    return int(page) if page is not None else 1


def _element_metadata(element: "Element") -> dict[str, Any]:
    meta: dict[str, Any] = {"category": element.category}
    if element.metadata is not None:
        if hasattr(element.metadata, "to_dict"):
            meta.update(element.metadata.to_dict())
        elif hasattr(element.metadata, "model_dump"):
            meta.update(element.metadata.model_dump())
    return meta


def _category_to_block_type(category: str | None) -> "BlockType | None":
    if not category:
        return "text"
    if category in SKIP_CATEGORIES:
        return None
    if category in HEADING_CATEGORIES:
        return "heading"
    return "text"


def _type_sort_key(block_type: "BlockType") -> int:
    return {"heading": 0, "text": 1, "table": 2, "image": 3}.get(block_type, 99)


# ─────────────────────────────────────────────────────────────────
# RATE LIMIT HELPERS
# ─────────────────────────────────────────────────────────────────

def _parse_retry_wait(error_str: str) -> float | None:
    match = re.search(r"try again in (\d+(?:\.\d+)?)(ms|s)", error_str)
    if match:
        value = float(match.group(1))
        unit  = match.group(2)
        return value / 1000.0 if unit == "ms" else value
    return None


def _is_rate_limit_error(error_str: str) -> bool:
    return "rate_limit_exceeded" in error_str or "429" in error_str


def _backoff_wait(attempt: int, base: float, cap: float) -> float:
    raw = base * (2 ** attempt) + random.uniform(0.0, 1.0)
    return min(raw, cap)


# ─────────────────────────────────────────────────────────────────
# VECTOR FIGURE HELPERS  (module-level, used by PDFParser)
# ─────────────────────────────────────────────────────────────────

def _get_figure_captions(page: fitz.Page) -> list[dict[str, Any]]:
    """
    Return figure caption blocks on a page as list of
    {"y_top": float, "y_bot": float, "text": str}.

    Matches "Figure N", "Fig. N", "FIGURE N", "FIG. N" (any case).
    Multi-line caption text is joined from all spans in the block.
    """
    captions: list[dict[str, Any]] = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        full_text = "".join(
            s["text"]
            for line in b.get("lines", [])
            for s in line.get("spans", [])
        ).strip()
        if FIGURE_CAPTION_RE.match(full_text):
            captions.append({
                "y_top": b["bbox"][1],
                "y_bot": b["bbox"][3],
                "text":  _sanitize_text(full_text),  # ← sanitize caption text
            })
    return captions


def _cluster_drawing_paths(
    drawings: list[dict],
    gap_threshold: float,
    min_paths: int,
    min_pt_w: float,
    min_pt_h: float,
) -> list[fitz.Rect]:
    """
    Group PDF drawing path records into spatial clusters.

    Each cluster whose bounding box meets the minimum size thresholds
    is returned as a fitz.Rect.  Sorting by top-Y then clustering on
    Y-gaps is robust for both single-column and two-column layouts.
    """
    raw: list[fitz.Rect] = []
    for d in drawings:
        r = fitz.Rect(d.get("rect") or fitz.Rect())
        if r.is_valid and not r.is_empty:
            raw.append(r)

    if len(raw) < min_paths:
        return []

    raw.sort(key=lambda r: (r.y0, r.x0))

    clusters: list[list[fitz.Rect]] = []
    current: list[fitz.Rect] = [raw[0]]

    for r in raw[1:]:
        cluster_bottom = max(cr.y1 for cr in current)
        if r.y0 - cluster_bottom > gap_threshold:
            clusters.append(current)
            current = [r]
        else:
            current.append(r)
    clusters.append(current)

    results: list[fitz.Rect] = []
    for cluster in clusters:
        if len(cluster) < min_paths:
            continue
        bbox = fitz.Rect(
            min(r.x0 for r in cluster),
            min(r.y0 for r in cluster),
            max(r.x1 for r in cluster),
            max(r.y1 for r in cluster),
        )
        if bbox.width >= min_pt_w and bbox.height >= min_pt_h:
            results.append(bbox)

    return results


def _render_clip_to_png(
    page: fitz.Page,
    clip: fitz.Rect,
    dpi: int,
) -> tuple[bytes, int, int]:
    """
    Render a rectangular clip of a PDF page to PNG bytes.
    Returns (png_bytes, pixel_width, pixel_height).
    """
    zoom   = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix    = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
    return pix.tobytes("png"), pix.width, pix.height


# ─────────────────────────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────────────────────────

class PDFParser:

    def __init__(
        self,
        *,
        openai_client: OpenAI | None = None,
        vision_model: str | None = None,
        image_output_dir: Path | str | None = None,
        vision_prompt: str = DEFAULT_VISION_PROMPT,
        unstructured_strategy: str | None = None,
        skip_vision: bool = True,
        compress_images: bool = True,
        jpeg_quality: int = JPEG_QUALITY,
        # Table extraction settings — UNCHANGED
        use_vision_tables: bool = True,
        table_render_dpi: int = TABLE_RENDER_DPI,
        # Rate limit / reliability settings
        pages_per_batch: int = PAGES_PER_BATCH,
        batch_pause_seconds: float = BATCH_PAUSE_SECONDS,
        max_retries_per_page: int = MAX_RETRIES_PER_PAGE,
        base_backoff_seconds: float = BASE_BACKOFF_SECONDS,
        max_backoff_seconds: float = MAX_BACKOFF_SECONDS,
        second_pass_cooldown: float = SECOND_PASS_COOLDOWN,
        # Vector figure extraction settings
        extract_vector_figures: bool = True,
        vector_fig_render_dpi: int = VECTOR_FIG_RENDER_DPI,
        vector_fig_cluster_gap: float = VECTOR_FIG_CLUSTER_GAP,
        vector_fig_min_paths: int = VECTOR_FIG_MIN_PATHS,
    ) -> None:
        self._client = openai_client or OpenAI(api_key=settings.openai_api_key)
        self.vision_model = vision_model or settings.openai_vision_model
        resolved_image_dir = image_output_dir or settings.image_output_dir
        self.image_output_dir = (
            Path(resolved_image_dir) if resolved_image_dir else None
        )
        self.vision_prompt          = vision_prompt
        self.unstructured_strategy  = unstructured_strategy or settings.unstructured_strategy
        self.skip_vision            = skip_vision
        self.compress_images        = compress_images
        self.jpeg_quality           = jpeg_quality
        # Table extraction — quality settings, unchanged
        self.use_vision_tables      = use_vision_tables
        self.table_render_dpi       = table_render_dpi
        # Rate limit / reliability
        self.pages_per_batch        = pages_per_batch
        self.batch_pause_seconds    = batch_pause_seconds
        self.max_retries_per_page   = max_retries_per_page
        self.base_backoff_seconds   = base_backoff_seconds
        self.max_backoff_seconds    = max_backoff_seconds
        self.second_pass_cooldown   = second_pass_cooldown
        # Vector figure extraction
        self.extract_vector_figures  = extract_vector_figures
        self.vector_fig_render_dpi   = vector_fig_render_dpi
        self.vector_fig_cluster_gap  = vector_fig_cluster_gap
        self.vector_fig_min_paths    = vector_fig_min_paths

    # ── PUBLIC ENTRY ──────────────────────────────────────────────

    def parse(
        self,
        pdf_path: str | Path,
        *,
        doc_id: str | None = None,
    ) -> list[ParsedBlock]:
        path = Path(pdf_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")

        resolved_doc_id = doc_id or path.stem
        blocks: list[ParsedBlock] = []
        blocks.extend(self._extract_text_and_headings(path, resolved_doc_id))
        blocks.extend(self._extract_tables(path, resolved_doc_id))
        blocks.extend(self._extract_images(path, resolved_doc_id))
        blocks.sort(key=lambda b: (b["page"], _type_sort_key(b["type"])))

        logger.info(
            "Parsed %d blocks from %s (text+heading=%d, table=%d, image=%d)",
            len(blocks), resolved_doc_id,
            sum(1 for b in blocks if b["type"] in ("text", "heading")),
            sum(1 for b in blocks if b["type"] == "table"),
            sum(1 for b in blocks if b["type"] == "image"),
        )
        return blocks

    # ── TEXT ──────────────────────────────────────────────────────

    def _extract_text_and_headings(self, pdf_path: Path, doc_id: str) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        try:
            from unstructured.partition.pdf import partition_pdf
            elements = partition_pdf(
                filename=str(pdf_path),
                strategy=self.unstructured_strategy,
                infer_table_structure=False,
            )
        except Exception:
            logger.exception("unstructured partition failed for %s", pdf_path)
            return blocks

        for element in elements:
            block_type = _category_to_block_type(element.category)
            if block_type is None:
                continue
            text = _sanitize_text((element.text or "").strip())  # ← sanitize
            if not text:
                continue
            blocks.append({
                "doc_id":   doc_id,
                "page":     _element_page(element),
                "type":     block_type,
                "content":  text,
                "metadata": _element_metadata(element),
            })
        return blocks

    # ── TABLES — Vision-first cascade ────────────────────────────

    def _extract_tables(self, pdf_path: Path, doc_id: str) -> list[ParsedBlock]:
        if self.use_vision_tables:
            blocks = self._extract_tables_vision(pdf_path, doc_id)
            if blocks:
                logger.info("Vision tables: %d extracted from %s", len(blocks), doc_id)
                return blocks
            logger.info("Vision found no tables in %s, trying vector methods", doc_id)

        blocks = self._try_camelot_lattice(pdf_path, doc_id)
        if blocks:
            logger.info("Camelot lattice: %d tables in %s", len(blocks), doc_id)
            return blocks

        blocks = self._try_pdfplumber_lines(pdf_path, doc_id)
        if blocks:
            logger.info("pdfplumber lines: %d tables in %s", len(blocks), doc_id)
            return blocks

        blocks = self._try_pdfplumber_default(pdf_path, doc_id)
        logger.info("pdfplumber default: %d tables in %s", len(blocks), doc_id)
        return blocks

    # ── Vision table extraction — guaranteed no skip ──────────────

    def _call_vision_api_with_retry(
        self,
        b64: str,
        page_number: int,
        doc_id: str,
    ) -> str | None:
        for attempt in range(self.max_retries_per_page):
            try:
                response = self._client.chat.completions.create(
                    model=self.vision_model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": TABLE_VISION_PROMPT},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": "high",
                            }},
                        ],
                    }],
                    max_tokens=4096,
                    timeout=60,
                )
                return (response.choices[0].message.content or "").strip()

            except Exception as e:
                err_str = str(e)

                if _is_rate_limit_error(err_str):
                    suggested = _parse_retry_wait(err_str)
                    if suggested is not None:
                        wait_time = min(suggested + 0.5, self.max_backoff_seconds)
                    else:
                        wait_time = _backoff_wait(
                            attempt, self.base_backoff_seconds, self.max_backoff_seconds
                        )

                    if attempt < self.max_retries_per_page - 1:
                        logger.warning(
                            "Rate limit on page %d of %s (attempt %d/%d). Waiting %.2fs...",
                            page_number, doc_id, attempt + 1, self.max_retries_per_page, wait_time,
                        )
                        time.sleep(wait_time)
                    else:
                        logger.warning(
                            "Rate limit retries exhausted for page %d of %s after %d attempts. "
                            "Will retry in second pass.",
                            page_number, doc_id, self.max_retries_per_page,
                        )
                        return None

                else:
                    logger.warning(
                        "Vision API error on page %d of %s (attempt %d/%d): %s",
                        page_number, doc_id, attempt + 1, self.max_retries_per_page, e,
                    )
                    if attempt < self.max_retries_per_page - 1:
                        wait_time = _backoff_wait(
                            attempt, self.base_backoff_seconds, self.max_backoff_seconds
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            "All %d attempts failed for page %d of %s. Will use per-page fallback.",
                            self.max_retries_per_page, page_number, doc_id,
                        )
                        return None

        return None

    def _render_page_b64(self, page: fitz.Page) -> str:
        zoom   = self.table_render_dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix    = page.get_pixmap(matrix=matrix, alpha=False)
        return base64.standard_b64encode(pix.tobytes("png")).decode("ascii")

    def _parse_vision_response(
        self,
        raw: str,
        page_number: int,
        doc_id: str,
    ) -> list[list[list[str]]] | None:
        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\n?```$",        "", raw, flags=re.MULTILINE)
        raw = raw.strip()

        if not raw or raw == "[]":
            return []

        try:
            tables = json.loads(raw)
            if not isinstance(tables, list):
                raise ValueError("Expected JSON array")
            return tables
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "JSON parse failed page %d of %s: %s — raw: %.200s",
                page_number, doc_id, e, raw,
            )
            return None

    def _tables_from_vision_response(
        self,
        tables_on_page: list,
        page_number: int,
        doc_id: str,
    ) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        for tbl_idx, table in enumerate(tables_on_page):
            if not isinstance(table, list) or not table:
                continue
            rows = [[_clean_cell(cell) for cell in row]
                    for row in table if isinstance(row, list)]
            if not _is_valid_table(rows):
                continue
            blocks.append({
                "doc_id":  doc_id,
                "page":    page_number,
                "type":    "table",
                "content": json.dumps(rows, ensure_ascii=False),
                "metadata": {
                    "table_index":       tbl_idx,
                    "extraction_method": "gpt4o_vision",
                    "render_dpi":        self.table_render_dpi,
                    "rows":              len(rows),
                    "cols":              len(rows[0]) if rows else 0,
                },
            })
        return blocks

    def _fallback_tables_for_page(
        self,
        pdf_path: Path,
        doc_id: str,
        page_number: int,
    ) -> list[ParsedBlock]:
        page_str = str(page_number)

        try:
            import camelot
            table_list = camelot.read_pdf(str(pdf_path), pages=page_str, flavor="lattice")
            blocks: list[ParsedBlock] = []
            for idx, tbl in enumerate(table_list):
                rows = [[_clean_cell(c) for c in row]
                        for row in tbl.df.fillna("").values.tolist()]
                if not _is_valid_table(rows):
                    continue
                blocks.append({
                    "doc_id":  doc_id,
                    "page":    page_number,
                    "type":    "table",
                    "content": json.dumps(rows, ensure_ascii=False),
                    "metadata": {
                        "table_index":       idx,
                        "extraction_method": "camelot_lattice_fallback",
                        "accuracy":          float(tbl.accuracy) if tbl.accuracy else None,
                        "rows":              len(rows),
                        "cols":              len(rows[0]) if rows else 0,
                    },
                })
            if blocks:
                logger.info(
                    "Per-page fallback (camelot): found %d tables on page %d of %s",
                    len(blocks), page_number, doc_id,
                )
                return blocks
        except Exception:
            pass

        try:
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[page_number - 1]
                settings_lines = {
                    "vertical_strategy":   "lines",
                    "horizontal_strategy": "lines",
                }
                blocks = []
                for idx, tbl in enumerate(page.extract_tables(settings_lines) or []):
                    rows = [[_clean_cell(c) for c in row] for row in tbl]
                    if not _is_valid_table(rows):
                        continue
                    blocks.append({
                        "doc_id":  doc_id,
                        "page":    page_number,
                        "type":    "table",
                        "content": json.dumps(rows, ensure_ascii=False),
                        "metadata": {
                            "table_index":       idx,
                            "extraction_method": "pdfplumber_lines_fallback",
                            "rows":              len(rows),
                            "cols":              len(rows[0]) if rows else 0,
                        },
                    })
                if blocks:
                    logger.info(
                        "Per-page fallback (pdfplumber lines): found %d tables on page %d of %s",
                        len(blocks), page_number, doc_id,
                    )
                    return blocks
        except Exception:
            pass

        try:
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[page_number - 1]
                blocks = []
                for idx, tbl in enumerate(page.extract_tables() or []):
                    rows = [[_clean_cell(c) for c in row] for row in tbl]
                    if not _is_valid_table(rows):
                        continue
                    blocks.append({
                        "doc_id":  doc_id,
                        "page":    page_number,
                        "type":    "table",
                        "content": json.dumps(rows, ensure_ascii=False),
                        "metadata": {
                            "table_index":       idx,
                            "extraction_method": "pdfplumber_default_fallback",
                            "rows":              len(rows),
                            "cols":              len(rows[0]) if rows else 0,
                        },
                    })
                if blocks:
                    logger.info(
                        "Per-page fallback (pdfplumber default): found %d tables on page %d of %s",
                        len(blocks), page_number, doc_id,
                    )
                    return blocks
        except Exception:
            pass

        logger.warning(
            "All extraction methods failed for page %d of %s. Page has no table blocks.",
            page_number, doc_id,
        )
        return []

    def _extract_tables_vision(self, pdf_path: Path, doc_id: str) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        failed_pages: list[tuple[int, str]] = []

        doc = fitz.open(pdf_path)
        try:
            for page_index in range(len(doc)):
                page        = doc[page_index]
                page_number = page_index + 1

                if page_index > 0 and page_index % self.pages_per_batch == 0:
                    logger.info(
                        "Proactive rate limit pause: sleeping %.1fs after page %d of %s",
                        self.batch_pause_seconds, page_number, doc_id,
                    )
                    time.sleep(self.batch_pause_seconds)

                b64 = self._render_page_b64(page)
                raw = self._call_vision_api_with_retry(b64, page_number, doc_id)

                if raw is None:
                    failed_pages.append((page_number, b64))
                    continue

                tables_on_page = self._parse_vision_response(raw, page_number, doc_id)
                if tables_on_page is None:
                    failed_pages.append((page_number, b64))
                    continue

                page_blocks = self._tables_from_vision_response(
                    tables_on_page, page_number, doc_id
                )
                blocks.extend(page_blocks)

                if page_blocks:
                    logger.info(
                        "Page %d of %s: vision found %d tables",
                        page_number, doc_id, len(page_blocks),
                    )

        finally:
            doc.close()

        if failed_pages:
            logger.info(
                "%d pages failed vision extraction in %s. "
                "Waiting %.1fs then retrying (second pass)...",
                len(failed_pages), doc_id, self.second_pass_cooldown,
            )
            time.sleep(self.second_pass_cooldown)

            still_failed: list[int] = []
            for page_number, b64 in failed_pages:
                logger.info("Second pass: retrying page %d of %s", page_number, doc_id)
                raw = self._call_vision_api_with_retry(b64, page_number, doc_id)

                if raw is None:
                    still_failed.append(page_number)
                    continue

                tables_on_page = self._parse_vision_response(raw, page_number, doc_id)
                if tables_on_page is None:
                    still_failed.append(page_number)
                    continue

                page_blocks = self._tables_from_vision_response(
                    tables_on_page, page_number, doc_id
                )
                blocks.extend(page_blocks)

                if page_blocks:
                    logger.info(
                        "Second pass success: page %d of %s yielded %d tables",
                        page_number, doc_id, len(page_blocks),
                    )

            if still_failed:
                logger.warning(
                    "%d pages still failed after second pass in %s: %s. "
                    "Using per-page fallback extraction (Camelot/pdfplumber).",
                    len(still_failed), doc_id, still_failed,
                )
                for page_number in still_failed:
                    fallback_blocks = self._fallback_tables_for_page(
                        pdf_path, doc_id, page_number
                    )
                    blocks.extend(fallback_blocks)

        return blocks

    # ── Document-level vector fallbacks ──────────────────────────

    def _try_camelot_lattice(self, pdf_path: Path, doc_id: str) -> list[ParsedBlock]:
        try:
            import camelot
            table_list = camelot.read_pdf(str(pdf_path), pages="all", flavor="lattice")
        except Exception:
            logger.debug("Camelot lattice failed for %s", pdf_path.name)
            return []

        blocks: list[ParsedBlock] = []
        for idx, tbl in enumerate(table_list):
            rows = [[_clean_cell(c) for c in row]
                    for row in tbl.df.fillna("").values.tolist()]
            if not _is_valid_table(rows):
                continue
            blocks.append({
                "doc_id":  doc_id,
                "page":    int(tbl.page),
                "type":    "table",
                "content": json.dumps(rows, ensure_ascii=False),
                "metadata": {
                    "table_index":       idx,
                    "extraction_method": "camelot_lattice",
                    "accuracy":          float(tbl.accuracy) if tbl.accuracy else None,
                    "rows":              len(rows),
                    "cols":              len(rows[0]) if rows else 0,
                },
            })
        return blocks

    def _try_pdfplumber_lines(self, pdf_path: Path, doc_id: str) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        settings_lines = {
            "vertical_strategy":   "lines",
            "horizontal_strategy": "lines",
        }
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                for idx, tbl in enumerate(page.extract_tables(settings_lines) or []):
                    rows = [[_clean_cell(c) for c in row] for row in tbl]
                    if not _is_valid_table(rows):
                        continue
                    blocks.append({
                        "doc_id":  doc_id,
                        "page":    page_num,
                        "type":    "table",
                        "content": json.dumps(rows, ensure_ascii=False),
                        "metadata": {
                            "table_index":       idx,
                            "extraction_method": "pdfplumber_lines",
                            "rows":              len(rows),
                            "cols":              len(rows[0]) if rows else 0,
                        },
                    })
        return blocks

    def _try_pdfplumber_default(self, pdf_path: Path, doc_id: str) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                for idx, tbl in enumerate(page.extract_tables() or []):
                    rows = [[_clean_cell(c) for c in row] for row in tbl]
                    if not _is_valid_table(rows):
                        continue
                    blocks.append({
                        "doc_id":  doc_id,
                        "page":    page_num,
                        "type":    "table",
                        "content": json.dumps(rows, ensure_ascii=False),
                        "metadata": {
                            "table_index":       idx,
                            "extraction_method": "pdfplumber_default",
                            "rows":              len(rows),
                            "cols":              len(rows[0]) if rows else 0,
                        },
                    })
        return blocks

    # ── IMAGES ───────────────────────────────────────────────────

    def _extract_images(self, pdf_path: Path, doc_id: str) -> list[ParsedBlock]:
        output_dir = self._resolve_image_dir(pdf_path, doc_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        blocks: list[ParsedBlock] = []
        blocks.extend(self._extract_raster_images(pdf_path, doc_id, output_dir))
        if self.extract_vector_figures:
            blocks.extend(self._extract_vector_figures(pdf_path, doc_id, output_dir))
        return blocks

    def _extract_raster_images(
        self,
        pdf_path: Path,
        doc_id: str,
        output_dir: Path,
    ) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        doc = fitz.open(pdf_path)
        try:
            for page_index in range(len(doc)):
                page        = doc[page_index]
                page_number = page_index + 1
                for image_index, image_info in enumerate(page.get_images(full=True)):
                    xref = image_info[0]
                    try:
                        pix = fitz.Pixmap(doc, xref)

                        if pix.colorspace is None:
                            continue

                        if pix.alpha:
                            pix = fitz.Pixmap(pix, 0)

                        if pix.colorspace and pix.colorspace.n > 3:
                            pix = fitz.Pixmap(fitz.csRGB, pix)

                        width, height = pix.width, pix.height
                        if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                            continue

                        image_bytes = pix.tobytes("png")

                    except Exception:
                        logger.warning(
                            "Failed raster image xref=%s page=%s",
                            xref, page_number, exc_info=True,
                        )
                        continue

                    image_path = (
                        output_dir
                        / f"{doc_id}_p{page_number}_raster{image_index}.png"
                    )
                    image_path.write_bytes(image_bytes)

                    description = self._get_vision_description(
                        image_path, page_number, doc_id
                    )

                    if self.compress_images:
                        image_path = self._compress_to_jpeg(image_path)

                    blocks.append({
                        "doc_id":   doc_id,
                        "page":     page_number,
                        "type":     "image",
                        "content":  description,  # already sanitized via _get_vision_description
                        "metadata": {
                            "image_path":        str(image_path),
                            "extraction_method": "raster_xobject",
                            "image_index":       image_index,
                            "xref":              xref,
                            "width":             width,
                            "height":            height,
                            "vision_model":      self.vision_model,
                            "compressed":        self.compress_images,
                        },
                    })
        finally:
            doc.close()

        logger.info(
            "Raster images: %d extracted from %s", len(blocks), doc_id
        )
        return blocks

    def _extract_vector_figures(
        self,
        pdf_path: Path,
        doc_id: str,
        output_dir: Path,
    ) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        doc = fitz.open(pdf_path)
        try:
            for page_index in range(len(doc)):
                page        = doc[page_index]
                page_number = page_index + 1
                drawings    = page.get_drawings()

                if not drawings:
                    continue

                drawing_clusters = _cluster_drawing_paths(
                    drawings,
                    gap_threshold=self.vector_fig_cluster_gap,
                    min_paths=self.vector_fig_min_paths,
                    min_pt_w=VECTOR_FIG_MIN_PT_WIDTH,
                    min_pt_h=VECTOR_FIG_MIN_PT_HEIGHT,
                )

                if not drawing_clusters:
                    continue

                captions       = _get_figure_captions(page)  # already sanitizes cap["text"]
                claimed: set[int] = set()

                # ── Pass 1: Caption-anchored figures ──────────────────
                for cap in captions:
                    relevant_indices = [
                        i for i, c in enumerate(drawing_clusters)
                        if c.y1 <= cap["y_top"] + 10
                    ]
                    if not relevant_indices:
                        continue

                    relevant = [drawing_clusters[i] for i in relevant_indices]
                    fig_rect = fitz.Rect(
                        min(c.x0 for c in relevant) - VECTOR_FIG_PADDING,
                        min(c.y0 for c in relevant) - VECTOR_FIG_PADDING,
                        max(c.x1 for c in relevant) + VECTOR_FIG_PADDING,
                        cap["y_bot"] + VECTOR_FIG_CAPTION_MARGIN,
                    ) & page.rect

                    png_bytes, w, h = _render_clip_to_png(
                        page, fig_rect, self.vector_fig_render_dpi
                    )

                    if w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT:
                        continue

                    claimed.update(relevant_indices)

                    image_path = (
                        output_dir
                        / f"{doc_id}_p{page_number}_vecfig_{len(blocks)}.png"
                    )
                    image_path.write_bytes(png_bytes)

                    seed_description = cap["text"]  # already sanitized
                    description = self._get_vision_description(
                        image_path, page_number, doc_id,
                        fallback=seed_description,
                    )

                    if self.compress_images:
                        image_path = self._compress_to_jpeg(image_path)

                    blocks.append({
                        "doc_id":   doc_id,
                        "page":     page_number,
                        "type":     "image",
                        "content":  description,  # already sanitized via _get_vision_description
                        "metadata": {
                            "image_path":        str(image_path),
                            "extraction_method": "vector_figure_captioned",
                            "figure_caption":    cap["text"],
                            "clip_rect":         list(fig_rect),
                            "width":             w,
                            "height":            h,
                            "vision_model":      self.vision_model,
                            "compressed":        self.compress_images,
                        },
                    })
                    logger.info(
                        "Vector figure (captioned) page %d of %s: '%s' [%dx%d px]",
                        page_number, doc_id, cap["text"][:60], w, h,
                    )

                # ── Pass 2: Uncaptioned standalone drawings ────────────
                for i, cluster in enumerate(drawing_clusters):
                    if i in claimed:
                        continue
                    if (cluster.width < VECTOR_FIG_UNCAP_MIN_W
                            or cluster.height < VECTOR_FIG_UNCAP_MIN_H):
                        continue

                    fig_rect = fitz.Rect(
                        cluster.x0 - VECTOR_FIG_PADDING,
                        cluster.y0 - VECTOR_FIG_PADDING,
                        cluster.x1 + VECTOR_FIG_PADDING,
                        cluster.y1 + VECTOR_FIG_PADDING,
                    ) & page.rect

                    png_bytes, w, h = _render_clip_to_png(
                        page, fig_rect, self.vector_fig_render_dpi
                    )

                    if w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT:
                        continue

                    image_path = (
                        output_dir
                        / f"{doc_id}_p{page_number}_vecfig_{len(blocks)}.png"
                    )
                    image_path.write_bytes(png_bytes)

                    fallback = f"Diagram on page {page_number} of {doc_id}"
                    description = self._get_vision_description(
                        image_path, page_number, doc_id,
                        fallback=fallback,
                    )

                    if self.compress_images:
                        image_path = self._compress_to_jpeg(image_path)

                    blocks.append({
                        "doc_id":   doc_id,
                        "page":     page_number,
                        "type":     "image",
                        "content":  description,  # already sanitized via _get_vision_description
                        "metadata": {
                            "image_path":        str(image_path),
                            "extraction_method": "vector_figure_uncaptioned",
                            "clip_rect":         list(fig_rect),
                            "width":             w,
                            "height":            h,
                            "vision_model":      self.vision_model,
                            "compressed":        self.compress_images,
                        },
                    })
                    logger.info(
                        "Vector figure (uncaptioned) page %d of %s: cluster %d [%dx%d px]",
                        page_number, doc_id, i, w, h,
                    )

        finally:
            doc.close()

        logger.info(
            "Vector figures: %d extracted from %s", len(blocks), doc_id
        )
        return blocks

    # ── Shared image helpers ──────────────────────────────────────

    def _get_vision_description(
        self,
        image_path: Path,
        page_number: int,
        doc_id: str,
        fallback: str | None = None,
    ) -> str:
        if self.skip_vision:
            raw = fallback or f"Image on page {page_number} of {doc_id}"
            return _sanitize_text(raw)  # ← sanitize
        try:
            return _sanitize_text(self._describe_image(image_path))  # ← sanitize
        except Exception as e:
            logger.warning("Vision failed %s: %s", image_path.name, e)
            raw = fallback or f"Image on page {page_number} of {doc_id}"
            return _sanitize_text(raw)  # ← sanitize

    def _compress_to_jpeg(self, image_path: Path) -> Path:
        try:
            jpeg_path = image_path.with_suffix(".jpg")
            pil_img   = Image.open(image_path)
            if pil_img.mode in ("RGBA", "LA", "P"):
                pil_img = pil_img.convert("RGB")
            pil_img.save(
                jpeg_path, format="JPEG",
                quality=self.jpeg_quality, optimize=True,
            )
            image_path.unlink()
            return jpeg_path
        except Exception as e:
            logger.warning("Compression failed %s: %s", image_path.name, e)
            return image_path

    def _resolve_image_dir(self, pdf_path: Path, doc_id: str) -> Path:
        if self.image_output_dir is not None:
            return self.image_output_dir / doc_id
        return Path("data/images") / doc_id

    def _describe_image(self, image_path: Path) -> str:
        b64 = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
        response = self._client.chat.completions.create(
            model=self.vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.vision_prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            max_tokens=512,
            timeout=30,
        )
        return (response.choices[0].message.content or "").strip()


# ─────────────────────────────────────────────────────────────────
# PUBLIC HELPERS
# ─────────────────────────────────────────────────────────────────

def parse_pdf(
    pdf_path: str | Path,
    *,
    doc_id: str | None = None,
    **parser_kwargs: Any,
) -> list[ParsedBlock]:
    return PDFParser(**parser_kwargs).parse(pdf_path, doc_id=doc_id)


def blocks_to_json(blocks: list[ParsedBlock], *, indent: int | None = 2) -> str:
    return json.dumps(blocks, ensure_ascii=False, indent=indent)
