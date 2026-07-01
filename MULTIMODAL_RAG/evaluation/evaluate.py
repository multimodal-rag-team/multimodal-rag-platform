"""
evaluate.py — Evaluation harness for the Multimodal RAG Platform.

Runs every question in golden_dataset.json against the live API,
scores each answer, and writes a Markdown + JSON comparative report.

Usage:
    python evaluation/evaluate.py                          # all questions
    python evaluation/evaluate.py --doc-id Attention_Is_All_You_Need
    python evaluation/evaluate.py --limit 10 --top-k 5
    python evaluation/evaluate.py --dataset path/to/other.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_DATASET = Path(__file__).parent / "golden_dataset.json"
DEFAULT_API_BASE = "http://localhost:8000"
REPORT_DIR = Path(__file__).parent.parent / "evaluation" / "reports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Scoring helpers ────────────────────────────────────────────────────────────

def _token_overlap_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between two strings (recall-precision harmonic mean)."""
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _route_correct(predicted_route: str, expected_route: str) -> bool:
    return predicted_route.strip().lower() == expected_route.strip().lower()


def _llm_judge(
    client: OpenAI,
    question: str,
    expected: str,
    actual: str,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """
    Ask an LLM to judge whether `actual` correctly answers `question`
    given that `expected` is the gold answer.

    Returns: {"score": 0-5, "reasoning": str}
    """
    prompt = f"""You are an expert evaluator for a RAG system. Score the model's answer from 0 to 5.

Scoring rubric:
5 — Fully correct, complete, matches the gold answer in all key facts.
4 — Mostly correct, minor omission or slight imprecision.
3 — Partially correct, captures the main idea but misses important details.
2 — Partially related but with significant errors or gaps.
1 — Barely relevant, mostly wrong.
0 — Completely wrong or no answer provided.

Question: {question}
Gold answer: {expected}
Model answer: {actual}

Reply ONLY with valid JSON in this format:
{{"score": <0-5>, "reasoning": "<one sentence>"}}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        raw = (response.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        return {"score": int(result.get("score", 0)), "reasoning": result.get("reasoning", "")}
    except Exception as exc:
        logger.warning("LLM judge failed: %s", exc)
        return {"score": -1, "reasoning": f"Judge error: {exc}"}


# ── API client ────────────────────────────────────────────────────────────────

def query_api(
    question: str,
    doc_id: str | None,
    top_k: int,
    api_base: str,
    timeout: int = 60,
) -> dict[str, Any]:
    """POST /query and return the parsed response dict."""
    payload: dict[str, Any] = {"query": question, "top_k": top_k}
    if doc_id:
        payload["doc_id"] = doc_id
    url = f"{api_base.rstrip('/')}/query"
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ── Main evaluation loop ───────────────────────────────────────────────────────

def run_evaluation(
    dataset_path: Path,
    api_base: str,
    top_k: int,
    doc_id_filter: str | None,
    limit: int | None,
    use_llm_judge: bool,
    openai_client: OpenAI | None,
    judge_model: str,
) -> dict[str, Any]:
    dataset: list[dict[str, Any]] = json.loads(dataset_path.read_text(encoding="utf-8"))

    if doc_id_filter:
        dataset = [q for q in dataset if q.get("doc_id") == doc_id_filter]
    if limit:
        dataset = dataset[:limit]

    logger.info("Evaluating %d questions against %s", len(dataset), api_base)

    results: list[dict[str, Any]] = []
    failed = 0

    for idx, item in enumerate(dataset, start=1):
        qid = item.get("id", f"q{idx}")
        question = item["question"]
        expected_answer = item["expected_answer"]
        expected_route = item.get("expected_route", "")
        item_doc_id = item.get("doc_id") if not doc_id_filter else doc_id_filter

        logger.info("[%d/%d] %s — %s", idx, len(dataset), qid, question[:80])

        t0 = time.monotonic()
        try:
            api_response = query_api(question, item_doc_id, top_k, api_base)
        except Exception as exc:
            logger.warning("API call failed for %s: %s", qid, exc)
            failed += 1
            results.append({
                "id": qid,
                "doc_id": item_doc_id,
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": "",
                "expected_route": expected_route,
                "actual_route": "",
                "route_correct": False,
                "token_f1": 0.0,
                "llm_score": -1,
                "llm_reasoning": f"API error: {exc}",
                "latency_s": round(time.monotonic() - t0, 3),
                "sources_returned": 0,
                "error": str(exc),
            })
            continue

        latency = round(time.monotonic() - t0, 3)
        actual_answer = api_response.get("answer", "")
        actual_route = api_response.get("route", "")
        sources = api_response.get("sources", [])
        tables = api_response.get("tables", [])
        images = api_response.get("images", [])

        token_f1 = _token_overlap_f1(actual_answer, expected_answer)
        route_ok = _route_correct(actual_route, expected_route) if expected_route else None

        llm_score = -1
        llm_reasoning = "LLM judge disabled"
        if use_llm_judge and openai_client:
            judge_result = _llm_judge(
                openai_client, question, expected_answer, actual_answer, judge_model
            )
            llm_score = judge_result["score"]
            llm_reasoning = judge_result["reasoning"]

        results.append({
            "id": qid,
            "doc_id": item_doc_id,
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "expected_route": expected_route,
            "actual_route": actual_route,
            "route_correct": route_ok,
            "token_f1": round(token_f1, 4),
            "llm_score": llm_score,
            "llm_reasoning": llm_reasoning,
            "latency_s": latency,
            "sources_returned": len(sources) + len(tables) + len(images),
            "error": None,
        })

        logger.info(
            "  → route=%s (%s) | F1=%.3f | LLM=%s | %.2fs",
            actual_route,
            "✓" if route_ok else ("✗" if route_ok is False else "—"),
            token_f1,
            llm_score if llm_score >= 0 else "—",
            latency,
        )

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    scored = [r for r in results if r["error"] is None]
    route_items = [r for r in scored if r["route_correct"] is not None]
    llm_items = [r for r in scored if r["llm_score"] >= 0]

    avg_f1 = statistics.mean(r["token_f1"] for r in scored) if scored else 0.0
    route_accuracy = (
        sum(1 for r in route_items if r["route_correct"]) / len(route_items)
        if route_items else None
    )
    avg_llm = (
        statistics.mean(r["llm_score"] for r in llm_items) / 5.0
        if llm_items else None
    )
    avg_latency = statistics.mean(r["latency_s"] for r in scored) if scored else 0.0
    p95_latency = (
        sorted(r["latency_s"] for r in scored)[int(len(scored) * 0.95) - 1]
        if len(scored) >= 2 else avg_latency
    )

    summary = {
        "total_questions": len(dataset),
        "answered": len(scored),
        "failed_api_calls": failed,
        "avg_token_f1": round(avg_f1, 4),
        "route_accuracy": round(route_accuracy, 4) if route_accuracy is not None else None,
        "avg_llm_score_normalized": round(avg_llm, 4) if avg_llm is not None else None,
        "avg_latency_s": round(avg_latency, 3),
        "p95_latency_s": round(p95_latency, 3),
    }

    return {"summary": summary, "results": results}


# ── Report writers ─────────────────────────────────────────────────────────────

def write_json_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("JSON report → %s", path)


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    s = report["summary"]
    results = report["results"]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    route_acc_str = f"{s['route_accuracy']:.4f}" if s['route_accuracy'] is not None else "N/A"
    llm_score_str = f"{s['avg_llm_score_normalized']:.4f}" if s['avg_llm_score_normalized'] is not None else "N/A"

    lines: list[str] = [
        f"# Multimodal RAG Evaluation Report",
        f"",
        f"**Generated:** {ts}",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total questions | {s['total_questions']} |",
        f"| Answered successfully | {s['answered']} |",
        f"| Failed API calls | {s['failed_api_calls']} |",
        f"| Avg Token F1 | {s['avg_token_f1']:.4f} |",
        f"| Route Accuracy | {route_acc_str} |",
        f"| Avg LLM Score (0–1) | {llm_score_str} |",
        f"| Avg Latency | {s['avg_latency_s']:.3f}s |",
        f"| P95 Latency | {s['p95_latency_s']:.3f}s |",
        f"",
        f"## Per-Question Results",
        f"",
        f"| ID | Doc | Question (truncated) | Route ✓/✗ | F1 | LLM | Latency |",
        f"|----|-----|----------------------|------------|-----|-----|---------|",
    ]

    for r in results:
        q_short = r["question"][:60].replace("|", "∣")
        route_sym = (
            "✓" if r["route_correct"] is True
            else ("✗" if r["route_correct"] is False else "—")
        )
        route_cell = f"{r['actual_route']} {route_sym}"
        f1_cell = f"{r['token_f1']:.3f}"
        llm_cell = str(r["llm_score"]) if r["llm_score"] >= 0 else "—"
        lat_cell = f"{r['latency_s']}s"
        doc_short = (r["doc_id"] or "")[:30]
        lines.append(
            f"| {r['id']} | {doc_short} | {q_short} | {route_cell} | {f1_cell} | {llm_cell} | {lat_cell} |"
        )

    lines += [
        f"",
        f"## Failure Cases",
        f"",
    ]

    failures = [r for r in results if r.get("error")]
    if failures:
        for r in failures:
            lines.append(f"- **{r['id']}**: {r['error']}")
    else:
        lines.append("_No failures._")

    lines += [
        f"",
        f"## Low-Scoring Answers (Token F1 < 0.2)",
        f"",
    ]

    low_score = [r for r in results if r["token_f1"] < 0.2 and not r.get("error")]
    if low_score:
        for r in low_score:
            lines += [
                f"### {r['id']}",
                f"**Question:** {r['question']}",
                f"",
                f"**Expected:** {r['expected_answer']}",
                f"",
                f"**Actual:** {r['actual_answer'] or '_No answer_'}",
                f"",
                f"**LLM reasoning:** {r['llm_reasoning']}",
                f"",
            ]
    else:
        lines.append("_All answered questions scored Token F1 ≥ 0.2._")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown report → %s", path)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the Multimodal RAG API against a golden dataset.")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to golden_dataset.json")
    p.add_argument("--api-base", default=DEFAULT_API_BASE, help="Base URL of the running API (default: http://localhost:8000)")
    p.add_argument("--top-k", type=int, default=5, help="top_k passed to /query (default: 5)")
    p.add_argument("--doc-id", default=None, help="Filter to a single doc_id")
    p.add_argument("--limit", type=int, default=None, help="Run only the first N questions")
    p.add_argument("--llm-judge", action="store_true", help="Use GPT-4o-mini to score each answer (requires OPENAI_API_KEY)")
    p.add_argument("--judge-model", default="gpt-4o-mini", help="OpenAI model to use as LLM judge")
    p.add_argument("--output-dir", type=Path, default=REPORT_DIR, help="Directory to write reports")
    p.add_argument("--report-name", default=None, help="Custom report filename stem (default: timestamp)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset.exists():
        logger.error("Dataset not found: %s", args.dataset)
        sys.exit(1)

    openai_client: OpenAI | None = None
    if args.llm_judge:
        try:
            # Settings import works whether run from project root or evaluation/
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from config import settings  # type: ignore
            openai_client = OpenAI(api_key=settings.openai_api_key)
            logger.info("LLM judge enabled with model: %s", args.judge_model)
        except Exception as exc:
            logger.warning("Could not load OpenAI client for LLM judge: %s — disabling judge.", exc)
            args.llm_judge = False

    report = run_evaluation(
        dataset_path=args.dataset,
        api_base=args.api_base,
        top_k=args.top_k,
        doc_id_filter=args.doc_id,
        limit=args.limit,
        use_llm_judge=args.llm_judge,
        openai_client=openai_client,
        judge_model=args.judge_model,
    )

    stem = args.report_name or datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    write_json_report(report, args.output_dir / f"{stem}.json")
    write_markdown_report(report, args.output_dir / f"{stem}.md")

    s = report["summary"]
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Questions:      {s['answered']}/{s['total_questions']} answered")
    print(f"  Token F1:       {s['avg_token_f1']:.4f}")
    if s["route_accuracy"] is not None:
        print(f"  Route Accuracy: {s['route_accuracy']:.4f}")
    if s["avg_llm_score_normalized"] is not None:
        print(f"  LLM Score:      {s['avg_llm_score_normalized']:.4f} (0–1)")
    print(f"  Avg Latency:    {s['avg_latency_s']:.3f}s")
    print(f"  P95 Latency:    {s['p95_latency_s']:.3f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()