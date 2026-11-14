"""Run the Day 06 data-ingestion and retrieval benchmark end to end."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import unicodedata
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pypdf import PdfReader

from .chunking import by_paragraph, fixed_size
from .embeddings import DEFAULT_MODEL_NAME, MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .models import Chunk, DocumentPage
from .store import PgVectorStore

# pypdf emits one warning per Type1 font when fontTools is not installed. Its
# text extraction still works, and the repeated warnings would hide pipeline
# output, so keep only real errors from that logger.
logging.getLogger("pypdf._font").setLevel(logging.ERROR)

STARTER_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = STARTER_DIR.parent
DEFAULT_CORPUS = STARTER_DIR / "data" / "data.pdf"
DEFAULT_DOCUMENT_ID = "international-debt-report-2026"
DEFAULT_OUTPUT_DIR = STARTER_DIR / "results"

MIN_CONTENT_PAGES = 30
MIN_CONTENT_CHARS = 200
MANUAL_COHERENCE_POSITIONS = 10
MANUAL_TEST_QUESTIONS = 5

# Reading-order body of this corpus (Part 1 + Part 2). Front matter, the
# appendix/glossary and the cover are deliberately outside the stored corpus.
CONTENT_PAGE_START = 25
CONTENT_PAGE_END = 263

# Pages inside the range that are not useful content and must not be counted or
# indexed. The GUIDE requires dropping blank pages, part dividers and pages that
# contain only references/bibliography.
BLANK_PAGES = {26, 106}
DIVIDER_PAGES = {25, 135}
REFERENCE_ONLY_PAGES = {105, 134}
EXCLUDED_PAGES = BLANK_PAGES | DIVIDER_PAGES | REFERENCE_ONLY_PAGES

# These top-level outline groups are indexes of boxes/figures/tables, not the
# reading order of the document. They must not replace the current chapter.
_INDEX_GROUP_TITLES = {"boxes", "figures", "tables"}
_REGION_PROFILE_TITLES = {
    "east asia and pacific",
    "europe and central asia",
    "latin america and the caribbean",
    "middle east, north africa, afghanistan, and pakistan",
    "south asia",
    "sub-saharan africa",
}

_WHITESPACE_RE = re.compile(r"\s+")
_ELLIPSIS_RE = re.compile(r"\s*(?:\.{3,}|…|\[\.{3,}\]|\[…\])\s*")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_QUOTE_TRANSLATION = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "–": "-",
        "—": "-",
        "−": "-",
    }
)


# --------------------------------------------------------------------- helpers
def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _norm_text(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    text = text.translate(_QUOTE_TRANSLATION)
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def _evidence_in_text(text: str, quote: str) -> bool:
    """Check every non-ellipsis fragment of a gold quote against page text."""
    page = _norm_text(text)
    normalized_quote = _norm_text(quote)
    if not page or not normalized_quote:
        return False
    fragments = [
        fragment.strip()
        for fragment in _ELLIPSIS_RE.split(normalized_quote)
        if fragment.strip()
    ]
    return bool(fragments) and all(fragment in page for fragment in fragments)


def _value_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    try:
        return int(str(left).strip()) == int(str(right).strip())
    except (TypeError, ValueError):
        return str(left).strip().casefold() == str(right).strip().casefold()


def _matches_filter(record: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if key not in record:
            return False
        if not _value_equal(record[key], expected):
            return False
    return True


_GOLD_SOURCE_KEYS = (
    "gold_source_id",
    "gold_source_ids",
    "gold_source",
    "gold_sources",
    "gold_document_id",
    "gold_document_ids",
    "source_ids",
)
_GOLD_PAGE_KEYS = (
    "gold_page",
    "gold_pages",
    "gold_page_number",
    "gold_page_numbers",
)
_GOLD_EVIDENCE_KEYS = (
    "evidence_quote",
    "evidence_quotes",
    "gold_evidence",
    "gold_evidence_quotes",
    "alternative_evidence",
    "acceptable_evidence",
)


def _gold_values(question: dict[str, Any], keys: tuple[str, ...]) -> list[Any]:
    values: list[Any] = []
    for key in keys:
        value = question.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            values.extend(item for item in value if item is not None)
        else:
            values.append(value)
    return values


def _gold_quotes(question: dict[str, Any]) -> list[str]:
    quotes: list[str] = []
    for item in _gold_values(question, _GOLD_EVIDENCE_KEYS):
        if isinstance(item, dict):
            for key in ("quote", "text", "evidence_quote", "evidence"):
                if item.get(key):
                    quotes.append(str(item[key]))
                    break
        else:
            quotes.append(str(item))
    return [quote for quote in quotes if quote.strip()]


def _trace_filter_compliance(trace: dict[str, Any],
                             filters: dict[str, Any]) -> bool:
    hits = trace.get("hits") or []
    if not hits:
        return False
    for hit in hits:
        record = {**hit.get("metadata", {}), **{
            key: value for key, value in hit.items() if key != "metadata"
        }}
        if not _matches_filter(record, filters):
            return False
    return True


def _source_for(corpus: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    return _project_path(corpus)


def _project_path(path: Path) -> str:
    """Format a path relative to the repository, never as a machine path."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_DIR.resolve()).as_posix()
    except ValueError:
        # Keep explicitly supplied external inputs portable in logs/artifacts.
        # The relative path still records where the input sits from the repo.
        return Path(os.path.relpath(resolved, REPO_DIR.resolve())).as_posix()


def _redact_database_url(url: str) -> str:
    """Hide the password before writing the URL into a result file."""
    return re.sub(r"(://[^:/@]+:)[^@]+(@)", r"\1***\2", url)


# --------------------------------------------------------------- corpus loading
def _outline_entries(pdf_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Flatten the PDF outline, retaining parent titles and physical pages."""
    reader = PdfReader(str(pdf_path))
    metadata = reader.metadata or {}
    entries: list[dict[str, Any]] = []

    def walk(items: list[Any], depth: int, parent: str | None) -> None:
        last_parent = parent
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1, last_parent)
                continue
            title = _WHITESPACE_RE.sub(" ", str(getattr(item, "title", ""))).strip()
            if not title:
                continue
            try:
                page = reader.get_destination_page_number(item) + 1
            except Exception:
                continue
            entries.append(
                {
                    "title": title,
                    "page": page,
                    "depth": depth,
                    "parent": parent,
                }
            )
            last_parent = title

    try:
        walk(reader.outline, 0, None)
    except Exception:
        # A malformed outline must not prevent ingestion; metadata enrichment
        # will simply fall back to the front-matter bucket.
        entries = []
    return entries, dict(metadata)


def _first_page(entries: list[dict[str, Any]], predicate) -> int | None:
    for entry in sorted(entries, key=lambda item: item["page"]):
        if predicate(entry["title"].casefold()):
            return int(entry["page"])
    return None


def _last_entry(entries: list[dict[str, Any]], page: int, predicate=None,
                minimum_page: int | None = None) -> dict[str, Any] | None:
    candidates = [
        entry
        for entry in entries
        if entry["page"] <= page
        and (minimum_page is None or entry["page"] >= minimum_page)
        and (predicate is None or predicate(entry))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item["page"], item["depth"]))


def _enrich_pages(pdf_path: Path, pages: list[DocumentPage]) -> list[DocumentPage]:
    """Attach document/part/section metadata derived from the PDF outline."""
    entries, pdf_metadata = _outline_entries(pdf_path)
    top_level = [
        entry
        for entry in entries
        if entry["depth"] == 0
        and entry["title"].casefold() not in _INDEX_GROUP_TITLES
    ]

    chapter1_page = _first_page(top_level, lambda title: title.startswith("chapter 1."))
    part2_page = _first_page(top_level, lambda title: title.startswith("part 2"))
    appendix_page = _first_page(top_level, lambda title: title == "appendix")

    title = str(pdf_metadata.get("/Title") or "").strip()
    year_match = _YEAR_RE.search(title)
    year = int(year_match.group(0)) if year_match else None
    document_type = "report" if "report" in title.casefold() else "document"

    def part2_child(page: int) -> dict[str, Any] | None:
        if part2_page is None or page < part2_page:
            return None
        if appendix_page is not None and page >= appendix_page:
            return None
        return _last_entry(
            entries,
            page,
            predicate=lambda entry: bool(
                entry["parent"] and entry["parent"].casefold().startswith("part 2")
            ),
            minimum_page=part2_page,
        )

    enriched: list[DocumentPage] = []
    for source_page in pages:
        page_number = source_page.page
        metadata = dict(source_page.metadata)
        metadata.update(
            {
                "pdf_title": title or metadata.get("pdf_title", ""),
                "total_pages": metadata.get("total_pages", len(pages)),
                "document_type": document_type,
            }
        )
        if year is not None:
            metadata["year"] = year

        if part2_page is not None and page_number >= part2_page and (
            appendix_page is None or page_number < appendix_page
        ):
            metadata["part"] = "Part 2. Aggregate and Country Profiles"
            child = part2_child(page_number)
            if child is not None:
                profile_title = child["title"]
                metadata["profile"] = profile_title
                metadata["section"] = profile_title
                if profile_title.casefold() in _REGION_PROFILE_TITLES:
                    metadata["profile_type"] = "regional_profile"
                    metadata["region"] = profile_title
                elif profile_title.casefold().startswith("all low-"):
                    metadata["profile_type"] = "aggregate_profile"
                else:
                    metadata["profile_type"] = "country_profile"
                    metadata["country"] = profile_title
            else:
                metadata["section"] = metadata["part"]
        elif appendix_page is not None and page_number >= appendix_page:
            metadata["part"] = "Appendix"
            division = _last_entry(
                top_level, page_number, minimum_page=appendix_page
            )
            metadata["section"] = (
                division["title"] if division is not None else "Appendix"
            )
        elif chapter1_page is not None and page_number >= chapter1_page:
            metadata["part"] = "Part 1. Overview"
            chapter = _last_entry(
                top_level,
                page_number,
                predicate=lambda entry: entry["title"].casefold().startswith("chapter "),
                minimum_page=chapter1_page,
            )
            if chapter is not None:
                metadata["section"] = chapter["title"]
                metadata["chapter"] = chapter["title"]
                subsection = _last_entry(
                    entries,
                    page_number,
                    predicate=lambda entry, chapter_title=chapter["title"]: (
                        entry["parent"] == chapter_title
                    ),
                    minimum_page=chapter["page"],
                )
                if subsection is not None:
                    metadata["subsection"] = subsection["title"]
            else:
                metadata["section"] = metadata["part"]
        else:
            metadata["part"] = "Front matter"
            division = _last_entry(top_level, page_number)
            metadata["section"] = (
                division["title"] if division is not None else "Front matter"
            )

        enriched.append(replace(source_page, metadata=metadata))
    return enriched


def _content_pages(pages: list[DocumentPage]) -> list[DocumentPage]:
    """Physical pages selected as the useful reading-order corpus.

    The range is 25-263 inclusive, minus blank pages, part dividers and pages
    that contain only references. Every remaining page has useful narrative,
    notes, figures or profile data and is counted in the manifest.
    """
    return [
        page
        for page in pages
        if CONTENT_PAGE_START <= page.page <= CONTENT_PAGE_END
        and page.page not in EXCLUDED_PAGES
    ]


def _indexable_pages(pages: list[DocumentPage]) -> list[DocumentPage]:
    """Return indexable pages only from the selected physical page range.

    Repeat the page-range and exclusion checks here so callers cannot
    accidentally index front matter, appendices, blank pages, dividers, or
    reference-only pages by passing an unfiltered page list.
    """
    return [
        page
        for page in pages
        if CONTENT_PAGE_START <= page.page <= CONTENT_PAGE_END
        and page.page not in EXCLUDED_PAGES
        and len(page.text.strip()) >= MIN_CONTENT_CHARS
    ]


def _corpus_manifest(pdf_path: Path, pages: list[DocumentPage],
                     content: list[DocumentPage],
                     indexable: list[DocumentPage]) -> dict[str, Any]:
    text_pages = [page for page in pages if page.text.strip()]
    empty_pages = [page.page for page in pages if not page.text.strip()]
    parts = Counter(str(page.metadata.get("part", "unknown")) for page in pages)
    physical_in_range = CONTENT_PAGE_END - CONTENT_PAGE_START + 1
    excluded = sorted(
        page
        for page in range(CONTENT_PAGE_START, CONTENT_PAGE_END + 1)
        if page in EXCLUDED_PAGES
    )

    return {
        "source_path": _project_path(pdf_path),
        "document_id": pages[0].document_id if pages else None,
        "source": pages[0].source if pages else None,
        "pdf_title": pages[0].metadata.get("pdf_title", "") if pages else "",
        "year": pages[0].metadata.get("year") if pages else None,
        "physical_pages": len(pages),
        "pages_with_text": len(text_pages),
        "physical_pages_in_range": physical_in_range,
        "content_pages": len(content),
        "content_page_range": f"{CONTENT_PAGE_START}-{CONTENT_PAGE_END}",
        "content_page_numbers": [page.page for page in content],
        "content_page_count_rule": (
            f"{CONTENT_PAGE_START}-{CONTENT_PAGE_END} is {physical_in_range} "
            f"physical pages; excluding {len(excluded)} non-content pages "
            f"(blank, part dividers, references only) leaves {len(content)} "
            "useful content pages."
        ),
        "excluded_pages_in_range": {
            "blank": sorted(BLANK_PAGES),
            "part_dividers": sorted(DIVIDER_PAGES),
            "references_only": sorted(REFERENCE_ONLY_PAGES),
        },
        "indexed_pages": len(indexable),
        "indexed_page_numbers": [page.page for page in indexable],
        "empty_or_scanned_pages": empty_pages,
        "total_characters": sum(len(page.text) for page in pages),
        "content_characters": sum(len(page.text) for page in content),
        "indexed_characters": sum(len(page.text) for page in indexable),
        "parts": dict(parts),
        "minimum_content_pages_required": MIN_CONTENT_PAGES,
        "minimum_content_chars_per_page": MIN_CONTENT_CHARS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ------------------------------------------------------------- benchmark input
def _benchmark_candidates() -> list[Path]:
    return [
        STARTER_DIR / "data" / "benchmark.json",
        STARTER_DIR / "data" / "questions.json",
        STARTER_DIR / "benchmark" / "benchmark.json",
        STARTER_DIR / "benchmark" / "questions.json",
        REPO_DIR / "benchmark" / "questions.json",
        REPO_DIR / "benchmark" / "benchmark.json",
        REPO_DIR / "data" / "benchmark.json",
    ]


def _resolve_benchmark(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit if explicit.is_absolute() else (Path.cwd() / explicit)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")
        return path
    for candidate in _benchmark_candidates():
        if candidate.exists():
            return candidate
    searched = "\n  ".join(_project_path(path) for path in _benchmark_candidates())
    raise FileNotFoundError(
        "Benchmark file not found. Create data/benchmark.json (or "
        "benchmark/questions.json at the repository root) with at least 20 "
        f"questions. Searched:\n  {searched}"
    )


def _load_questions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("questions", "items", "benchmark"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            if "id" in payload and "question" in payload:
                payload = [payload]
            else:
                raise ValueError(
                    f"{path} must contain a list of questions or a 'questions' key"
                )
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path} does not contain any benchmark questions")

    questions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for position, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question #{position} in {path} is not an object")
        question = dict(item)
        question.setdefault("id", f"q{position:02d}")
        question_id = str(question["id"])
        if question_id in seen_ids:
            raise ValueError(f"Duplicate question id: {question_id}")
        seen_ids.add(question_id)
        questions.append(question)
    return questions


def _validate_questions(questions: list[dict[str, Any]],
                        pages_by_number: dict[int, DocumentPage]) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    splits = Counter(str(question.get("split", "unknown")) for question in questions)
    filter_questions = [
        question
        for question in questions
        if isinstance(question.get("filter"), dict) and question["filter"]
    ]
    test_filter_questions = [
        question for question in filter_questions if question.get("split") == "test"
    ]

    for question in questions:
        question_id = str(question.get("id", "?"))
        for field in ("split", "question", "expected_answer"):
            if field not in question:
                warnings.append(f"{question_id}: missing field {field!r}")

        sources = _gold_values(question, _GOLD_SOURCE_KEYS)
        pages = _gold_values(question, _GOLD_PAGE_KEYS)
        quotes = _gold_quotes(question)
        if not sources:
            warnings.append(f"{question_id}: no gold source id supplied")
        if not pages:
            warnings.append(f"{question_id}: no gold page supplied")
        elif pages:
            try:
                page_number = int(str(pages[0]).strip())
            except (TypeError, ValueError):
                errors.append(f"{question_id}: gold page is not an integer")
                page_number = None
            if page_number is not None:
                page = pages_by_number.get(page_number)
                if page is None:
                    errors.append(
                        f"{question_id}: gold_page {page_number} is not an indexable "
                        f"content page in {CONTENT_PAGE_START}-{CONTENT_PAGE_END} "
                        f"(minimum {MIN_CONTENT_CHARS} extracted characters)"
                    )
                else:
                    if quotes and not _evidence_in_text(page.text, quotes[0]):
                        warnings.append(
                            f"{question_id}: evidence quote was not found on page "
                            f"{page_number} after normalization"
                        )
                    filters = question.get("filter")
                    if isinstance(filters, dict) and filters:
                        record = {
                            **page.metadata,
                            "document_id": page.document_id,
                            "source": page.source,
                            "page": page.page,
                        }
                        if not _matches_filter(record, filters):
                            warnings.append(
                                f"{question_id}: filter {filters!r} does not match "
                                f"gold page {page_number} metadata"
                            )
        if not quotes:
            warnings.append(f"{question_id}: no gold evidence quote supplied")

    if len(questions) < 20:
        warnings.append(f"Only {len(questions)} questions; the lab requires >= 20")
    if splits.get("dev", 0) < 15:
        warnings.append(f"Only {splits.get('dev', 0)} dev questions; requires >= 15")
    if splits.get("test", 0) < 5:
        warnings.append(f"Only {splits.get('test', 0)} test questions; requires >= 5")
    if len(filter_questions) < 3:
        warnings.append(
            f"Only {len(filter_questions)} questions use a metadata filter; "
            "the lab requires >= 3"
        )
    if len(test_filter_questions) < 1:
        warnings.append("No test question uses a metadata filter; requires >= 1")

    return {
        "question_count": len(questions),
        "split_counts": dict(splits),
        "filter_questions": len(filter_questions),
        "test_filter_questions": len(test_filter_questions),
        "errors": errors,
        "warnings": warnings,
    }


# ------------------------------------------------------------------- chunking
def _build_chunks(pages: list[DocumentPage], fixed_length: int,
                  fixed_overlap: int,
                  paragraph_length: int) -> tuple[list[Chunk], list[Chunk]]:
    fixed_chunks: list[Chunk] = []
    paragraph_chunks: list[Chunk] = []
    for page in pages:
        if not page.text.strip():
            continue
        fixed_chunks.extend(fixed_size(page, fixed_length, fixed_overlap))
        paragraph_chunks.extend(by_paragraph(page, paragraph_length))
    return fixed_chunks, paragraph_chunks


def _chunk_statistics(strategy: str, chunks: list[Chunk]) -> dict[str, Any]:
    lengths = [len(chunk.text) for chunk in chunks]
    if not lengths:
        return {
            "strategy": strategy,
            "chunk_count": 0,
            "characters": 0,
            "min_chars": 0,
            "mean_chars": 0.0,
            "max_chars": 0,
            "estimated_tokens": 0,
            "documents": 0,
            "pages": 0,
        }
    return {
        "strategy": strategy,
        "chunk_count": len(chunks),
        "characters": sum(lengths),
        "min_chars": min(lengths),
        "mean_chars": round(sum(lengths) / len(lengths), 2),
        "max_chars": max(lengths),
        "estimated_tokens": sum(len(chunk.text.split()) for chunk in chunks),
        "documents": len({chunk.document_id for chunk in chunks}),
        "pages": len({(chunk.document_id, chunk.page) for chunk in chunks}),
    }


# --------------------------------------------------------------------- queries
def _run_queries(store: PgVectorStore, embedder: MyEmbedder,
                 strategies: list[dict[str, Any]],
                 questions: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for strategy in strategies:
        key = strategy["key"]
        for question in questions:
            query_start = time.perf_counter()
            query_vector = embedder.embed_query(str(question["question"]))
            filters = None
            if strategy["uses_filter"] and isinstance(question.get("filter"), dict):
                filters = question["filter"] or None
            hits = store.search(key, query_vector, top_k=top_k, filters=filters)
            latency_ms = (time.perf_counter() - query_start) * 1000.0

            evaluated = evaluate_query(
                {**question, "strategy": key}, hits
            )
            evaluated["strategy"] = key
            evaluated["filter_applied"] = filters is not None
            evaluated["filter_compliant"] = (
                _trace_filter_compliance(evaluated, filters)
                if filters is not None
                else None
            )
            evaluated["latency_ms"] = round(latency_ms, 3)
            traces.append(evaluated)
    return traces


def _filter_compliance_summary(traces: list[dict[str, Any]],
                               strategies: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for strategy in strategies:
        key = strategy["key"]
        relevant = [
            trace
            for trace in traces
            if trace["strategy"] == key and trace.get("filter_applied")
        ]
        compliant = [trace for trace in relevant if trace.get("filter_compliant")]
        result[key] = {
            "questions_with_filter": len(relevant),
            "compliant_questions": len(compliant),
            "compliance_rate": (
                len(compliant) / len(relevant) if relevant else None
            ),
            "note": (
                "A question is compliant when every returned hit matches its "
                "filter. Questions without a filter are not counted."
            ),
        }
    return result


# -------------------------------------------------------------- manual samples
def _select_coherence_positions(pages: list[DocumentPage],
                               count: int) -> list[DocumentPage]:
    candidates = [page for page in pages if page.text.strip()]
    if len(candidates) <= count:
        return candidates
    step = (len(candidates) - 1) / (count - 1)
    selected: list[DocumentPage] = []
    seen: set[int] = set()
    for index in range(count):
        position = round(index * step)
        page = candidates[position]
        if page.page not in seen:
            selected.append(page)
            seen.add(page.page)
    return selected


def _coherence_samples(pages: list[DocumentPage],
                       strategies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positions = _select_coherence_positions(pages, MANUAL_COHERENCE_POSITIONS)
    samples: list[dict[str, Any]] = []
    for page in positions:
        for strategy in strategies:
            chunks = strategy["chunks"]
            page_chunks = [chunk for chunk in chunks if chunk.page == page.page]
            if not page_chunks:
                samples.append(
                    {
                        "strategy": strategy["key"],
                        "page": page.page,
                        "source": page.source,
                        "section": page.metadata.get("section"),
                        "chunk_id": None,
                        "chunk_text": None,
                        "coherence_score": None,
                        "reason": "No chunk covers this source position.",
                    }
                )
                continue
            chunk = min(page_chunks, key=lambda item: item.metadata.get("char_start", 0))
            samples.append(
                {
                    "strategy": strategy["key"],
                    "page": page.page,
                    "source": page.source,
                    "section": page.metadata.get("section"),
                    "chunk_id": chunk.chunk_id,
                    "chunk_text": chunk.text,
                    "metadata": chunk.metadata,
                    "coherence_score": None,
                    "reason": None,
                    "rubric": {
                        "0": "cut mid-idea or lacks context to understand",
                        "1": "partially understandable, missing context",
                        "2": "self-contained and keeps the complete idea",
                    },
                }
            )
    return samples


def _grounding_samples(traces: list[dict[str, Any]],
                       strategies: list[dict[str, Any]],
                       questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    test_questions = [
        question for question in questions if question.get("split") == "test"
    ][:MANUAL_TEST_QUESTIONS]
    samples: list[dict[str, Any]] = []
    for strategy in strategies:
        key = strategy["key"]
        for question in test_questions:
            trace = next(
                (
                    item
                    for item in traces
                    if item["strategy"] == key and item.get("id") == question.get("id")
                ),
                None,
            )
            if trace is None:
                continue
            samples.append(
                {
                    "strategy": key,
                    "question_id": trace.get("id"),
                    "split": trace.get("split"),
                    "question": trace.get("question"),
                    "expected_answer": trace.get("expected_answer"),
                    "gold_sources": trace.get("gold_sources"),
                    "gold_pages": trace.get("gold_pages"),
                    "evidence_quotes": trace.get("evidence_quotes"),
                    "hit@3": trace.get("hit@3"),
                    "hits": trace.get("hits"),
                    "grounding_score": None,
                    "reason": None,
                    "rubric": {
                        "0": "no evidence in top-3",
                        "1": "evidence present but missing a condition/context",
                        "2": "top-3 is sufficient to answer and trace the source",
                    },
                }
            )
    return samples


def _metadata_samples(traces: list[dict[str, Any]],
                      strategies: list[dict[str, Any]],
                      questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    test_questions = [
        question for question in questions if question.get("split") == "test"
    ][:MANUAL_TEST_QUESTIONS]
    samples: list[dict[str, Any]] = []
    for strategy in strategies:
        key = strategy["key"]
        for question in test_questions:
            trace = next(
                (
                    item
                    for item in traces
                    if item["strategy"] == key and item.get("id") == question.get("id")
                ),
                None,
            )
            if trace is None:
                continue
            samples.append(
                {
                    "strategy": key,
                    "question_id": trace.get("id"),
                    "filter": trace.get("filter"),
                    "filter_applied": trace.get("filter_applied"),
                    "filter_compliant": trace.get("filter_compliant"),
                    "hits": [
                        {
                            "rank": hit.get("rank"),
                            "chunk_id": hit.get("chunk_id"),
                            "source": hit.get("source"),
                            "page": hit.get("page"),
                            "metadata": hit.get("metadata"),
                            "metadata_correct": None,
                            "reason": None,
                        }
                        for hit in (trace.get("hits") or [])
                    ],
                }
            )
    return samples


# ----------------------------------------------------------------------- main
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Day 06 ingestion + retrieval benchmark."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                        help="PDF file to ingest (default: data/data.pdf).")
    parser.add_argument("--document-id", default=DEFAULT_DOCUMENT_ID,
                        help="Stable document id used as gold_source_id.")
    parser.add_argument("--source", default=None,
                        help="Source label stored with every chunk.")
    parser.add_argument("--benchmark", type=Path, default=None,
                        help="Benchmark JSON (default: data/benchmark.json).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for traces, metrics and samples.")
    parser.add_argument("--model", default=None,
                        help=f"Embedding model (default: {DEFAULT_MODEL_NAME}).")
    parser.add_argument("--device", default=None,
                        help="Torch device, e.g. cpu or cuda (default: auto).")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Embedding batch size (default: 64).")
    parser.add_argument("--fixed-size", type=int, default=500,
                        help="Strategy A character window size (default: 500).")
    parser.add_argument("--fixed-overlap", type=int, default=50,
                        help="Strategy A character overlap (default: 50).")
    parser.add_argument("--paragraph-size", type=int, default=800,
                        help="Strategy B/C maximum chunk size (default: 800).")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of hits per question (default: 3).")
    parser.add_argument("--database-url", default=None,
                        help="PostgreSQL URL (default: DATABASE_URL from .env).")
    parser.add_argument("--strict-benchmark", action="store_true",
                        help="Abort when the benchmark fails validation.")
    parser.add_argument("--skip-idempotency-check", action="store_true",
                        help="Do not re-upsert to verify stable chunk counts.")
    parser.add_argument("--limit-questions", type=int, default=0,
                        help="Run only the first N questions (debugging only).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    load_dotenv(STARTER_DIR / ".env")

    corpus_path = args.corpus if args.corpus.is_absolute() else (Path.cwd() / args.corpus)
    corpus_path = corpus_path.resolve()
    if not corpus_path.exists():
        raise SystemExit(f"Corpus not found: {_project_path(corpus_path)}")

    try:
        benchmark_path = _resolve_benchmark(args.benchmark)
        questions = _load_questions(benchmark_path)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Benchmark error: {exc}") from exc
    if args.limit_questions > 0:
        questions = questions[: args.limit_questions]
        print(f"WARNING: debugging run with only {len(questions)} question(s).")

    output_dir = args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    database_url = args.database_url or os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit(
            "DATABASE_URL is missing. Copy starter/.env.example to starter/.env "
            "and start PostgreSQL with docker compose."
        )

    source = _source_for(corpus_path, args.source)

    # 1. Corpus ---------------------------------------------------------------
    print(f"Reading {_project_path(corpus_path)} ...")
    load_start = time.perf_counter()
    pages = load_pdf(corpus_path, args.document_id, source)
    pages = _enrich_pages(corpus_path, pages)
    load_seconds = time.perf_counter() - load_start

    content_pages = _content_pages(pages)
    indexable_pages = _indexable_pages(content_pages)
    if len(indexable_pages) < MIN_CONTENT_PAGES:
        raise SystemExit(
            f"Only {len(indexable_pages)} usable content pages found; the lab "
            f"requires at least {MIN_CONTENT_PAGES}. Check the PDF extraction "
            "or use another corpus."
        )

    manifest = _corpus_manifest(corpus_path, pages, content_pages,
                                indexable_pages)
    manifest["load_seconds"] = round(load_seconds, 3)
    _json_write(output_dir / "ingestion_manifest.json", manifest)
    print(
        f"Corpus: {manifest['physical_pages']} physical pages, "
        f"{manifest['pages_with_text']} with text, "
        f"{manifest['content_pages']} content pages in "
        f"{manifest['content_page_range']}, "
        f"{manifest['indexed_pages']} indexed (>= {MIN_CONTENT_CHARS} chars)."
    )

    # 2. Benchmark preflight --------------------------------------------------
    pages_by_number = {page.page: page for page in indexable_pages}
    benchmark_report = _validate_questions(questions, pages_by_number)
    benchmark_report["benchmark_path"] = _project_path(benchmark_path)
    benchmark_report["loaded_questions"] = len(questions)
    _json_write(output_dir / "benchmark_validation.json", benchmark_report)

    if benchmark_report["errors"]:
        for error in benchmark_report["errors"]:
            print(f"BENCHMARK ERROR: {error}")
        if args.strict_benchmark:
            raise SystemExit("Benchmark validation failed.")
    for warning in benchmark_report["warnings"]:
        print(f"BENCHMARK WARNING: {warning}")

    # 3. Embedder and vector store -------------------------------------------
    model_name = args.model or os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL_NAME
    print(f"Loading embedding model {model_name!r} ...")
    model_start = time.perf_counter()
    embedder = MyEmbedder(
        model_name,
        device=args.device,
        batch_size=args.batch_size,
    )
    model_seconds = time.perf_counter() - model_start

    store = PgVectorStore(database_url, embedder.dimensions)
    store.create_schema()

    # 4. Chunking -------------------------------------------------------------
    chunk_start = time.perf_counter()
    fixed_chunks, paragraph_chunks = _build_chunks(
        indexable_pages,
        args.fixed_size,
        args.fixed_overlap,
        args.paragraph_size,
    )
    chunk_seconds = time.perf_counter() - chunk_start
    if not fixed_chunks or not paragraph_chunks:
        raise SystemExit("Chunking produced no chunks; check the corpus text.")

    strategies: list[dict[str, Any]] = [
        {
            "key": "A_fixed",
            "label": "A - fixed-size windows",
            "description": (
                f"{args.fixed_size} characters, {args.fixed_overlap} overlap"
            ),
            "chunks": fixed_chunks,
            "uses_filter": False,
        },
        {
            "key": "B_paragraph",
            "label": "B - paragraph-aware",
            "description": f"paragraph packing, max {args.paragraph_size} characters",
            "chunks": paragraph_chunks,
            "uses_filter": False,
        },
        {
            "key": "C_paragraph_filter",
            "label": "C - paragraph-aware + metadata filter",
            "description": (
                "same chunks as B; applies question metadata filter when provided"
            ),
            "chunks": paragraph_chunks,
            "uses_filter": True,
        },
    ]

    # 5. Embedding and ingestion ---------------------------------------------
    embed_start = time.perf_counter()
    fixed_vectors = embedder.embed_documents(
        [chunk.text for chunk in fixed_chunks]
    )
    paragraph_vectors = embedder.embed_documents(
        [chunk.text for chunk in paragraph_chunks]
    )
    embed_seconds = time.perf_counter() - embed_start
    if len(fixed_vectors) != len(fixed_chunks):
        raise RuntimeError("Embedder returned the wrong number of A vectors")
    if len(paragraph_vectors) != len(paragraph_chunks):
        raise RuntimeError("Embedder returned the wrong number of B/C vectors")

    vector_by_strategy = {
        "A_fixed": fixed_vectors,
        "B_paragraph": paragraph_vectors,
        "C_paragraph_filter": paragraph_vectors,
    }

    ingest_seconds: dict[str, float] = {}
    stored_counts: dict[str, int] = {}
    for strategy in strategies:
        key = strategy["key"]
        chunks = strategy["chunks"]
        vectors = vector_by_strategy[key]
        print(f"Writing strategy {key} ({len(chunks)} chunks) ...")
        write_start = time.perf_counter()
        store.upsert(key, chunks, vectors)
        stored_counts[key] = store.count(key)
        if not args.skip_idempotency_check:
            # Re-ingesting the same source version must not create duplicates.
            store.upsert(key, chunks, vectors)
            repeated_count = store.count(key)
            if repeated_count != stored_counts[key]:
                raise RuntimeError(
                    f"Re-ingestion changed {key} count from "
                    f"{stored_counts[key]} to {repeated_count}"
                )
        ingest_seconds[key] = round(time.perf_counter() - write_start, 3)

    # 6. Retrieval traces -----------------------------------------------------
    print(f"Running {len(questions)} questions x {len(strategies)} strategies ...")
    retrieval_start = time.perf_counter()
    traces = _run_queries(store, embedder, strategies, questions, args.top_k)
    retrieval_seconds = time.perf_counter() - retrieval_start

    summary = summarize(traces)
    summary["filter_compliance"] = _filter_compliance_summary(traces, strategies)
    summary["resources"] = {
        "model_name": embedder.model_name,
        "dimensions": embedder.dimensions,
        "device": embedder.device,
        "batch_size": embedder.batch_size,
        "cost_usd": 0.0,
        "corpus_load_seconds": round(load_seconds, 3),
        "model_load_seconds": round(model_seconds, 3),
        "chunking_seconds": round(chunk_seconds, 3),
        "embedding_seconds": round(embed_seconds, 3),
        "ingest_seconds_by_strategy": ingest_seconds,
        "retrieval_seconds": round(retrieval_seconds, 3),
        "stored_chunks": stored_counts,
    }
    summary["chunk_statistics"] = [
        _chunk_statistics(strategy["key"], strategy["chunks"])
        for strategy in strategies
    ]

    # 7. Persist traces, metrics and manual-review samples --------------------
    for strategy in strategies:
        key = strategy["key"]
        _json_write(
            output_dir / f"traces_{key}.json",
            [trace for trace in traces if trace["strategy"] == key],
        )
    _json_write(output_dir / "traces.json", traces)
    _json_write(output_dir / "summary.json", summary)

    coherence = _coherence_samples(indexable_pages, strategies)
    grounding = _grounding_samples(traces, strategies, questions)
    metadata_review = _metadata_samples(traces, strategies, questions)
    _json_write(output_dir / "manual_coherence_samples.json", coherence)
    _json_write(output_dir / "manual_grounding_samples.json", grounding)
    _json_write(output_dir / "manual_metadata_samples.json", metadata_review)

    config = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "corpus": _project_path(corpus_path),
        "document_id": args.document_id,
        "source": source,
        "benchmark": _project_path(benchmark_path),
        "database_url": _redact_database_url(database_url),
        "model": {
            "name": embedder.model_name,
            "dimensions": embedder.dimensions,
            "device": embedder.device,
            "normalized": embedder.normalize,
            "batch_size": embedder.batch_size,
        },
        "strategies": [
            {
                "key": strategy["key"],
                "label": strategy["label"],
                "description": strategy["description"],
                "uses_filter": strategy["uses_filter"],
                "chunk_count": len(strategy["chunks"]),
            }
            for strategy in strategies
        ],
        "parameters": {
            "fixed_size": args.fixed_size,
            "fixed_overlap": args.fixed_overlap,
            "paragraph_size": args.paragraph_size,
            "top_k": args.top_k,
        },
        "model_load_seconds": round(model_seconds, 3),
    }
    _json_write(output_dir / "run_config.json", config)

    # 8. Console report -------------------------------------------------------
    print("\nRetrieval metrics (counts / n):")
    for strategy in strategies:
        key = strategy["key"]
        rows = summary["by_strategy_split"].get(key, {})
        for split in sorted(rows):
            block = rows[split]
            if not block["n"]:
                continue
            print(
                f"  {key:<22} {split:<5} "
                f"Hit@1 {block['hit@1_count']}/{block['n']} "
                f"({block['hit@1']:.3f})  "
                f"Evidence Hit@3 {block['hit@3_count']}/{block['n']} "
                f"({block['hit@3']:.3f})"
            )

    print("\nFilter compliance (strategy C, questions with a filter):")
    for key, block in summary["filter_compliance"].items():
        if block["questions_with_filter"]:
            rate = block["compliance_rate"]
            rate_text = f"{rate:.3f}" if rate is not None else "n/a"
            print(
                f"  {key}: {block['compliant_questions']}/"
                f"{block['questions_with_filter']} ({rate_text})"
            )

    print("\nChunk statistics:")
    for stats in summary["chunk_statistics"]:
        print(
            f"  {stats['strategy']:<22} chunks={stats['chunk_count']:<5} "
            f"chars={stats['characters']:<9} mean={stats['mean_chars']:<7} "
            f"max={stats['max_chars']}"
        )

    print(f"\nWrote results to {_project_path(output_dir)}")
    print("Manual-review files contain null scores; fill them in with a human label.")


if __name__ == "__main__":
    main()
