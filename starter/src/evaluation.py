"""Evaluate retrieved evidence and preserve per-query traces."""
import re
import unicodedata
from typing import Any

# Gold fields accepted from the benchmark format and documented alternatives.
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

_WHITESPACE_RE = re.compile(r"\s+")
_ELLIPSIS_RE = re.compile(r"\s*(?:\.{3,}|…|\[\.{3,}\]|\[…\])\s*")
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


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _norm_text(value: Any) -> str:
    """Normalize text for deterministic evidence containment checks."""
    text = unicodedata.normalize("NFC", str(value or ""))
    text = text.translate(_QUOTE_TRANSLATION)
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = text.casefold()
        if key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _gold_sources(question: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in _GOLD_SOURCE_KEYS:
        values.extend(str(value) for value in _as_list(question.get(key)))
    return _dedupe(values)


def _gold_pages(question: dict[str, Any]) -> list[Any]:
    values: list[Any] = []
    for key in _GOLD_PAGE_KEYS:
        for value in _as_list(question.get(key)):
            if value is not None and str(value).strip():
                values.append(value)
    return values


def _gold_evidence(question: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in _GOLD_EVIDENCE_KEYS:
        for item in _as_list(question.get(key)):
            if isinstance(item, dict):
                for sub_key in ("quote", "text", "evidence_quote", "evidence"):
                    if item.get(sub_key):
                        values.append(str(item[sub_key]))
                        break
            elif item is not None:
                values.append(str(item))
    return _dedupe(values)


def _hit_view(hit: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(metadata, merged_view)`` for starter-store or Qdrant hits."""
    if not isinstance(hit, dict):
        dump = getattr(hit, "model_dump", None) or getattr(hit, "dict", None)
        if dump is None:
            raise TypeError(
                "each hit must be a dict or expose model_dump()/dict()"
            )
        hit = dump()

    raw_metadata = hit.get("metadata") or hit.get("payload") or {}
    if not isinstance(raw_metadata, dict):
        raw_metadata = {}
    metadata = dict(raw_metadata)

    # Some Qdrant payloads nest the chunk metadata under a "metadata" key.
    nested = metadata.get("metadata")
    if isinstance(nested, dict):
        flattened = dict(nested)
        flattened.update(
            {key: value for key, value in metadata.items() if key != "metadata"}
        )
        metadata = flattened

    view: dict[str, Any] = dict(metadata)
    for key, value in hit.items():
        if key not in ("metadata", "payload") and value is not None:
            view[key] = value
    if "text" not in view and "chunk_text" in view:
        view["text"] = view["chunk_text"]
    return dict(metadata), view


def _source_matches(gold_sources: list[str], view: dict[str, Any]) -> bool | None:
    if not gold_sources:
        return None
    hit_sources = [
        str(value).strip()
        for key in ("document_id", "source_id", "source")
        for value in [view.get(key)]
        if value
    ]
    if not hit_sources:
        return False

    def key(value: str) -> str:
        return value.replace("\\", "/").strip().casefold()

    def basename(value: str) -> str:
        return key(value).rsplit("/", 1)[-1]

    gold_keys = {key(value) for value in gold_sources}
    hit_keys = {key(value) for value in hit_sources}
    if gold_keys & hit_keys:
        return True
    # Tolerate absolute/relative path differences for the same file name.
    if {basename(value) for value in gold_sources} & {
        basename(value) for value in hit_sources
    }:
        return True
    return False


def _page_key(value: Any) -> tuple[str, Any] | None:
    if isinstance(value, bool):
        return None
    try:
        return ("int", int(str(value).strip()))
    except (TypeError, ValueError):
        text = str(value).strip().casefold()
        return ("str", text) if text else None


def _page_matches(gold_pages: list[Any], view: dict[str, Any]) -> bool | None:
    if not gold_pages:
        return None
    hit_page = _page_key(view.get("page"))
    if hit_page is None:
        return False
    gold_keys = {_page_key(page) for page in gold_pages}
    gold_keys.discard(None)
    return hit_page in gold_keys


def _evidence_matches(chunk_text: str, quote: str) -> bool:
    """Check every fragment of one gold quote against a chunk.

    ``...``/``…`` in a gold quote marks omitted text: every remaining fragment
    must occur in the chunk. This is still deterministic substring matching,
    not a fuzzy/semantic judgement of correctness.
    """
    chunk = _norm_text(chunk_text)
    normalized_quote = _norm_text(quote)
    if not chunk or not normalized_quote:
        return False
    fragments = [
        fragment.strip()
        for fragment in _ELLIPSIS_RE.split(normalized_quote)
        if fragment.strip()
    ]
    return bool(fragments) and all(fragment in chunk for fragment in fragments)


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check source/page and gold evidence, and keep hit@1/hit@3 traces.

    ``hits`` may use either ``{"chunk_id", "text", "score", "metadata"}`` from
    the starter store or Qdrant's ``{"id", "score", "payload"}`` shape. Any
    documented alternative gold source/page/evidence field is accepted.
    """
    gold_sources = _gold_sources(question)
    gold_pages = _gold_pages(question)
    evidence_quotes = _gold_evidence(question)

    hit_traces: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        metadata, view = _hit_view(hit)
        chunk_text = view.get("text") or ""
        source_match = _source_matches(gold_sources, view)
        page_match = _page_matches(gold_pages, view)

        matches = [match for match in (source_match, page_match) if match is not None]
        source_page_match = bool(matches) and all(matches)

        if evidence_quotes:
            evidence_match: bool | None = any(
                _evidence_matches(chunk_text, quote) for quote in evidence_quotes
            )
        else:
            evidence_match = None

        is_hit = source_page_match and evidence_match is not False
        chunk_id = view.get("chunk_id") or view.get("id")
        hit_traces.append(
            {
                "rank": rank,
                "chunk_id": str(chunk_id) if chunk_id is not None else None,
                "score": view.get("score", view.get("similarity")),
                "source": view.get("source"),
                "page": view.get("page"),
                "text": chunk_text,
                "metadata": metadata,
                "source_match": source_match,
                "page_match": page_match,
                "source_page_match": source_page_match,
                "evidence_match": evidence_match,
                "is_hit": is_hit,
            }
        )

    top_3 = hit_traces[:3]
    hit_at_1 = bool(hit_traces) and hit_traces[0]["is_hit"]
    hit_at_3 = any(trace["is_hit"] for trace in top_3)
    source_page_hit_at_1 = bool(hit_traces) and hit_traces[0]["source_page_match"]
    source_page_hit_at_3 = any(trace["source_page_match"] for trace in top_3)

    question_id = question.get("id", question.get("question_id"))
    notes = [
        "Automatic retrieval trace only; grounding quality and chunk coherence "
        "require manual labels."
    ]
    if not evidence_quotes:
        notes.append(
            "No gold evidence quote supplied; hit uses source/page match only."
        )
    if not gold_sources and not gold_pages:
        notes.append(
            "No gold source or page supplied, so no hit can be verified."
        )
    if len(hits) < 3:
        notes.append(
            f"Only {len(hits)} hit(s) returned; this question still counts in "
            "the Hit@3 denominator."
        )

    return {
        "id": question_id,
        "split": question.get("split") or "unknown",
        "strategy": question.get("strategy") or question.get("config") or "unknown",
        "question": question.get("question"),
        "type": question.get("type"),
        "expected_answer": question.get("expected_answer"),
        "filter": question.get("filter"),
        "gold_sources": gold_sources,
        "gold_pages": gold_pages,
        "evidence_quotes": evidence_quotes,
        "returned_hits": len(hits),
        "hits": hit_traces,
        "hit@1": hit_at_1,
        "hit@3": hit_at_3,
        "evidence_hit@3": hit_at_3,
        "source_page_hit@1": source_page_hit_at_1,
        "source_page_hit@3": source_page_hit_at_3,
        "notes": notes,
    }


def _trace_bool(trace: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        if key in trace:
            return bool(trace[key])
    return False


def _metric_block(traces: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(traces)

    def count(*keys: str) -> int:
        return sum(1 for trace in traces if _trace_bool(trace, *keys))

    def rate(value: int) -> float | None:
        return (value / total) if total else None

    hit_1 = count("hit@1", "hit_at_1")
    hit_3 = count("hit@3", "hit_at_3", "evidence_hit@3")
    source_page_1 = count("source_page_hit@1", "source_page_hit_at_1")
    source_page_3 = count("source_page_hit@3", "source_page_hit_at_3")

    return {
        "n": total,
        "hit@1_count": hit_1,
        "hit@1": rate(hit_1),
        "hit@3_count": hit_3,
        "hit@3": rate(hit_3),
        "evidence_hit@3_count": hit_3,
        "evidence_hit@3": rate(hit_3),
        "source_page_hit@1_count": source_page_1,
        "source_page_hit@1": rate(source_page_1),
        "source_page_hit@3_count": source_page_3,
        "source_page_hit@3": rate(source_page_3),
    }


def _trace_strategy(trace: dict[str, Any]) -> str:
    value = (
        trace.get("strategy")
        or trace.get("config")
        or trace.get("configuration")
    )
    return str(value) if value else "unknown"


def _trace_split(trace: dict[str, Any]) -> str:
    value = trace.get("split")
    return str(value) if value else "unknown"


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and dev/test split.

    Counts are reported alongside rates (``n`` is the denominator), including
    zero-sample combinations so missing splits are visible instead of silently
    disappearing.
    """
    total = len(traces)
    strategies = sorted({_trace_strategy(trace) for trace in traces})
    splits = sorted({_trace_split(trace) for trace in traces})

    by_strategy_split: dict[str, dict[str, dict[str, Any]]] = {}
    for strategy in strategies:
        by_strategy_split[strategy] = {}
        for split in splits:
            rows = [
                trace
                for trace in traces
                if _trace_strategy(trace) == strategy and _trace_split(trace) == split
            ]
            by_strategy_split[strategy][split] = _metric_block(rows)

    by_strategy = {
        strategy: _metric_block(
            [trace for trace in traces if _trace_strategy(trace) == strategy]
        )
        for strategy in strategies
    }
    by_split = {
        split: _metric_block(
            [trace for trace in traces if _trace_split(trace) == split]
        )
        for split in splits
    }

    return {
        "total_traces": total,
        "overall": _metric_block(traces),
        "by_strategy": by_strategy,
        "by_split": by_split,
        "by_strategy_split": by_strategy_split,
    }
