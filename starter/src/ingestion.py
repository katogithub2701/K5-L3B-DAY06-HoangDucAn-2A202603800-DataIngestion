"""Read and normalize source data while keeping page-level traceability."""
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage

# Characters that carry no visible content but can break tokenization/search.
_INVISIBLE_CHARS = dict.fromkeys(
    map(ord, "\ufeff\u200b\u200c\u200d\u2060\u00ad"),
    None,
)
_INTRA_LINE_WHITESPACE = re.compile(r"[ \t\f\v]+")


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract every physical PDF page and retain its 1-based page number.

    Text is normalized with :func:`normalize_text` before it is stored. Pages
    whose extraction is empty (blank or scanned pages) are kept in the result
    with ``text == ""`` and ``metadata["needs_ocr"] is True`` so that a later
    OCR step can handle them. Extraction failures are recorded per page in
    ``metadata["extraction_error"]`` instead of aborting the whole document.
    """
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.is_file():
        raise IsADirectoryError(f"Expected a PDF file, got directory: {pdf_path}")

    reader = PdfReader(str(pdf_path))

    info = reader.metadata
    pdf_title = str(info.title).strip() if info and info.title else ""
    total_pages = len(reader.pages)

    pages: list[DocumentPage] = []
    for page_number, page in enumerate(reader.pages, start=1):
        extraction_error: str | None = None
        raw_text = ""
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:  # A bad page must not hide the rest of the file.
            extraction_error = f"{type(exc).__name__}: {exc}"

        text = normalize_text(raw_text)
        has_text = bool(text)
        metadata = {
            "pdf_title": pdf_title,
            "total_pages": total_pages,
            "has_text": has_text,
            "needs_ocr": not has_text,
            "char_count": len(text),
            "raw_char_count": len(raw_text),
            "extraction_error": extraction_error,
        }
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata=metadata,
            )
        )
    return pages


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving line structure.

    - Compose Unicode to NFC so equivalent Vietnamese/accented text compares
      the same (``e`` + combining acute equals ``é``).
    - Unify line endings and convert non-breaking/Unicode spaces to spaces.
    - Remove zero-width and soft-hyphen formatting characters.
    - Collapse repeated spaces/tabs inside a line, but keep single line breaks
      and at most one blank line between paragraphs.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Unicode line/paragraph separators and NEL are logical newlines.
    text = text.replace("\u2028", "\n").replace("\u2029", "\n").replace("\x85", "\n")
    text = text.translate(_INVISIBLE_CHARS)

    # Turn any remaining non-newline whitespace (NBSP, thin space, ...) into a
    # plain space so downstream tokenizers see a single whitespace character.
    text = "".join(
        " " if char.isspace() and char != "\n" else char for char in text
    )

    lines: list[str] = []
    previous_was_blank = False
    for raw_line in text.split("\n"):
        line = _INTRA_LINE_WHITESPACE.sub(" ", raw_line).strip()
        if line:
            lines.append(line)
            previous_was_blank = False
        elif lines and not previous_was_blank:
            lines.append("")
            previous_was_blank = True

    return "\n".join(lines).strip()
