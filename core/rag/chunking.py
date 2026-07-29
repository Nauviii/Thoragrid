"""Chunk StatPearls .txt files into token-bounded segments for Pinecone ingestion."""

import re
import hashlib
from pathlib import Path
from transformers import AutoTokenizer

from config.settings import settings


# Loaded once at module level to avoid repeated disk I/O.
_TOKENIZER = AutoTokenizer.from_pretrained(settings.rag_embedding_model)

MAX_TOKENS = 200
OVERLAP_TOKENS = 30

# Sections stripped entirely (lowercased for comparison)
_STRIP_SECTIONS = {
    "enhancing healthcare team outcomes",
    "deterrence and patient education",
}

# Header silently ignored; content continues into current section
_TRANSPARENT_SECTIONS = {"statpearls [internet]."}

# CEA section — extract clinical intro, drop learning objective bullets
_CEA = "continuing education activity"

# First word of CEA learning objective bullets to detect and skip
_OBJECTIVE_VERBS = {
    "identify", "differentiate", "implement", "assess", "summarize",
    "review", "evaluate", "describe", "outline", "recall", "apply",
    "collaborate", "communicate", "select",
}

# Line-level noise matched against stripped line; any match → skip
_LINE_NOISE = re.compile(
    r"^NCBI Bookshelf\."
    r"|^StatPearls \[Internet\]"
    r"|^Authors?\s"
    r"|^Last Update:"
    r"|^Disclosure:"
    r"|^This book is distributed"
    r"|This publication is provided for historical"
    r"|http[s]?://"
    r"| ; [A-Z]"
    r"|\- Access free multiple choice"
    r"|\- Click here"
    r"|\- Comment on this article"
    r"|Contributed by\s"
    r"|Image courtesy\s"
    r"|Public Domain, via"
    r"|Copyright ©"
)

_CITATION_RE  = re.compile(r"\[\d+\]")
_SEE_IMAGE_RE = re.compile(r"\(see Images?\.[^)]*\)")


def _clean_text(text: str) -> str:
    """Remove citation markers and image cross-references from text."""
    text = _CITATION_RE.sub("", text)
    text = _SEE_IMAGE_RE.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _count_tokens(text: str) -> int:
    """Return token count using the embedding model's tokenizer."""
    return len(_TOKENIZER.encode(text, add_special_tokens=False))


def _is_objective_bullet(line: str) -> bool:
    """Return True if a list item is a CEA learning objective."""
    if not line.startswith("- "):
        return False
    first_word = line[2:].split()[0].rstrip(".").lower() if len(line) > 2 else ""
    return first_word in _OBJECTIVE_VERBS


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on punctuation boundaries."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _chunk_text(text: str) -> list[str]:
    """Split section text into MAX_TOKENS chunks with OVERLAP_TOKENS carry-over."""
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _count_tokens(sent)
        if current_tokens + sent_tokens > MAX_TOKENS and current:
            chunks.append(" ".join(current))
            overlap: list[str] = []
            overlap_tokens = 0
            for s in reversed(current):
                t = _count_tokens(s)
                if overlap_tokens + t > OVERLAP_TOKENS:
                    break
                overlap.insert(0, s)
                overlap_tokens += t
            current, current_tokens = overlap, overlap_tokens
        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


def _parse_sections(raw: str) -> dict[str, str]:
    """
    Parse raw file text into {section_name: content} keeping only clinical sections.

    CEA clinical intro → 'Introduction'; objective bullets dropped.
    Transparent headers (StatPearls boilerplate) are skipped without changing state.
    """
    sections: dict[str, str] = {}
    current_header = "Introduction"
    buffer: list[str] = []
    in_cea = False

    def _flush(header: str, buf: list[str]) -> None:
        content = _clean_text("\n".join(buf))
        if content and len(content) > 40:
            existing = sections.get(header, "")
            sections[header] = (existing + "\n" + content).strip()

    for line in raw.splitlines():
        ls = line.strip()

        if ls.startswith("## "):
            header = ls[3:].strip()
            hl = header.lower()

            if hl in _TRANSPARENT_SECTIONS:
                continue

            _flush(current_header, buffer)
            buffer = []

            if hl in _STRIP_SECTIONS:
                current_header = "__skip__"
                in_cea = False
            elif hl == _CEA:
                current_header = "Introduction"
                in_cea = True
            else:
                current_header = header
                in_cea = False
            continue

        if current_header == "__skip__":
            continue
        if _LINE_NOISE.search(ls):
            continue
        if in_cea:
            if ls.startswith("Objectives:"):
                continue
            if _is_objective_bullet(ls):
                continue
        if ls:
            buffer.append(ls)

    _flush(current_header, buffer)
    return sections


def chunk_condition_file(
    filepath: Path,
    condition: str,
    source: str = "statpearls",
) -> list[dict]:
    """Chunk one condition .txt file into metadata-tagged dicts ready for embedding."""
    raw = filepath.read_text(encoding="utf-8")
    sections = _parse_sections(raw)
    chunks: list[dict] = []

    for section, content in sections.items():
        if len(content.split()) < 10:
            continue
        for i, text in enumerate(_chunk_text(content)):
            chunk_id = hashlib.md5(
                f"{condition}:{section}:{i}".encode()
            ).hexdigest()[:12]
            chunks.append({
                "chunk_id":  chunk_id,
                "condition": condition,
                "section":   section,
                "source":    source,
                "text":      text,
            })

    return chunks