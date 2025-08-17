from __future__ import annotations
from typing import List, Dict, Any, Tuple
import os
import glob
import hashlib
from datetime import datetime

from bs4 import BeautifulSoup  # type: ignore
from pypdf import PdfReader  # type: ignore
from docx import Document  # type: ignore

ALLOWED_EXTS = {".md", ".mdx", ".txt", ".pdf", ".html", ".htm", ".docx"}


def _read_file_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".md", ".mdx", ".txt"}:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext in {".html", ".htm"}:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text("\n")
    if ext == ".pdf":
        reader = PdfReader(path)
        parts: List[str] = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
        return "\n\n".join(parts)
    if ext == ".docx":
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    # Fallback
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _chunk_text(text: str, target_chars: int = 600, overlap: int = 150) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # Paragraph-aware split first
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    cur_len = 0
    for p in paras:
        if cur_len + len(p) + 2 <= target_chars:
            buf.append(p)
            cur_len += len(p) + 2
        else:
            if buf:
                chunks.append("\n\n".join(buf))
            # start new buffer; if paragraph is huge, hard-split
            if len(p) > target_chars:
                for i in range(0, len(p), target_chars - overlap):
                    chunks.append(p[i : i + (target_chars - overlap)])
                buf = []
                cur_len = 0
            else:
                buf = [p]
                cur_len = len(p)
    if buf:
        chunks.append("\n\n".join(buf))

    # Add sliding overlap between consecutive chunks
    if overlap > 0 and len(chunks) > 1:
        with_overlap: List[str] = []
        for i, ch in enumerate(chunks):
            if i == 0:
                with_overlap.append(ch)
            else:
                prev_tail = chunks[i - 1][-overlap:]
                with_overlap.append(prev_tail + ch)
        chunks = with_overlap
    return chunks


def discover_files(paths: List[str]) -> List[str]:
    discovered: List[str] = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in files:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in ALLOWED_EXTS:
                        discovered.append(os.path.join(root, fn))
        else:
            # allow simple glob patterns
            matches = glob.glob(p, recursive=True)
            for m in matches or [p]:
                if os.path.isdir(m):
                    for root, _, files in os.walk(m):
                        for fn in files:
                            ext = os.path.splitext(fn)[1].lower()
                            if ext in ALLOWED_EXTS:
                                discovered.append(os.path.join(root, fn))
                else:
                    ext = os.path.splitext(m)[1].lower()
                    if ext in ALLOWED_EXTS and os.path.exists(m):
                        discovered.append(m)
    # de-dup and stable sort
    return sorted(list(dict.fromkeys(discovered)))


def load_and_chunk(paths: List[str], tags: List[str] | None = None) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    files = discover_files(paths)
    ids: List[str] = []
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    now = datetime.utcnow().isoformat()

    for fp in files:
        raw = _read_file_text(fp)
        chunks = _chunk_text(raw)
        base_id = os.path.abspath(fp)
        for idx, ch in enumerate(chunks):
            checksum = hashlib.md5(ch.encode("utf-8")).hexdigest()
            ids.append(f"{base_id}::chunk::{idx}")
            texts.append(ch)
            metas.append({
                "doc_id": base_id,
                "chunk_index": idx,
                "source_uri": fp,
                "checksum": checksum,
                # ChromaDB metadata requires primitives; store tags as CSV
                "tags": ",".join(tags) if tags else "",
                "ingested_at": now,
            })
    return ids, texts, metas
