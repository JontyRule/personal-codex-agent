from __future__ import annotations

import re
from typing import List, Dict


def _heading_path(lines: List[str], idx: int) -> str:
    """Backtrack to build a heading path like H1 > H2 > H3 for a given line index."""
    h1 = h2 = h3 = None
    for i in range(idx, -1, -1):
        line = lines[i].strip()
        if line.startswith("### ") and not h3:
            h3 = line[4:].strip()
        elif line.startswith("## ") and not h2:
            h2 = line[3:].strip()
        elif line.startswith("# ") and not h1:
            h1 = line[2:].strip()
        if h1 and h2 and h3:
            break
    parts = [p for p in [h1, h2, h3] if p]
    return " > ".join(parts) if parts else "Document"


def split_markdown(md_text: str, source_path: str, target_words: int = 950, overlap_words: int = 120) -> List[Dict]:
    """Split markdown by headings and approximate length.

    We target ~800–1200 tokens; with a rough 0.75 words/token approximation, we chunk by ~950 words.
    """
    lines = md_text.splitlines()
    chunks: List[Dict] = []

    # Identify section boundaries by headings (#, ##, ###)
    boundaries = [0]
    for i, line in enumerate(lines):
        if re.match(r"^#{1,3} ", line):
            if i not in boundaries:
                boundaries.append(i)
    if len(lines) not in boundaries:
        boundaries.append(len(lines))
    boundaries = sorted(set(boundaries))

    # Build sections
    sections: List[str] = []
    section_starts: List[int] = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        section = "\n".join(lines[start:end]).strip()
        if section:
            sections.append(section)
            section_starts.append(start)

    # Further split sections into word-bounded chunks with overlap
    for sec, start_idx in zip(sections, section_starts):
        words = sec.split()
        if not words:
            continue
        step = max(1, target_words - overlap_words)
        for wstart in range(0, len(words), step):
            wend = min(len(words), wstart + target_words)
            text = " ".join(words[wstart:wend]).strip()
            if not text:
                continue
            heading = _heading_path(lines, start_idx)
            chunks.append({
                "text": text,
                "source": source_path,
                "heading": heading,
            })
            if wend == len(words):
                break

    return chunks
