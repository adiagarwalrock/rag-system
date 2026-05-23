"""Page-marker parsing utilities for external parsers."""

from __future__ import annotations

import re

_PAGE_START = re.compile(r"^\[\[START OF PAGE (\d+)\]\]$")
_PAGE_END = re.compile(r"^\[\[END OF PAGE (\d+)\]\]$")


def split_by_page_markers(markdown: str) -> list[tuple[int, str]]:
    """Split markdown on [[START OF PAGE n]] / [[END OF PAGE n]] pairs.

    Returns [(page_num, content), ...]. Falls back to [(1, markdown)] when no
    markers are present so callers always get at least one page.
    """
    pages: list[tuple[int, str]] = []
    current_page: int | None = None
    buffer: list[str] = []

    for line in markdown.splitlines():
        m_start = _PAGE_START.match(line.strip())
        m_end = _PAGE_END.match(line.strip())
        if m_start:
            buffer = []
            current_page = int(m_start.group(1))
        elif m_end:
            if current_page is not None:
                pages.append((current_page, "\n".join(buffer).strip()))
            buffer = []
            current_page = None
        else:
            buffer.append(line)

    if not pages:
        # No markers — treat whole document as page 1
        return [(1, markdown.strip())]

    return [(p, c) for p, c in pages if c]
