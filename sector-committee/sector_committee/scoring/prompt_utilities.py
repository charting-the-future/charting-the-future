"""Text formatting utilities for prompt processing.

This module contains general-purpose text formatting and normalization functions
that can be reused across different prompt modules. The utilities handle text
wrapping, bullet point formatting, header preservation, and other text
processing tasks common to prompt engineering.
"""

import re
import textwrap
from typing import Dict, List, Tuple


_BULLET_RE = re.compile(r"^(\s*)(-\s+|\u2022\s+|\d+\.\s+)(.*)$")
_HEADER_RE = re.compile(r"^\s*([A-Z][A-Z \-/&]+:)\s*$")


def format_prompt(
    raw: str,
    width: int = 88,
    list_indent: int = 2,
    titlecase_sector_phrase: bool = True,
    replacements: Dict[str, str] | None = None,
) -> str:
    """Normalize a multi-line prompt into a clean, human-readable block.

    - Preserves section headers (ALL CAPS + ':').
    - Collapses wrapped bullets/numbered items to single logical lines.
    - Wraps paragraphs/bullets to `width`.
    - Optionally title-cases '<something> sector' in the opening sentence.
    - Applies optional regex replacements at the end.

    Args:
        raw: The raw multi-line string (e.g., a triple-quoted prompt).
        width: Target wrap width (default 88).
        list_indent: Spaces to indent bullets/numbered items (default 2).
        titlecase_sector_phrase: If True, title-cases '<x> sector' after 'on the'.
        replacements: Optional {pattern: replacement} regex map applied last.

    Returns:
        Formatted string.
    """
    text = textwrap.dedent(raw).strip("\n")
    lines = text.splitlines()

    # Parse into logical paragraphs.
    out: List[Tuple[str, ...] | str] = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # Preserve blank lines
        if not line.strip():
            out.append("")
            i += 1
            continue

        # Headers like 'RESEARCH OBJECTIVES:'
        m_header = _HEADER_RE.match(line)
        if m_header:
            out.append(("header", m_header.group(1).strip()))
            i += 1
            continue

        # Bullets / numbered items
        m_bullet = _BULLET_RE.match(line)
        if m_bullet:
            indent, bullet_token, body = m_bullet.groups()
            parts = [body.strip()]
            i += 1
            # Consume continuation lines until next bullet/header/blank
            while i < len(lines):
                nxt = lines[i].rstrip()
                if not nxt.strip() or _BULLET_RE.match(nxt) or _HEADER_RE.match(nxt):
                    break
                parts.append(nxt.strip())
                i += 1
            full = " ".join(parts)
            out.append(("bullet", bullet_token.strip(), full))
            continue

        # Plain paragraph (join until blank/header/bullet)
        parts = [line.strip()]
        i += 1
        while i < len(lines):
            nxt = lines[i].rstrip()
            if not nxt.strip() or _BULLET_RE.match(nxt) or _HEADER_RE.match(nxt):
                break
            parts.append(nxt.strip())
            i += 1
        out.append(("para", " ".join(parts)))

    # Render with wrapping
    rendered: List[str] = []
    for chunk in out:
        if chunk == "":
            rendered.append("")
            continue

        kind = chunk[0]
        if kind == "header":
            rendered.append(chunk[1])
        elif kind == "bullet":
            bullet_token, body = chunk[1], chunk[2]
            init = " " * list_indent + bullet_token + " "
            subs = " " * (list_indent + len(bullet_token) + 1)
            w = textwrap.TextWrapper(
                width=width, initial_indent=init, subsequent_indent=subs
            )
            rendered.append(w.fill(body))
        elif kind == "para":
            body = chunk[1]
            w = textwrap.TextWrapper(width=width)
            rendered.append(w.fill(body))

    result = "\n".join(rendered)

    # Optional: title-case '<x> sector' after 'on the '
    if titlecase_sector_phrase:

        def _tc(m: re.Match) -> str:
            return "on the " + m.group(1).title() + " sector"

        result = re.sub(
            r"\bon the ([a-z][a-z /&-]+) sector\b", _tc, result, flags=re.IGNORECASE
        )

    # Optional final regex replacements
    if replacements:
        for pat, repl in replacements.items():
            result = re.sub(pat, repl, result)

    return result
