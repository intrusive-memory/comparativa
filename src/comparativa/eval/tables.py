"""A tiny shared Markdown-table renderer for the eval reports.

No dependency worth pulling in for this — GitHub-flavoured Markdown pipe
tables are three lines of string joining.
"""

from __future__ import annotations


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render ``headers``/``rows`` as a Markdown pipe table.

    Returns a placeholder line, not an empty table, when ``rows`` is empty —
    an empty pipe table renders as nothing useful in most Markdown viewers.
    """
    if not rows:
        return "_no rows._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


__all__ = ["markdown_table"]
