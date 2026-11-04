"""Minimal, safe Markdown rendering for LLM answers in the Ask Fred panel.

The model emits Markdown (`**bold**`, `-` lists, `##` headings) which the chat
bubble used to print literally. We do NOT pull in a Markdown library plus a
sanitizer, because the text is UNTRUSTED: the answer is generated from a meeting
transcript, so anything a participant said — or a line planted in a shared
meeting — reaches this renderer.

The safety property is structural, not a blocklist: the whole string is HTML
escaped FIRST, so no markup from the model can survive, and only then is a fixed
subset converted into a closed set of tags we emit ourselves. There is no path
by which model output becomes an attribute, a URL or a tag name. Unsupported
syntax degrades to its literal characters, which is what the panel already did.

Supported: paragraphs, hard line breaks, `-`/`*`/`1.` lists, `**bold**`,
`*italic*`/`_italic_`, `` `code` ``, `#` headings (rendered as bold lines, since
a chat bubble is no place for an <h1>). Deliberately absent: links, images, raw
HTML, tables, block quotes, fenced code blocks.
"""
from __future__ import annotations

import re

from markupsafe import Markup, escape

_BULLET = re.compile(r"^\s*[-*]\s+(.*)$")
_ORDERED = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_HEADING = re.compile(r"^\s*#{1,6}\s+(.*)$")

_CODE = re.compile(r"`([^`\n]+)`")

# Applied to ALREADY-ESCAPED text, so the patterns can only ever wrap plain
# characters. Code spans are pulled out first and re-inserted last: running
# these afterwards over the emitted <code> would still rewrite its CONTENT
# (`**x**` became <code><strong>x</strong></code> until a test caught it).
_INLINE = (
    (re.compile(r"\*\*([^\s*](?:[^*]*[^\s*])?)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<![\w*])\*([^\s*](?:[^*]*[^\s*])?)\*(?![\w*])"), r"<em>\1</em>"),
    (re.compile(r"(?<![\w_])_([^\s_](?:[^_]*[^\s_])?)_(?![\w_])"), r"<em>\1</em>"),
)


def _inline(raw: str) -> str:
    # \x00 is the sentinel for extracted code spans, so it must not survive from
    # the input (it cannot appear in HTML output anyway).
    out = str(escape(raw.replace("\x00", "")))

    code: list[str] = []

    def stash(match: re.Match[str]) -> str:
        code.append(match.group(1))
        return f"\x00{len(code) - 1}\x00"

    out = _CODE.sub(stash, out)
    for pattern, repl in _INLINE:
        out = pattern.sub(repl, out)
    for i, snippet in enumerate(code):
        out = out.replace(f"\x00{i}\x00", f"<code>{snippet}</code>")
    return out


def render_markdown(text: str) -> Markup:
    """Escape `text`, then convert a small Markdown subset. Safe to mark safe."""
    if not text:
        return Markup("")

    html: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    list_tag = ""

    def flush_paragraph() -> None:
        if paragraph:
            html.append("<p>" + "<br>".join(paragraph) + "</p>")
            paragraph.clear()

    def flush_list() -> None:
        nonlocal list_tag
        if list_items:
            html.append(f"<{list_tag}>" + "".join(f"<li>{i}</li>" for i in list_items) + f"</{list_tag}>")
            list_items.clear()
            list_tag = ""

    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not line.strip():
            flush_paragraph()
            flush_list()
            continue

        heading = _HEADING.match(line)
        if heading:
            flush_paragraph()
            flush_list()
            html.append(f"<p><strong>{_inline(heading.group(1))}</strong></p>")
            continue

        bullet = _BULLET.match(line)
        ordered = None if bullet else _ORDERED.match(line)
        if bullet or ordered:
            flush_paragraph()
            tag = "ul" if bullet else "ol"
            if list_tag and list_tag != tag:
                flush_list()
            list_tag = tag
            list_items.append(_inline((bullet or ordered).group(1)))
            continue

        flush_list()
        paragraph.append(_inline(line.strip()))

    flush_paragraph()
    flush_list()
    return Markup("".join(html))
