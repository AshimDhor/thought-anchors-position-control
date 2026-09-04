"""Sentence segmentation for reasoning traces.

Reasoning traces are not ordinary prose: they contain LaTeX, decimal numbers,
bare newlines used as soft breaks, and abbreviations.  A naive ``split('.')``
shatters ``3.14`` and ``\\frac{1}{2}.`` alike, so we use a guarded regex.

Design note (this one matters).  We return *character spans into the original
trace*, not a list of detached strings.  A prefix is then literally
``trace[:spans[i].end]`` -- the exact bytes the model produced -- so conditioning
on a prefix never introduces reconstruction artefacts (lost newlines, changed
spacing).  Any measured effect is therefore about the sentence, not about our
segmentation re-rendering the text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ABBREV = r"(?:e\.g|i\.e|etc|vs|cf|approx|Eq|Fig|Mr|Dr|St|no|No)"

_BOUNDARY = re.compile(r"""[.?!]["')\]]?(?=\s)""")

# Variable-width lookbehind is not supported, so abbreviations and decimals are
# rejected by inspecting the text before the candidate boundary instead.
# Only an *inter-digit* period is a decimal point ("3.14"); a period after a
# trailing digit ("the answer is 2.") is a genuine boundary.  Getting this
# backwards silently merged sentences in an earlier version.
_NOT_A_BOUNDARY = re.compile(r"(?:\b[A-Z]|\b" + _ABBREV + r")$")


def _is_real_boundary(text: str, dot: int) -> bool:
    """``dot`` indexes the terminator character; decide if it ends a sentence."""
    if text[dot] != ".":
        return True  # '?' and '!' are unambiguous here
    if dot + 1 < len(text) and text[dot + 1].isdigit():
        return False  # 3.14
    return _NOT_A_BOUNDARY.search(text[:dot]) is None


_HARD_BREAK = re.compile(r"\n\s*\n+")

# These models do not write prose paragraphs; they write outlines.  A single
# newline followed by a bullet or a number is the real unit boundary, and
# treating it as ordinary whitespace produced units like
#   "Here's a thinking process to solve the problem: 1."
# where the list marker for the *next* item was swallowed by the previous one.
# Splitting on the marker instead gives one reasoning step per unit, which is
# what the sentence-level analysis is supposed to be about.
_LIST_ITEM = re.compile(
    r"""
    \n[ \t]*                       # start of a line
    (?:
        [-*+•]\s              # bullet
      | \d{1,2}[.)]\s              # 1.  or  1)
      | \#{1,6}\s                  # markdown heading
      | \*\*[^*\n]{1,60}\*\*[:.]?  # a bolded step label, e.g. **Analyse f(x):**
    )
    """,
    re.VERBOSE,
)

_MATH_BLOCK = re.compile(
    r"\$\$.*?\$\$|\\\[.*?\\\]|\\begin\{(\w+\*?)\}.*?\\end\{\1\}",
    re.DOTALL,
)


@dataclass(frozen=True)
class Sentence:
    """One sentence, located in the trace it came from."""

    index: int
    start: int   # inclusive, into the original trace
    end: int     # exclusive; trace[:end] is a valid conditioning prefix
    text: str

    def __len__(self) -> int:
        return len(self.text)


def _masked_positions(text: str) -> list[bool]:
    """True where a character sits inside a display-math block."""
    mask = [False] * len(text)
    for m in _MATH_BLOCK.finditer(text):
        for k in range(m.start(), m.end()):
            mask[k] = True
    return mask


def split_sentences(text: str, min_chars: int = 12) -> list[Sentence]:
    """Segment ``text`` into sentences, returned as spans into ``text``.

    Boundaries are sentence terminators outside display math, plus blank lines.
    Fragments shorter than ``min_chars`` are absorbed into the previous sentence:
    a stray ``"Ok."`` is not an independent unit of reasoning, and treating it as
    one inflates the sentence count and so distorts normalised position -- which
    is the very variable this project is about.
    """
    in_math = _masked_positions(text)

    # The period in a list marker ("1.") is not a sentence terminator.  Without
    # this mask the ordinary boundary rule cut *after* the marker as well as
    # before it, leaving a two-character fragment that then merged backwards and
    # glued the next item's number onto the previous unit.
    in_marker = [False] * len(text)
    for m in _LIST_ITEM.finditer(text):
        for k in range(m.start(), m.end()):
            in_marker[k] = True

    cuts: set[int] = set()
    for m in _BOUNDARY.finditer(text):
        if in_math[m.start()] or in_marker[m.start()]:
            continue
        if _is_real_boundary(text, m.start()):
            cuts.add(m.end())
    for m in _HARD_BREAK.finditer(text):
        if not in_math[m.start()]:
            cuts.add(m.start())
    # Cut *before* the list marker, not after it, so the marker belongs to the
    # item it introduces.
    for m in _LIST_ITEM.finditer(text):
        if not in_math[m.start()]:
            cuts.add(m.start())

    ordered = sorted(cuts | {len(text)})

    raw: list[tuple[int, int]] = []
    prev = 0
    for cut in ordered:
        if cut <= prev:
            continue
        raw.append((prev, cut))
        prev = cut

    # Absorb leading whitespace into the preceding span so that
    # trace[:end] tiles the trace exactly with no gaps.
    merged: list[list[int]] = []
    for start, end in raw:
        body = text[start:end].strip()
        if not body:
            if merged:
                merged[-1][1] = end
            continue
        if merged and len(body) < min_chars:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    return [
        Sentence(index=i, start=s, end=e, text=text[s:e].strip())
        for i, (s, e) in enumerate(merged)
    ]


def prefix(text: str, sentences: list[Sentence], i: int) -> str:
    """The trace up to and including sentence ``i`` (``i = -1`` gives ``""``).

    This is an exact slice of the model's own output, so it is in-distribution
    by construction.
    """
    if i < 0:
        return ""
    return text[: sentences[i].end]
