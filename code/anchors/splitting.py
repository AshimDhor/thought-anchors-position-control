from __future__ import annotations

import re
from dataclasses import dataclass

_ABBREV = r"(?:e\.g|i\.e|etc|vs|cf|approx|Eq|Fig|Mr|Dr|St|no|No)"

_BOUNDARY = re.compile(r"""[.?!]["')\]]?(?=\s)""")


_NOT_A_BOUNDARY = re.compile(r"(?:\b[A-Z]|\b" + _ABBREV + r")$")


def _is_real_boundary(text: str, dot: int) -> bool:
    """``dot`` indexes the terminator character; decide if it ends a sentence."""
    if text[dot] != ".":
        return True  # '?' and '!' are unambiguous here
    if dot + 1 < len(text) and text[dot + 1].isdigit():
        return False  # 3.14
    return _NOT_A_BOUNDARY.search(text[:dot]) is None


_HARD_BREAK = re.compile(r"\n\s*\n+")


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

    index: int
    start: int   # inclusive, into the original trace
    end: int     # exclusive; trace[:end] is a valid conditioning prefix
    text: str

    def __len__(self) -> int:
        return len(self.text)


def _masked_positions(text: str) -> list[bool]:
    mask = [False] * len(text)
    for m in _MATH_BLOCK.finditer(text):
        for k in range(m.start(), m.end()):
            mask[k] = True
    return mask


def split_sentences(text: str, min_chars: int = 12) -> list[Sentence]:

    in_math = _masked_positions(text)


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

    if i < 0:
        return ""
    return text[: sentences[i].end]
