"""Extracting and normalising final answers from reasoning traces.

Every downstream number depends on this file, so it is deliberately boring and
conservative.  Two failure modes we care about:

* **False disagreement.**  ``\\dfrac{1}{2}`` and ``\\frac12`` and ``0.5`` are the
  same answer.  If we call them different, answer distributions look more
  dispersed than they are and *every* importance score is inflated.
* **Silent truncation.**  A rollout that runs out of budget mid-thought has no
  answer.  Scoring it as "wrong" conflates "the model changed its mind" with
  "we cut it off", which is exactly the kind of thing that manufactures a
  positional effect (later prefixes leave less room to run over).  We give it
  its own outcome label, ``None``, and report its rate.
"""

from __future__ import annotations

import re
from fractions import Fraction

_BOXED = re.compile(r"\\boxed\s*\{")


def extract_boxed(text: str) -> str | None:
    """Return the content of the *last* ``\\boxed{...}``, brace-matched."""
    last = None
    for m in _BOXED.finditer(text):
        i = m.end()
        depth = 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            last = text[m.end() : i - 1]
    return last


_STRIP_PAIRS = [
    (r"\\left", ""), (r"\\right", ""), (r"\\!", ""), (r"\\,", ""),
    (r"\;", ""), (r"\\ ", " "), (r"\\$", ""), (r"\$", ""),
    (r"\\%", ""), (r"%", ""),
    # Keep the *content* of \text{...}: MATH answers are often words
    # ("blue", "even"), and dropping them made every text answer compare equal.
    (r"\\text\s*\{([^}]*)\}", r"\1"), (r"\\mbox\s*\{([^}]*)\}", r"\1"),
    (r"\\dfrac", r"\\frac"), (r"\\tfrac", r"\\frac"), (r"\\cdot", "*"),
    (r"\\times", "*"), (r"\s+", ""),
]

def _balanced(s: str) -> bool:
    depth = 0
    for ch in s:
        depth += (ch == "(") - (ch == ")")
        if depth < 0:
            return False
    return depth == 0


_FRAC = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
_FRAC_SHORT = re.compile(r"\\frac(\d)(\d)")


def normalise(ans: str | None) -> str | None:
    """Canonicalise a boxed answer so equivalent spellings compare equal."""
    if ans is None:
        return None
    s = ans.strip()
    for pat, rep in _STRIP_PAIRS:
        s = re.sub(pat, rep, s)
    s = _FRAC_SHORT.sub(r"\\frac{\1}{\2}", s)
    s = _FRAC.sub(r"(\1)/(\2)", s)
    s = s.rstrip(".").lstrip("+")
    # Drop a redundant outer paren pair, but never on tuples/intervals, where
    # the parens are part of the answer: (1,2) must not become 1,2.
    while (
        len(s) > 2 and s[0] == "(" and s[-1] == ")"
        and "," not in s and ";" not in s
        and _balanced(s[1:-1])
    ):
        s = s[1:-1]
    s = re.sub(r"^0+(?=\d)", "", s)
    if s.endswith(".0"):
        s = s[:-2]

    # Reduce a plain rational to lowest terms; leave anything else alone.
    m = re.fullmatch(r"(-?)\(?(-?\d+)\)?/\(?(-?\d+)\)?", s)
    if m:
        try:
            sign = -1 if m.group(1) == "-" else 1
            f = sign * Fraction(int(m.group(2)), int(m.group(3)))
            s = str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
        except ZeroDivisionError:
            pass
    else:
        try:  # 2.50 -> 5/2 -> "5/2"; keeps 0.5 and 1/2 in the same class
            if re.fullmatch(r"-?\d+\.\d+", s):
                f = Fraction(s).limit_denominator(10**6)
                s = str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
            elif re.fullmatch(r"-?\d+", s):
                s = str(int(s))
        except (ValueError, ZeroDivisionError):
            pass
    return s or None


def final_answer(text: str) -> str | None:
    """The normalised final answer of a completion, or ``None`` if it has none."""
    return normalise(extract_boxed(text))
