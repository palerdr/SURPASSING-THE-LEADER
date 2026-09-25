"""A basic slur filter for names that every player reads on the leaderboard.

The filter refuses slurs and a few hate terms; ordinary profanity passes. It is
a word list, so it stops casual abuse and no determined writer. The terms are
stored in rot13 so that no slur shows in the source, a diff, or a code search.
"""

from __future__ import annotations

import codecs
import re
import unicodedata
from itertools import groupby

# Refused wherever the letters occur inside a word.
_ANYWHERE = (
    "avttre", "avttn", "avtyrg", "snttbg", "jrgonpx", "gbjryurnq",
    "enturnq", "puvatpubat", "wvtnobb", "cbepuzbaxrl", "mvccreurnq",
    "furznyr", "ergneq", "uvgyre",
)
# Refused as a whole word only, because ordinary words and names contain them
# ("raccoon", "spice", "Pakistan", "grape", "Benazir").
_WHOLE_WORD = (
    "fcvp", "pbba", "puvax", "tbbx", "cnxv", "snt", "xvxr", "xlxr", "urro",
    "genaal", "genaavr", "ornare", "xxx", "anmv", "arbanmv", "encr",
    "encvfg", "encvat",
)
# Ordinary words that hold a refused term.
_ALLOWED = re.compile(r"snigger(s|ed|ing)?")
# Words this short may be one word spelled with gaps ("k k k", "n i g ...").
_SPELLED_OUT = 3

# Digits and symbols that stand in for letters, then Cyrillic and Greek
# letters that look like Latin ones.
_LOOKALIKES = str.maketrans(
    "013457@$!|" "\u0430\u0435\u043e\u0456\u0441\u0440\u0445\u0443\u043a\u0455\u0458\u0433"
    "\u03bf\u03b1\u03b9\u03b5\u03ba\u03c1\u03c4\u03c5\u03bd",
    "oieastasii" "aeoicpxyksjr" "oaiekptuv",
)


def _stretched(term: str) -> str:
    """A pattern for the term with any letter held longer: ``coon`` matches
    ``cooon`` and never ``con``, so a doubled letter stays part of the word."""

    return "".join(
        f"{letter}{{{len(list(run))},}}" for letter, run in groupby(term)
    )


_TERMS = [codecs.decode(term, "rot13") for term in _ANYWHERE]
_WORDS = [codecs.decode(term, "rot13") for term in _WHOLE_WORD]
_INSIDE = re.compile("|".join(_stretched(term) for term in _TERMS))
_ENTIRE = re.compile("(?:" + "|".join(_stretched(term) for term in _WORDS) + ")s?")


def _refused(word: str) -> bool:
    if _ALLOWED.fullmatch(word):
        return False
    return bool(_ENTIRE.fullmatch(word) or _INSIDE.search(word))


def is_offensive(name: str) -> bool:
    """Whether a display name holds a refused term."""

    folded = unicodedata.normalize("NFKD", name).casefold().translate(_LOOKALIKES)
    words = [word for word in re.split(r"[^a-z]+", folded) if word]
    if any(_refused(word) for word in words):
        return True
    # Join each run of short words, so a term spelled with gaps reads as one
    # word. Long words stay apart: "Dana Zimmer" must not read as one string.
    for short, run in groupby(words, key=lambda word: len(word) <= _SPELLED_OUT):
        if short and _refused("".join(run)):
            return True
    return False
