"""Caller verification helpers: what the caller said -> an order number and a last name,
and a tolerant last-name match (case, accents, punctuation, one speech-to-text slip).

The check itself runs in the Support Desk (`verify_caller`), so a caller can't skip it.
"""

from __future__ import annotations

import re
import unicodedata

from callie.speech.text import normalize_transcript

_FILLER = re.compile(
    r"\b(?:my|the|it's|it is|its|that's|that is|this is|i'm|i am|yes|yeah|so|um|uh|er|okay|ok|sure|"
    r"last|family|sur|name|surname|is|spelled|spelt|spell|on|account|the account|please)\b", re.IGNORECASE)


def _letters(text: str) -> str:
    plain = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in plain if ch.isalpha() or ch == " ").lower()


def normalize_name(text: str) -> str:
    """'Óscar  Godoy-Ruiz' -> 'oscar godoyruiz'"""
    return " ".join(_letters(text.replace("-", "").replace("'", "")).split())


def last_name_of(full_name: str) -> str:
    """The family name of a customer record: everything after the first given name."""
    words = normalize_name(full_name).split()
    return " ".join(words[1:]) if len(words) > 1 else (words[0] if words else "")


def edit_distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


# Letters that sound alike on a phone line (the "E-set" and friends). V/C, D/T and B/P
# were the confusions measured in P0.
_LETTER_GROUPS = ("bcdegptvz", "ajk", "fsx", "mn", "iy", "quw")
_LETTER_CLASS = {ch: group[0] for group in _LETTER_GROUPS for ch in group}


def letters_match(said: str, actual: str) -> bool:
    """A spelled name against the record, letter by letter, sound-alike letters allowed."""
    if len(said) != len(actual) or len(said) < 3:
        return False
    return all(a == b or _LETTER_CLASS.get(a, a) == _LETTER_CLASS.get(b, b) for a, b in zip(said, actual))


def names_match(spoken: str, actual: str, *, spelled: bool = False) -> bool:
    """Does the caller's last name match the record's? Exact after normalizing, or one edit
    for names of 5+ letters; compound names match on the whole or on their last part.
    A spelled name may also differ in sound-alike letters (V for C, T for D): measured,
    this rescues the phone's letter confusions while wrong names stay at ~0.03% per guess."""
    said = normalize_name(spoken).replace(" ", "")
    if not said:
        return False
    wanted = normalize_name(actual)
    for candidate in {wanted.replace(" ", ""), wanted.split()[-1] if wanted else ""}:
        if not candidate:
            continue
        if said == candidate or (len(candidate) >= 5 and edit_distance(said, candidate) <= 1):
            return True
        if spelled and letters_match(said, candidate):
            return True
    return False


_SPELLING = re.compile(r"\b(?:it'?s|it is|that'?s|that is|spelled|spelt|spelling|spells|spell|as in)\b", re.IGNORECASE)


def extract_last_name(utterance: str) -> str:
    """'My last name is Virk.' -> 'Virk'; "It's spelled V-I-R-K." / 'V, I, R, K' -> 'VIRK'."""
    text = _SPELLING.sub(" ", utterance.strip())
    tokens = [t for t in re.split(r"[\s,.;:!?-]+", text) if t]
    letters = [t for t in tokens if len(t) == 1 and t.isalpha()]
    if len(letters) >= 3 and len(letters) >= len(tokens) - 3:  # spelled out, maybe with a few filler words
        return "".join(letters).upper()
    words = re.findall(r"[^\W\d_][^\W\d_'-]*", _FILLER.sub(" ", text))
    return " ".join(words[-3:])


# How whisper sometimes writes a spoken letter ("vee" -> "the"); sound-alike matching does the rest.
_LETTER_WORDS = {"a": "A", "ay": "A", "be": "B", "bee": "B", "see": "C", "sea": "C", "cee": "C", "the": "D",
                 "dee": "D", "e": "E", "ee": "E", "eff": "F", "gee": "G", "aitch": "H", "eye": "I", "jay": "J",
                 "kay": "K", "okay": "K", "el": "L", "ell": "L", "em": "M", "en": "N", "oh": "O", "pee": "P",
                 "pea": "P", "cue": "Q", "queue": "Q", "are": "R", "ar": "R", "es": "S", "ess": "S", "tea": "T",
                 "tee": "T", "you": "U", "vee": "V", "ex": "X", "why": "Y", "zee": "Z", "zed": "Z"}


def extract_spelling(utterance: str) -> str:
    """A spelled last name, when the caller was asked to spell it: "V-I-R-K", "The I.R.K."
    (whisper's "the" for "vee") -> letters; falls back to the plain name extraction."""
    text = _SPELLING.sub(" ", utterance.strip())
    tokens = [t for t in re.split(r"[\s,.;:!?-]+", text) if t]
    letters = [t.upper() if len(t) == 1 and t.isalpha() else _LETTER_WORDS.get(t.lower(), "") for t in tokens]
    letters = [x for x in letters if x]
    if len(letters) >= 3 and len(letters) >= len(tokens) - 3:
        return "".join(letters)
    return extract_last_name(utterance)


def extract_order_number(utterance: str) -> int | None:
    """'My order number is five five one two' -> 5512 (spoken or typed digits)."""
    text = normalize_transcript(utterance)
    if not re.search(r"\border\b|\bnumber\b", text, re.IGNORECASE):
        text = normalize_transcript(f"order {text}")  # a bare "five five one two"
    found = re.findall(r"\b\d{2,7}\b", text)
    return int(found[0]) if found else None
