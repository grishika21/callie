"""Text on its way to and from speech.

* `speakable(reply)`: what Piper reads aloud. Markdown, tables and links are dropped,
  emails are never read out, ids are read digit by digit ("order 5 5 1 2"), and only the
  first two sentences are spoken (the full reply stays on screen).
* `normalize_transcript(text)`: what whisper heard, with spoken or mangled ids turned
  back into numbers: "order five five one two", "order fifty five twelve",
  "order 55 12", "order 12.03" and "order 6-0-2-0" all become digits.
"""

from __future__ import annotations

import re

from callie.audit import EMAIL_RE

# --- reply -> speech ---------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"“(A-Z0-9])")
_URL = re.compile(r"https?://\S+?(?=[.,;!?]?(?:\s|$))")
_SPOKEN_ID = re.compile(r"\b(order|ticket|customer)s?(\s+(?:number\s+|no\.?\s+|#\s*)?)(\d{2,7})\b", re.IGNORECASE)
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December")


def speakable(text: str, max_sentences: int = 2) -> str:
    """A reply made fit for reading aloud (see the module docstring)."""
    lines = []
    for line in text.replace("```", "\n").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("|") or set(stripped) <= set("-=_*#|: "):
            continue  # blank lines, table rows, rules
        stripped = re.sub(r"^(?:#{1,6}\s+|[-*•]\s+|\d+[.)]\s+)", "", stripped)  # headings and list markers
        lines.append(stripped if stripped.endswith((".", "!", "?", ":")) else stripped + ".")
    text = " ".join(lines)
    text = re.sub(r"[*_`~]+", "", text)  # emphasis and code marks
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [label](link) -> label
    text = re.sub(r"\s*\((?:customer|order|ticket|id)[^)]*\)", "", text, flags=re.IGNORECASE)  # "(customer 1139)": on screen
    text = _URL.sub("a link on screen", text)
    text = EMAIL_RE.sub(lambda m: f"an address at {m.group(2)}", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]\*{3}@([A-Za-z0-9.-]+)", r"an address at \1", text)  # already masked
    text = _ISO_DATE.sub(lambda m: _date_words(m), text)
    text = _SPOKEN_ID.sub(lambda m: f"{m.group(1)}{m.group(2)}{' '.join(m.group(3))}", text)
    sentences = [s.strip() for s in _SENTENCE_END.split(" ".join(text.split())) if s.strip()]
    return " ".join(sentences[:max_sentences])


def _date_words(match: re.Match[str]) -> str:
    year, month, day = (int(g) for g in match.groups())
    return f"{_MONTHS[month - 1]} {day}, {year}" if 1 <= month <= 12 and 1 <= day <= 31 else match.group(0)


# --- speech -> text ------------------------------------------------------------------------

_DIGITS = {"zero": 0, "oh": 0, "o": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
          "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
         "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000}
_REPEAT = {"double": 2, "triple": 3}
_ID_WORDS = {"order", "orders", "ticket", "tickets", "customer", "customers", "id"}
_FILLER = {"number", "no", "no.", "#", "is", "id"}
_NUMERIC = re.compile(r"^#?\d+(?:[.\-]\d+)*$")
_TOKEN = re.compile(r"[A-Za-z]+(?:'[a-z]+)?|#?\d+(?:[.\-]\d+)*|[^\sA-Za-z\d]+|\s+")


def _is_number_word(word: str, in_id: bool) -> bool:
    w = word.lower()
    if w in ("oh", "o"):
        return in_id  # "oh" is only a zero inside an id
    return w in _DIGITS or w in _TEENS or w in _TENS or w in _SCALES or w in _REPEAT or bool(_NUMERIC.match(w))


def _run_to_digits(words: list[str]) -> str | None:
    """Spoken number words (and digit tokens) of one id, as a digit string."""
    lower = [w.lower() for w in words]
    if any(w in _SCALES for w in lower):  # "five thousand five hundred twelve": an ordinary number
        total = current = 0
        for w in lower:
            if w in _SCALES:
                current = max(current, 1) * _SCALES[w]
                if w == "thousand":
                    total, current = total + current, 0
            elif w in _DIGITS or w in _TEENS or w in _TENS:
                current += _DIGITS.get(w) or _TEENS.get(w) or _TENS.get(w) or 0
            elif w.isdigit():
                current += int(w)
            else:
                return None
        return str(total + current)
    out, i = [], 0
    while i < len(lower):
        w = lower[i]
        if w in _REPEAT and i + 1 < len(lower) and lower[i + 1] in _DIGITS:
            out.append(str(_DIGITS[lower[i + 1]]) * _REPEAT[w])
            i += 2
        elif w in _TENS:
            unit = lower[i + 1] if i + 1 < len(lower) else ""
            if unit in _DIGITS and _DIGITS[unit] > 0:
                out.append(str(_TENS[w] + _DIGITS[unit]))
                i += 2
            else:
                out.append(str(_TENS[w]))
                i += 1
        elif w in _TEENS:
            out.append(str(_TEENS[w]))
            i += 1
        elif w in _DIGITS:
            out.append(str(_DIGITS[w]))
            i += 1
        elif _NUMERIC.match(w):  # "12.03", "6-0-2-0", "778-1", "#5512": whisper's spellings of one id
            out.append(re.sub(r"\D", "", w))
            i += 1
        else:
            return None
    return "".join(out) or None


def normalize_transcript(text: str) -> str:
    """Spoken ids back into digits (see the module docstring). Other words are kept."""
    tokens = _TOKEN.findall(text)
    words = [i for i, t in enumerate(tokens) if not t.isspace()]
    out = list(tokens)
    k = 0
    while k < len(words):
        i = words[k]
        word = tokens[i].lower().rstrip(".,")
        if word in _ID_WORDS:
            # skip "number", "#", "is" after the keyword, then take the run of number words
            j = k + 1
            while j < len(words) and tokens[words[j]].lower() in _FILLER and not _NUMERIC.match(tokens[words[j]]):
                j += 1
            run_start, run = j, []
            while j < len(words) and _is_number_word(tokens[words[j]].strip(",."), in_id=True):
                run.append(tokens[words[j]].strip(",."))
                j += 1
            digits = _run_to_digits(run) if run else None
            if digits:
                digits = digits.lstrip("0") or "0"  # ids have no leading zeros ("ticket oh four two")
                first, last = words[run_start], words[j - 1]
                trailing = tokens[last][len(tokens[last].rstrip(",.")):]
                out[first:last + 1] = [digits + trailing] + [""] * (last - first)
                k = j
                continue
        elif _is_number_word(tokens[i], in_id=False) and tokens[i].lower() in _DIGITS:
            # three or more digit words in a row ("five five one two") are an id anywhere
            j, run = k, []
            while j < len(words) and tokens[words[j]].lower() in _DIGITS:
                run.append(tokens[words[j]])
                j += 1
            if len(run) >= 3:
                first, last = words[k], words[j - 1]
                out[first:last + 1] = ["".join(str(_DIGITS[w.lower()]) for w in run)] + [""] * (last - first)
                k = j
                continue
        k += 1
    return re.sub(r"[ \t]{2,}", " ", "".join(out)).strip()
