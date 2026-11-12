"""Clean and correct OCR text for Turkish number plates.

Turkish plate format:  2 digits (province 01-81) + 1-3 letters + 2-5 digits
    34 M 3709    -> 1 letter  + 4-5 digits
    09 KS 627    -> 2 letters + 3-4 digits
    81 ADR 785   -> 3 letters + 2-3 digits

The project plan says "letters then numbers, e.g. ABC-123". Our plates are
Turkish, so we use the Turkish pattern instead (agreed in Week 3).

Pipeline:  raw OCR text -> basic_clean() -> fix_plate() -> is_valid_plate()
"""
import re

# Province code 01-81, then the three allowed letter/digit layouts.
PLATE_RE = re.compile(
    r"^(0[1-9]|[1-7][0-9]|8[01])"
    r"([A-Z]\d{4,5}|[A-Z]{2}\d{3,4}|[A-Z]{3}\d{2,3})$"
)

# OCR often confuses these. Used only where the position says digit / letter.
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "T": "1",
            "Z": "2", "S": "5", "G": "6", "B": "8", "A": "4"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B", "4": "A"}

# (letters, min digits, max digits) for each valid layout
LAYOUTS = [(1, 4, 5), (2, 3, 4), (3, 2, 3)]


def basic_clean(text):
    """Uppercase, keep only A-Z and 0-9, drop the 'TR' from the blue band."""
    t = re.sub(r"[^A-Z0-9]", "", str(text).upper())
    if t.startswith("TR") and (len(t) >= 8 or t[2:3].isdigit()):
        t = t[2:]
    return t


def is_valid_plate(text):
    return bool(PLATE_RE.match(text or ""))


def _convert(part, table, want_digit):
    """Convert one part to digits or letters. Returns (text, n_changes) or None."""
    out, changes = [], 0
    for ch in part:
        ok = ch.isdigit() if want_digit else ch.isalpha()
        if ok:
            out.append(ch)
        elif ch in table:
            out.append(table[ch])
            changes += 1
        else:
            return None
    return "".join(out), changes


def _candidates(t):
    """Every way to read t as  2 digits + letters + digits, with its change count."""
    for n_letters, dmin, dmax in LAYOUTS:
        n_digits = len(t) - 2 - n_letters
        if not dmin <= n_digits <= dmax:
            continue
        parts = [
            _convert(t[:2], TO_DIGIT, True),
            _convert(t[2:2 + n_letters], TO_LETTER, False),
            _convert(t[2 + n_letters:], TO_DIGIT, True),
        ]
        if None in parts:
            continue
        text = "".join(p[0] for p in parts)
        if is_valid_plate(text):
            yield text, sum(p[1] for p in parts)


def fix_plate(text):
    """Return (best_text, valid).

    Tries every valid layout, fixes O->0, I->1, S->5 ... only where the
    position needs a digit (and the reverse where it needs a letter), and
    keeps the reading that needs the fewest changes. If nothing fits, it
    also tries dropping one extra character at the start or end (frame
    edges, stickers). If still nothing fits, returns the cleaned text with
    valid=False.
    """
    t = basic_clean(text)
    tries = [t]
    if len(t) > 7:
        tries += [t[1:], t[:-1]]
    best = None
    for i, s in enumerate(tries):
        for cand, changes in _candidates(s):
            score = changes + (1 if i else 0)        # dropping a char costs 1
            if best is None or score < best[1]:
                best = (cand, score)
    if best:
        return best[0], True
    return t, False


def format_plate(text):
    """'34PBL657' -> '34 PBL 657' (for display)."""
    m = re.match(r"^(\d{2})([A-Z]{1,3})(\d+)$", text or "")
    return f"{m[1]} {m[2]} {m[3]}" if m else (text or "")
