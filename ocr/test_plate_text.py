"""Tests for plate_text.py.  Run:  python -m pytest ocr  (or: python ocr/test_plate_text.py)"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from plate_text import basic_clean, fix_plate, is_valid_plate, format_plate

HERE = os.path.dirname(__file__)
LABELS = os.path.join(HERE, "..", "data", "plates", "labels.csv")


def test_basic_clean():
    assert basic_clean("TR 34 PBL-657") == "34PBL657"
    assert basic_clean("06 dJp 36.") == "06DJP36"


def test_all_ground_truth_plates_are_valid():
    with open(LABELS) as f:
        for row in csv.DictReader(f):
            assert is_valid_plate(row["plate_text"]), row


def test_digit_letter_fixes():
    cases = {
        "34PBL6S7": "34PBL657",   # S -> 5 in number part
        "O6BUZ420": "06BUZ420",   # O -> 0 in province code
        "34D6R89": "34DGR89",     # 6 -> G in letter part
        "6GABF936": "66ABF936",   # G -> 6
        "34KJZ1g5": "34KJZ165",   # g -> G -> 6 in a digit spot
        "34M37O9": "34M3709",
        "8lADR785": "81ADR785",
    }
    for raw, want in cases.items():
        got, ok = fix_plate(raw)
        assert ok and got == want, (raw, got, want)


def test_correct_plates_unchanged():
    with open(LABELS) as f:
        for row in csv.DictReader(f):
            assert fix_plate(row["plate_text"]) == (row["plate_text"], True)


def test_extra_char_dropped():
    assert fix_plate("I34PBL657") == ("34PBL657", True)
    assert fix_plate("34PBL657|") == ("34PBL657", True)


def test_invalid_stays_invalid():
    assert fix_plate("HELLO")[1] is False
    assert fix_plate("")[1] is False
    assert fix_plate("99ABC123")[1] is False      # no province 99


def test_format():
    assert format_plate("34PBL657") == "34 PBL 657"




def test_ambiguous_valid_reading_is_kept():
    # "34 PB 1657" is itself a valid plate, so we do not change it
    assert fix_plate("34PB1657") == ("34PB1657", True)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
