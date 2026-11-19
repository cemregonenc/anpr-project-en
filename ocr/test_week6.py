"""Tests for Week 6 voting / de-duplication. Run: python ocr/test_week6.py"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from plate_vote import Voter, dedupe_tracks


def test_valid_reading_beats_more_frequent_invalid():
    v = Voter()
    for _ in range(3):
        v.add(1, "34PBL6S7", 0.9, False)
    v.add(1, "34PBL657", 0.7, True)
    v.add(1, "34PBL657", 0.8, True)
    b = v.best(1)
    assert b["text"] == "34PBL657" and b["valid"] and b["votes"] == 2 and b["readings"] == 5, b


def test_confidence_breaks_ties_between_valid_readings():
    v = Voter()
    v.add(2, "06BUZ420", 0.95, True)
    v.add(2, "06BUZ428", 0.40, True)
    assert v.best(2)["text"] == "06BUZ420"


def test_no_readings():
    assert Voter().best(9)["text"] is None


def test_dedupe_merges_same_plate_from_two_tracks():
    rows = [
        {"track_id": 1, "plate": "34 PBL 657", "plate_valid": True, "frames_seen": 8, "first_seen_s": 0.0, "last_seen_s": 3.0},
        {"track_id": 5, "plate": "34 PBL 657", "plate_valid": True, "frames_seen": 3, "first_seen_s": 3.5, "last_seen_s": 4.5},
        {"track_id": 2, "plate": None, "plate_valid": None, "frames_seen": 4, "first_seen_s": 1.0, "last_seen_s": 2.0},
        {"track_id": 3, "plate": None, "plate_valid": None, "frames_seen": 4, "first_seen_s": 1.5, "last_seen_s": 2.5},
    ]
    out = dedupe_tracks(rows)
    assert len(out) == 3, out                      # two cars without plates stay separate
    car = [r for r in out if r["plate"]][0]
    assert car["frames_seen"] == 11 and car["first_seen_s"] == 0.0 and car["last_seen_s"] == 4.5
    assert car["merged_track_ids"] == [5]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
