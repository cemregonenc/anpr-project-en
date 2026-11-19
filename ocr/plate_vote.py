"""Combine several readings of the same car into one answer (Week 6, Builder).

In a video the same plate is read in many frames, and some readings are wrong
("34PBL6S7", "34P8L657" ...). We keep the best one:
  1. readings that fit the Turkish plate pattern win over ones that do not;
  2. among those, the text with the highest total confidence wins
     (seen often AND read with high confidence).

    v = Voter()
    v.add(track_id=7, text="34PBL657", score=0.93, valid=True)
    v.best(7)  ->  {"text": "34PBL657", "valid": True, "votes": 1, "readings": 1, "confidence": 0.93}

dedupe_tracks() then makes sure each car is saved only once, even if the
tracker lost it for a moment and gave it a new id.
"""
from collections import defaultdict


class Voter:
    def __init__(self):
        self._r = defaultdict(lambda: defaultdict(lambda: [0, 0.0, False]))   # id -> text -> [n, sum_score, valid]

    def add(self, track_id, text, score=1.0, valid=None):
        if not text:
            return
        e = self._r[track_id][text]
        e[0] += 1
        e[1] += float(score or 0.0)
        e[2] = e[2] or bool(valid)

    def best(self, track_id):
        texts = self._r.get(track_id)
        if not texts:
            return {"text": None, "valid": None, "votes": 0, "readings": 0, "confidence": None}
        text, (n, total, valid) = max(texts.items(), key=lambda kv: (kv[1][2], kv[1][1], kv[1][0]))
        return {"text": text, "valid": valid, "votes": n,
                "readings": sum(v[0] for v in texts.values()),
                "confidence": round(total / n, 3)}

    def ids(self):
        return list(self._r)


def dedupe_tracks(rows):
    """Merge rows that share the same valid plate text (tracker split one car into two ids).

    rows: dicts with at least plate, plate_valid, frames_seen, first_seen_s, last_seen_s, track_id.
    The row seen in more frames is kept; times and frame counts are combined.
    """
    out, by_plate = [], {}
    for r in sorted(rows, key=lambda r: -r.get("frames_seen", 0)):
        key = r.get("plate") if r.get("plate_valid") else None
        if key and key in by_plate:
            keep = by_plate[key]
            keep["frames_seen"] += r.get("frames_seen", 0)
            keep["first_seen_s"] = min(keep["first_seen_s"], r["first_seen_s"])
            keep["last_seen_s"] = max(keep["last_seen_s"], r["last_seen_s"])
            keep.setdefault("merged_track_ids", []).append(r["track_id"])
            continue
        r = dict(r)
        out.append(r)
        if key:
            by_plate[key] = r
    return sorted(out, key=lambda r: r["first_seen_s"])
