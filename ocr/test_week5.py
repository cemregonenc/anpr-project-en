"""Tests for Week 5 tools. Run: python ocr/test_week5.py"""
import csv, os, sys, tempfile
import cv2, numpy as np, yaml
sys.path.insert(0, os.path.dirname(__file__))
from build_text_testset import build, chars_to_text
from plate_reader import ENHANCE_MODES, preprocess

CHARS = list("0123456789ABCDEFGHIJKLMNOPRSTUVWXYZ")


def _fake_plate(path, rows):
    """Draw a plate with the given text rows and return YOLO char lines."""
    w, h = 360, 80 * len(rows)
    img = np.full((h, w, 3), 255, np.uint8)
    lines = []
    for r, text in enumerate(rows):
        x = 20
        for ch in text:
            y = 60 + 80 * r
            cv2.putText(img, ch, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
            cx, cy = (x + 15) / w, (y - 20) / h
            lines.append(f"{CHARS.index(ch)} {cx:.4f} {cy:.4f} {30 / w:.4f} {45 / h:.4f}")
            x += 40
    cv2.imwrite(path, img)
    return lines


def test_chars_to_text_two_rows():
    chars = [("6", .6, .7, .1, .3), ("1", .2, .2, .1, .3), ("D", .5, .25, .1, .3),
             ("8", .4, .7, .1, .3), ("1", .3, .2, .1, .3), ("F", .6, .22, .1, .3), ("6", .5, .72, .1, .3)]
    assert chars_to_text(chars) == "11DF" + "866"


def test_build_from_char_dataset():
    d = tempfile.mkdtemp()
    os.makedirs(f"{d}/train/images"); os.makedirs(f"{d}/train/labels")
    samples = {"a": ["34PBL657"], "b": ["11DF", "686"], "c": ["HELLO"]}
    for k, rows in samples.items():
        open(f"{d}/train/labels/{k}.txt", "w").write("\n".join(_fake_plate(f"{d}/train/images/{k}.jpg", rows)))
    yaml.safe_dump({"train": "../train/images", "names": CHARS}, open(f"{d}/data.yaml", "w"))
    out = f"{d}/out"
    rows = build(f"{d}/data.yaml", out)
    got = {r["source_image"]: r["plate_text"] for r in rows}
    assert got == {"a.jpg": "34PBL657", "b.jpg": "11DF686"}, got   # HELLO is not a Turkish plate
    assert len(list(csv.DictReader(open(f"{out}/labels.csv")))) == 2
    assert all(cv2.imread(f"{out}/{r['filename']}") is not None for r in rows)


def test_enhance_modes_keep_shape():
    img = cv2.imread(os.path.join(os.path.dirname(__file__), "..", "data", "plates", "plate_01.jpg"))
    for m in ENHANCE_MODES:
        out = preprocess(img, True, 2.0, m)
        assert out.shape == (img.shape[0] * 2, img.shape[1] * 2, 3), (m, out.shape)


def test_fallback_only_when_invalid():
    import types
    sys.modules.setdefault("paddle", types.SimpleNamespace(set_flags=lambda d: None))
    calls = []

    class FakeOCR:
        def __init__(self, **k): pass
        def predict(self, img):
            calls.append(1)
            # first read (no enhance) is wrong, the sharpened read is right
            text = "34 P8" if len(calls) == 1 else "34 PBL 657"
            return [{"rec_texts": [text], "rec_scores": [0.8], "rec_boxes": np.array([[0, 0, 10, 10]])}]
    sys.modules["paddleocr"] = types.SimpleNamespace(PaddleOCR=FakeOCR)
    from plate_reader import PlateReader
    img = np.full((40, 120, 3), 200, np.uint8)
    r = PlateReader(fallback=("sharpen", "clahe"))
    out = r.read(img)
    assert out["text"] == "34PBL657" and out["valid"] and out["mode"] == "sharpen", out
    assert len(calls) == 2                      # stopped after the first valid fallback
    calls.clear()
    assert PlateReader().read(img)["valid"] is False and len(calls) == 1   # no fallback by default


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
