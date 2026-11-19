"""Build a plate-text test set from a character-labelled YOLO dataset (Week 5, Builder).

Datasets like Roboflow `metehan-yasar/plaka-cript` label every character on the
plate (classes 0-9, A-Z). Reading those boxes left-to-right (top row first for
two-line plates) gives the correct plate text for free, so we get hundreds of
labelled plates instead of our 50 hand-typed ones.

    python ocr/build_text_testset.py --data datasets/plaka-cript/data.yaml --out plates_v2

Output: plates_v2/<file>.jpg + plates_v2/labels.csv
    filename, plate_text, source_image, split, n_chars
Only plates that match the Turkish pattern are kept (use --keep-invalid to keep all).
"""
import argparse
import csv
import os
import sys

import cv2
import yaml

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
from plate_text import is_valid_plate  # noqa: E402
from merge_datasets import IMG_EXT, SPLIT_KEYS, _labels_dir, _names, _resolve_images_dir  # noqa: E402


def read_chars(label_path, names):
    """-> list of (char, cx, cy, w, h) in relative coords."""
    out = []
    if not os.path.exists(label_path):
        return out
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            cid = int(float(p[0]))
            if cid >= len(names):
                continue
            ch = str(names[cid]).strip().upper()
            if len(ch) != 1 or not ch.isalnum():      # skip 'plate', 'TR' and other non-character classes
                continue
            cx, cy, w, h = map(float, p[1:5])
            out.append((ch, cx, cy, w, h))
    return out


def chars_to_text(chars):
    """Group characters into rows (top to bottom), read each row left to right."""
    if not chars:
        return ""
    hs = sorted(c[4] for c in chars)
    med_h = hs[len(hs) // 2]
    rows = []
    for c in sorted(chars, key=lambda c: c[2]):
        if rows and abs(c[2] - rows[-1][-1][2]) < 0.6 * med_h:
            rows[-1].append(c)
        else:
            rows.append([c])
    return "".join(c[0] for row in rows for c in sorted(row, key=lambda c: c[1]))


def crop_box(chars, w, h, pad_x=0.15, pad_y=0.35):
    """Pixel box around all characters, with padding (relative to the text block size)."""
    x1 = min(c[1] - c[3] / 2 for c in chars) * w
    x2 = max(c[1] + c[3] / 2 for c in chars) * w
    y1 = min(c[2] - c[4] / 2 for c in chars) * h
    y2 = max(c[2] + c[4] / 2 for c in chars) * h
    bw, bh = x2 - x1, y2 - y1
    return (max(0, int(x1 - pad_x * bw)), max(0, int(y1 - pad_y * bh)),
            min(w, int(x2 + pad_x * bw)), min(h, int(y2 + pad_y * bh)))


def build(data_yaml, out_dir, keep_invalid=False, min_chars=5):
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    names = _names(cfg)
    os.makedirs(out_dir, exist_ok=True)
    rows, skipped = [], {"no_chars": 0, "invalid": 0}
    for key, split in SPLIT_KEYS.items():
        img_dir = _resolve_images_dir(data_yaml, cfg, key)
        if not img_dir:
            continue
        lab_dir = _labels_dir(img_dir)
        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith(IMG_EXT):
                continue
            chars = read_chars(os.path.join(lab_dir, os.path.splitext(fn)[0] + ".txt"), names)
            if len(chars) < min_chars:
                skipped["no_chars"] += 1
                continue
            text = chars_to_text(chars)
            if not keep_invalid and not is_valid_plate(text):
                skipped["invalid"] += 1
                continue
            img = cv2.imread(os.path.join(img_dir, fn))
            if img is None:
                continue
            x1, y1, x2, y2 = crop_box(chars, img.shape[1], img.shape[0])
            name = f"{split}_{os.path.splitext(fn)[0][:40]}.jpg"
            cv2.imwrite(os.path.join(out_dir, name), img[y1:y2, x1:x2])
            rows.append({"filename": name, "plate_text": text, "source_image": fn,
                         "split": split, "n_chars": len(chars)})
    with open(os.path.join(out_dir, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "plate_text", "source_image", "split", "n_chars"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} plates written to {out_dir}  (skipped: {skipped})")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="data.yaml of the character dataset")
    ap.add_argument("--out", default="plates_v2")
    ap.add_argument("--keep-invalid", action="store_true")
    a = ap.parse_args()
    build(a.data, a.out, a.keep_invalid)
