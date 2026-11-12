"""Crop number plates automatically with the trained plate detector (plate_v1 best.pt).

Usage (Colab or laptop):
    python models/crop_plates.py --weights models/plate_best.pt --source test_images/ --out auto_crops/

--source can be one image, a folder of images, or a glob ("images/*.jpg").
Writes one crop per detected plate plus crops.csv:
    crop_file, source_image, confidence, x1, y1, x2, y2
"""
import argparse
import csv
import glob
import os

import cv2

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(source):
    if os.path.isdir(source):
        files = [os.path.join(source, f) for f in sorted(os.listdir(source))]
    elif any(ch in source for ch in "*?["):
        files = sorted(glob.glob(source))
    else:
        files = [source]
    return [f for f in files if f.lower().endswith(IMG_EXT)]


def pad_box(x1, y1, x2, y2, w, h, pad):
    """Grow the box by `pad` (fraction of its size) so the plate edges are not cut."""
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    return (max(0, int(x1 - dx)), max(0, int(y1 - dy)),
            min(w, int(x2 + dx)), min(h, int(y2 + dy)))


def plate_boxes(model, img, conf=0.25):
    """Run the plate detector on one BGR image -> list of (x1, y1, x2, y2, conf).

    Works with a plate-only model, and with a multi-class model that has a 'plate' class.
    """
    res = model.predict(img, conf=conf, verbose=False)[0]
    names = res.names
    plate_ids = [i for i, n in names.items() if "plate" in n.lower()]
    if not plate_ids and len(names) == 1:      # single-class model: every box is a plate
        plate_ids = list(names)
    out = []
    for b in res.boxes:
        cls = int(b.cls)
        if cls not in plate_ids:
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        out.append((x1, y1, x2, y2, float(b.conf)))
    return out


def crop_plates(weights, source, out_dir, conf=0.25, pad=0.05):
    from ultralytics import YOLO
    model = YOLO(weights)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    images = list_images(source)
    for path in images:
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        stem = os.path.splitext(os.path.basename(path))[0]
        for k, (x1, y1, x2, y2, c) in enumerate(plate_boxes(model, img, conf)):
            px1, py1, px2, py2 = pad_box(x1, y1, x2, y2, w, h, pad)
            if px2 - px1 < 2 or py2 - py1 < 2:
                continue
            name = f"{stem}_plate{k}.jpg"
            cv2.imwrite(os.path.join(out_dir, name), img[py1:py2, px1:px2])
            rows.append({"crop_file": name, "source_image": os.path.basename(path),
                         "confidence": round(c, 3), "x1": px1, "y1": py1, "x2": px2, "y2": py2})
    with open(os.path.join(out_dir, "crops.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_file", "source_image", "confidence", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(images)} images -> {len(rows)} plate crops in {out_dir}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="models/plate_best.pt")
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="auto_crops")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--pad", type=float, default=0.05)
    a = ap.parse_args()
    crop_plates(a.weights, a.source, a.out, a.conf, a.pad)
