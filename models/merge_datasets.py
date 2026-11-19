"""Merge several YOLO datasets into one plate-only dataset (Week 5, Vision).

Each source is a folder with a data.yaml (Roboflow "YOLOv8" export, FiftyOne
YOLOv5 export, or our own). Only plate boxes are kept and all become class 0
"plate". Duplicate images (same bytes) are dropped. Each source keeps its own
train / valid / test split.

    python models/merge_datasets.py --out merged_plates \
        --source old=/content/drive/MyDrive/ANPR/plate_dataset/data.yaml \
        --source tr1=datasets/plate-detection-2/data.yaml \
        --source hard=datasets/openimages_plates/dataset.yaml

Writes merged_plates/data.yaml and prints a table: images and boxes per source and split.
"""
import argparse
import hashlib
import os
import shutil
from collections import defaultdict

import yaml

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLIT_KEYS = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}


def _names(cfg):
    n = cfg.get("names", [])
    if isinstance(n, dict):
        return [n[k] for k in sorted(n, key=int)]
    return list(n)


def plate_class_ids(names):
    ids = [i for i, n in enumerate(names) if "plate" in str(n).lower() or "plaka" in str(n).lower()]
    if not ids and len(names) == 1:          # single-class dataset: that class is the plate
        ids = [0]
    return set(ids)


def _resolve_images_dir(yaml_path, cfg, key):
    """Find the images folder for one split. Roboflow writes '../train/images', which is often wrong."""
    val = cfg.get(key)
    if not val:
        return None
    base = os.path.dirname(os.path.abspath(yaml_path))
    root = cfg.get("path")
    if isinstance(val, list):
        val = val[0]
    tries = [val, os.path.join(base, val), os.path.join(base, os.path.normpath(val).lstrip("./"))]
    if root:
        tries.insert(0, os.path.join(root if os.path.isabs(root) else os.path.join(base, root), val))
    # Roboflow quirk: '../train/images' -> '<base>/train/images'
    parts = os.path.normpath(val).split(os.sep)
    tries.append(os.path.join(base, *[p for p in parts if p not in ("..", ".")]))
    for t in tries:
        if os.path.isdir(t):
            return os.path.abspath(t)
    return None


def _labels_dir(images_dir):
    head, tail = os.path.split(images_dir)
    if tail == "images":
        return os.path.join(head, "labels")
    # FiftyOne style: images/<split> -> labels/<split>
    head2, tail2 = os.path.split(head)
    if tail2 == "images":
        return os.path.join(head2, "labels", tail)
    return images_dir.replace(os.sep + "images", os.sep + "labels")


def _to_box(values):
    """YOLO line values (after class id) -> (cx, cy, w, h). Polygons are turned into boxes."""
    v = [float(x) for x in values]
    if len(v) == 4:
        return v
    xs, ys = v[0::2], v[1::2]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def read_plate_labels(label_path, keep_ids):
    out = []
    if not os.path.exists(label_path):
        return out
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5 or int(float(p[0])) not in keep_ids:
                continue
            cx, cy, w, h = _to_box(p[1:])
            if w > 0 and h > 0:
                out.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out


def merge(sources, out_dir, keep_empty=True):
    """sources: list of (name, yaml_path). Returns stats {(name, split): [images, boxes]}."""
    seen = set()
    stats = defaultdict(lambda: [0, 0])
    for split in ("train", "valid", "test"):
        os.makedirs(os.path.join(out_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, split, "labels"), exist_ok=True)
    for name, ypath in sources:
        with open(ypath) as f:
            cfg = yaml.safe_load(f)
        keep = plate_class_ids(_names(cfg))
        if not keep:
            print(f"[{name}] no plate class in {_names(cfg)} - skipped")
            continue
        for key, split in SPLIT_KEYS.items():
            img_dir = _resolve_images_dir(ypath, cfg, key)
            if not img_dir:
                continue
            lab_dir = _labels_dir(img_dir)
            for fn in sorted(os.listdir(img_dir)):
                if not fn.lower().endswith(IMG_EXT):
                    continue
                src = os.path.join(img_dir, fn)
                with open(src, "rb") as f:
                    h = hashlib.md5(f.read()).hexdigest()
                if h in seen:
                    stats[(name, "duplicates")][0] += 1
                    continue
                seen.add(h)
                boxes = read_plate_labels(os.path.join(lab_dir, os.path.splitext(fn)[0] + ".txt"), keep)
                if not boxes and not keep_empty:
                    continue
                stem = f"{name}_{h[:10]}"
                ext = os.path.splitext(fn)[1].lower()
                shutil.copy(src, os.path.join(out_dir, split, "images", stem + ext))
                with open(os.path.join(out_dir, split, "labels", stem + ".txt"), "w") as f:
                    f.write("\n".join(boxes) + ("\n" if boxes else ""))
                stats[(name, split)][0] += 1
                stats[(name, split)][1] += len(boxes)
    out_abs = os.path.abspath(out_dir)
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        yaml.safe_dump({"path": out_abs, "train": "train/images", "val": "valid/images",
                        "test": "test/images", "nc": 1, "names": ["plate"]}, f, sort_keys=False)
    return dict(stats)


def print_table(stats):
    names = sorted({k[0] for k in stats})
    print("| Source | Train img/boxes | Valid img/boxes | Test img/boxes | Duplicates dropped |")
    print("|---|---|---|---|---|")
    tot = defaultdict(lambda: [0, 0])
    for n in names:
        cells = []
        for s in ("train", "valid", "test"):
            i, b = stats.get((n, s), [0, 0])
            tot[s][0] += i; tot[s][1] += b
            cells.append(f"{i} / {b}")
        print(f"| {n} | " + " | ".join(cells) + f" | {stats.get((n, 'duplicates'), [0])[0]} |")
    print("| **Total** | " + " | ".join(f"{tot[s][0]} / {tot[s][1]}" for s in ("train", "valid", "test")) + " | |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True, help="name=path/to/data.yaml")
    ap.add_argument("--out", default="merged_plates")
    ap.add_argument("--drop-empty", action="store_true", help="drop images with no plate box")
    a = ap.parse_args()
    srcs = [tuple(s.split("=", 1)) for s in a.source]
    print_table(merge(srcs, a.out, keep_empty=not a.drop_empty))
