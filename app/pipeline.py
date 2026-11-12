"""End-to-end pipeline: image in, results out (Week 4, Vision).

    from pipeline import ANPRPipeline
    pipe = ANPRPipeline()
    pipe.run("street.jpg")
    # [{"vehicle": "car", "plate": "34 PBL 657", "confidence": 0.91, ...}, ...]

Steps (our models, see docs/experiments.md):
    1. Find vehicles      - YOLOv8n pretrained on COCO (car, motorcycle, bus, truck)
    2. Name the type      - cls_v1  (YOLOv8n-cls: car, suv, van, truck, bus, motorbike)
    3. Find plates        - plate_v1 (YOLOv8n detect, mAP50 0.82)
    4. Read the plate     - PaddleOCR + Week 3 cleanup (ocr/plate_reader.py)
    5. Match plate to car - plate centre inside the vehicle box

Model files (not on GitHub, copy from Drive ANPR/models/):
    models/plate_best.pt, models/cls_best.pt
If one is missing the pipeline still runs and says what it skipped (see .warnings).

Command line:  python app/pipeline.py path/to/image.jpg
"""
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "models"))
sys.path.insert(0, os.path.join(ROOT, "ocr"))
from crop_plates import pad_box, plate_boxes  # noqa: E402

# COCO class id -> our vehicle name
COCO_VEHICLES = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}


def _contains(vbox, pbox):
    """True if the centre of the plate box is inside the vehicle box."""
    cx, cy = (pbox[0] + pbox[2]) / 2, (pbox[1] + pbox[3]) / 2
    return vbox[0] <= cx <= vbox[2] and vbox[1] <= cy <= vbox[3]


def match_plates(vehicles, plates):
    """Give each plate to the smallest vehicle box that contains it.

    vehicles: list of dicts with 'box'; plates: list of (x1, y1, x2, y2, conf).
    Returns {vehicle_index: plate_index} and the list of unmatched plate indexes.
    A vehicle keeps only its most confident plate.
    """
    best = {}
    unmatched = []
    for pi, p in enumerate(plates):
        inside = [vi for vi, v in enumerate(vehicles) if _contains(v["box"], p)]
        if not inside:
            unmatched.append(pi)
            continue
        vi = min(inside, key=lambda i: (vehicles[i]["box"][2] - vehicles[i]["box"][0]) *
                                       (vehicles[i]["box"][3] - vehicles[i]["box"][1]))
        if vi not in best or p[4] > plates[best[vi]][4]:
            if vi in best:
                unmatched.append(best[vi])
            best[vi] = pi
        else:
            unmatched.append(pi)
    return best, unmatched


class ANPRPipeline:
    def __init__(self,
                 vehicle_weights="yolov8n.pt",
                 type_weights=os.path.join(ROOT, "models", "cls_best.pt"),
                 plate_weights=os.path.join(ROOT, "models", "plate_best.pt"),
                 use_ocr=True, conf=0.25):
        from ultralytics import YOLO
        self.conf = conf
        self.warnings = []
        self.vehicle_model = YOLO(vehicle_weights)
        self.type_model = YOLO(type_weights) if os.path.exists(type_weights) else None
        if self.type_model is None:
            self.warnings.append(f"No type classifier at {os.path.relpath(type_weights, ROOT)} - using COCO names (no suv/van).")
        self.plate_model = YOLO(plate_weights) if os.path.exists(plate_weights) else None
        if self.plate_model is None:
            self.warnings.append(f"No plate detector at {os.path.relpath(plate_weights, ROOT)} - plates are skipped.")
        self.reader = None
        if use_ocr:
            try:
                from plate_reader import PlateReader
                self.reader = PlateReader(gray=True, scale=2.0, fix=True)
            except Exception as e:  # PaddleOCR not installed / failed to load
                self.warnings.append(f"OCR off: {e}")

    # --- steps -------------------------------------------------------------
    def find_vehicles(self, img):
        res = self.vehicle_model.predict(img, conf=self.conf, classes=list(COCO_VEHICLES), verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            out.append({"box": [x1, y1, x2, y2], "vehicle": COCO_VEHICLES[int(b.cls)], "confidence": float(b.conf)})
        return out

    def classify_type(self, img, v):
        if self.type_model is None:
            return v["vehicle"], None
        x1, y1, x2, y2 = v["box"]
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return v["vehicle"], None
        r = self.type_model.predict(crop, verbose=False)[0]
        return r.names[int(r.probs.top1)], float(r.probs.top1conf)

    def read_plate(self, img, p):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = pad_box(*p[:4], w, h, 0.05)
        if self.reader is None:
            return None, None, None
        o = self.reader.read(img[y1:y2, x1:x2])
        return o["text"] or None, o["score"], o["valid"]

    # --- main --------------------------------------------------------------
    def run(self, image):
        """image: file path or BGR numpy array -> list of result dicts."""
        img = cv2.imread(image) if isinstance(image, str) else image
        if img is None:
            raise ValueError(f"Could not read image: {image}")
        vehicles = self.find_vehicles(img)
        plates = plate_boxes(self.plate_model, img, self.conf) if self.plate_model else []
        matched, unmatched = match_plates(vehicles, plates)

        results = []
        for vi, v in enumerate(vehicles):
            vtype, tconf = self.classify_type(img, v)
            row = {"vehicle": vtype, "plate": None, "confidence": round(v["confidence"], 3),
                   "type_confidence": None if tconf is None else round(tconf, 3),
                   "plate_confidence": None, "ocr_confidence": None, "plate_valid": None,
                   "vehicle_box": v["box"], "plate_box": None}
            if vi in matched:
                p = plates[matched[vi]]
                self._add_plate(img, row, p)
            results.append(row)
        for pi in unmatched:  # plate found but no vehicle around it
            row = {"vehicle": "unknown", "plate": None, "confidence": None, "type_confidence": None,
                   "plate_confidence": None, "ocr_confidence": None, "plate_valid": None,
                   "vehicle_box": None, "plate_box": None}
            self._add_plate(img, row, plates[pi])
            results.append(row)
        return results

    def _add_plate(self, img, row, p):
        from plate_text import format_plate
        text, score, valid = self.read_plate(img, p)
        row.update(plate=format_plate(text) if text else None,
                   plate_confidence=round(p[4], 3), ocr_confidence=score, plate_valid=valid,
                   plate_box=[int(v) for v in p[:4]])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python app/pipeline.py image.jpg")
    pipe = ANPRPipeline()
    for w in pipe.warnings:
        print("WARNING:", w, file=sys.stderr)
    print(json.dumps(pipe.run(sys.argv[1]), indent=2))
