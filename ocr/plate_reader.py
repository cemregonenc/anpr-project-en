"""Read a plate crop with PaddleOCR (chosen in Week 2: 40/50 vs EasyOCR 12/50).

    from ocr.plate_reader import PlateReader
    reader = PlateReader()
    result = reader.read("data/plates/plate_01.jpg")   # path or BGR numpy image
    # {'text': '09KS627', 'raw': '09 KS 627', 'score': 0.98, 'valid': True}
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plate_text import basic_clean, fix_plate  # noqa: E402


ENHANCE_MODES = (None, "sharpen", "clahe", "otsu", "adaptive")


def enhance_gray(g, mode):
    """Week 5 extra cleanup for hard plates. g: grayscale image."""
    if mode is None:
        return g
    if mode == "sharpen":                       # unsharp mask: makes blurry edges crisper
        blur = cv2.GaussianBlur(g, (0, 0), 3)
        return cv2.addWeighted(g, 1.6, blur, -0.6, 0)
    if mode == "clahe":                         # local contrast: dark / over-exposed plates
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 8)).apply(g)
    if mode == "otsu":                          # global black/white threshold
        return cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if mode == "adaptive":                      # local threshold: uneven light, shadows
        return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    raise ValueError(f"unknown enhance mode {mode!r}, use one of {ENHANCE_MODES}")


def preprocess(img, gray=True, scale=2.0, enhance=None):
    """Week 3 cleanup (grayscale + enlarge) + optional Week 5 enhancement.
    Returns a 3-channel image for PaddleOCR."""
    if scale and scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if gray or enhance:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(enhance_gray(g, enhance), cv2.COLOR_GRAY2BGR)
    return img


class PlateReader:
    def __init__(self, gray=True, scale=2.0, fix=True, enhance=None, fallback=()):
        """fallback: enhance modes to try, in order, only when the first reading
        does not fit the plate pattern (e.g. ("sharpen", "clahe")). Week 5."""
        import paddle
        paddle.set_flags({"FLAGS_use_mkldnn": False})   # known oneDNN bug in Paddle 3.x
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(lang="en",
                             use_doc_orientation_classify=False,
                             use_doc_unwarping=False,
                             use_textline_orientation=False,
                             enable_mkldnn=False)
        self.gray, self.scale, self.fix, self.enhance = gray, scale, fix, enhance
        self.fallback = tuple(fallback)

    def raw_read(self, img):
        """Return (text, mean_score). Lines are joined top-to-bottom (two-line plates)."""
        res = self.ocr.predict(img)
        pieces = []
        for r in res:
            texts, scores = list(r["rec_texts"]), list(r["rec_scores"])
            boxes = r.get("rec_boxes")
            boxes = [] if boxes is None else np.asarray(boxes).tolist()
            for i, (t, s) in enumerate(zip(texts, scores)):
                if i < len(boxes):
                    x1, y1, _, y2 = boxes[i][:4]
                else:
                    x1, y1, y2 = i, 0, 1
                pieces.append((y1, x1, t, s, max(y2 - y1, 1)))
        if not pieces:
            return "", 0.0
        # group into rows: same row if the tops differ by less than half a line height
        pieces.sort(key=lambda p: p[0])
        rows, last_y = [], None
        for p in pieces:
            if last_y is None or abs(p[0] - last_y) > 0.5 * p[4]:
                rows.append([])
            rows[-1].append(p)
            last_y = p[0]
        ordered = [p for row in rows for p in sorted(row, key=lambda p: p[1])]
        text = " ".join(p[2] for p in ordered)
        score = float(np.mean([p[3] for p in ordered]))
        return text, score

    def read(self, image):
        img = cv2.imread(image) if isinstance(image, str) else image
        if img is None or img.size == 0:
            return {"text": "", "raw": "", "score": 0.0, "valid": False}
        out = self._read_once(img, self.enhance)
        for mode in self.fallback:            # only when the pattern check failed
            if out["valid"] or not self.fix:
                break
            alt = self._read_once(img, mode)
            if alt["valid"]:
                out = alt
        return out

    def _read_once(self, img, enhance):
        raw, score = self.raw_read(preprocess(img, self.gray, self.scale, enhance))
        if self.fix:
            text, valid = fix_plate(raw)
        else:
            text, valid = basic_clean(raw), None
        return {"text": text, "raw": raw, "score": round(score, 3), "valid": valid, "mode": enhance}
