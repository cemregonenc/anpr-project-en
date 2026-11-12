# Week 2 – OCR Comparison (Builder)

**Test set:** 50 hand-checked Turkish plate crops (`data/plates/`, ground truth in `labels.csv`).
**Notebook:** `ocr/week2_ocr_compare.ipynb`
**Scoring:** text upper-cased, spaces/symbols and the `TR` band removed; no other fixes.

| Tool | Correct (of 50) | Accuracy | Avg. character similarity | Time (50 plates) |
|---|---|---|---|---|
| EasyOCR | 12 | 24.0% | 78.4% | 2.9 s (GPU) |
| **PaddleOCR** | **40** | **80.0%** | **97.3%** | 26.3 s (CPU) |

## Decision
**PaddleOCR** is used in the project. It reads 80% of plates exactly with no cleanup, and its
misses are mostly a single wrong character (97% similarity), which the Week 3 text fixes target.
It is slower (~0.5 s per plate on CPU) but fast enough for images.

## Notes
- Paddle 3.x crashes on CPU with a oneDNN error (`ConvertPirAttribute2RuntimeAttribute not support`).
  Fix: `enable_mkldnn=False` and `paddle.set_flags({'FLAGS_use_mkldnn': False})`.
- EasyOCR often splits the plate or adds characters from the frame and the `TR` band.
