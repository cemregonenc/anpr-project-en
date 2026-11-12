# Phase 2 To-Do – 3 failures each (from MVP testing)

Confirm or replace each item after the 10-image test (`docs/week4_test_log.md`).

## Doğukan Yeni – Vision
1. **Small / far plates are missed** – plate recall 0.73 on test. → Week 5: add far/small plate images, try yolov8s or imgsz 960.
2. **car ↔ van confusion** in the type classifier. → add more borderline van/minivan images, check labels.
3. **SUV images come from one source** (possible shortcut learning). → add SUVs from other sources and angles, re-check accuracy.

## Cemre Gönenç – Builder
1. **Blurry / dark plates** are read wrong even after cleanup. → Week 5: try sharpening and thresholding, build a 30-image "hard plates" set.
2. **Two-line and non-standard plates** (and foreign plates in the dataset) do not fit the Turkish pattern. → handle two-line layout, mark foreign plates as "unknown format" instead of forcing a fix.
3. **Plate crop edges** (frame, TR band, stickers) add extra characters. → tune crop padding with Doğukan, keep the "drop one extra character" rule only when the pattern then fits.
