# MVP Demo Plan (10 minutes)

## Slide 1 – Problem and pipeline (Cemre, 1.5 min)
- Goal: upload an image → each vehicle, its type, its plate text, saved.
- Pipeline: Find vehicle (YOLOv8n COCO) → Type (cls_v1) → Find plate (plate_v1) → Read plate (PaddleOCR) → Clean text → Save & show.

## Slide 2 – Models and numbers (Doğukan, 2 min)
| Part | Model | Result |
|---|---|---|
| Plate detector | YOLOv8n, 50 epochs | mAP50 0.818 (test), P 0.895, R 0.729 |
| Vehicle type | YOLOv8n-cls, 30 epochs | top-1 0.937 (car↔van main confusion) |
| OCR | PaddleOCR vs EasyOCR | 40/50 (80%) vs 12/50 (24%) |
| OCR + cleanup | gray, 2x, position fixes, Turkish pattern | *fill in from week 3 notebook* |
- One good and one bad example image (from `ANPR/results/week2_20_images/`).

## Slide 3 – What fails and Phase 2 plan (both, 1.5 min)
- 3 failures each from `docs/phase2_todo.md`.
- Phase 2 targets: mAP ≥ 0.75, OCR ≥ 80%, video, database search.

## Live demo (Cemre drives, Doğukan explains, 4 min)
1. `streamlit run app/app.py`
2. Upload a clear daytime photo → point at boxes, type, plate text, table.
3. Upload a hard photo (night / angle) → show what goes wrong honestly.
4. Download CSV, open History.

## Questions (1 min)

**Before the meeting:** test the demo laptop once; keep 3 backup photos and screenshots in case the live run fails.
