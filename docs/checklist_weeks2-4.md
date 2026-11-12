# Weeks 2–4 Checklist

Legend: [x] done · [ ] still to do (needs a run in Colab / a photo shoot / the meeting)

## Week 2 – Train the model + try OCR
**Doğukan Yeni – Vision**
- [x] Plate detector trained (plate_v1, YOLOv8n, 50 epochs): mAP50 0.80 val / 0.818 test
- [x] Vehicle-type classifier trained (cls_v1, YOLOv8n-cls, 30 epochs): top-1 0.937
- [x] Results logged in `docs/experiments.md`, class table in `docs/table.md`
- [ ] `plate_best.pt` and `cls_best.pt` copied to Drive `ANPR/models/`
- [ ] Test on 20 new images, save good/bad screenshots → `models/week3_plate_crops.ipynb`, step 1

**Cemre Gönenç – Builder**
- [x] 50 plate crops run through EasyOCR and PaddleOCR (`ocr/week2_ocr_compare.ipynb`)
- [x] Better tool picked: **PaddleOCR 40/50 (80%)** vs EasyOCR 12/50 (24%) → `docs/week2_ocr_results.md`

**Done when:** mAP ≥ 0.6 ✅ (0.818) and B knows the better OCR tool ✅

## Week 3 – Improve OCR + connect
**Doğukan Yeni – Vision**
- [x] Script that crops plates from any image: `models/crop_plates.py`
- [ ] Run it on the 20 new images → Drive `ANPR/auto_crops/` (`models/week3_plate_crops.ipynb`, step 2)
- [ ] Optional: plate recall is 0.73 → try yolov8s / more epochs (notebook step 3)

**Cemre Gönenç – Builder**
- [x] OCR cleanup: grayscale, 2x enlarge, remove symbols (`ocr/plate_reader.py`)
- [x] Fix common mistakes by position: O→0, I→1, S→5 … (`ocr/plate_text.py`)
- [x] Check against the plate pattern – **Turkish format** (2 digits + 1–3 letters + 2–5 digits) instead of "ABC-123"
- [x] Unit tests (`ocr/test_plate_text.py`)
- [ ] Run `ocr/week3_ocr_cleanup.ipynb` → accuracy of each cleanup step
- [ ] Label the auto crops and measure accuracy on real crops (notebook step 4)

**Done when:** OCR ≥ 70% — baseline is already 80% on the hand crops; confirm on real crops.

## Week 4 – MVP app + demo
**Doğukan Yeni – Vision**
- [x] `app/pipeline.py`: image in → `{"vehicle": "car", "plate": "34 PBL 657", "confidence": 0.91, ...}`

**Cemre Gönenç – Builder**
- [x] Streamlit app `app/app.py`: upload button, image with boxes, results table, CSV download, SQLite save

**Both**
- [ ] Test on 10 never-seen images → `docs/week4_test_log.md`
- [ ] 10-minute demo: 3 slides + live app → `docs/demo_plan.md`
- [ ] Each writes 3 failures → `docs/phase2_todo.md`

**Milestone:** MVP review with supervisor. **Done when:** upload a photo and see vehicle type and plate text on screen.
