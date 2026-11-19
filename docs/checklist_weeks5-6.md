# Weeks 5–6 Checklist

Legend: [x] done · [ ] still to do (needs a run in Colab / filming a video)

## Week 5 – Fix the weak spots
**Doğukan Yeni – Vision**
- [x] Script to merge datasets into one plate-only set: `models/merge_datasets.py`
- [x] Notebook that downloads Turkish plate sets (Roboflow) + Open Images hard cases and retrains with augmentation: `models/week5_data_and_retrain.ipynb`
- [ ] Run it (needs a free Roboflow API key) → plate_v2
- [ ] Old vs new mAP on the same test sets → `docs/experiments.md` (notebook prints the rows)

**Cemre Gönenç – Builder**
- [x] Build a bigger OCR test set from character labels: `ocr/build_text_testset.py`
- [x] Sharpening, CLAHE, Otsu and adaptive thresholding + "fallback only when the pattern fails": `ocr/plate_reader.py`
- [x] Tests: `ocr/test_week5.py`
- [ ] Run `ocr/week5_hard_plates.ipynb` → accuracy on the bigger set, hard-30 set saved to Drive `ANPR/hard_plates/`
- [ ] Put the best setting into `app/pipeline.py`

**Done when:** mAP ≥ 0.75 (plate_v1 already 0.818) and OCR ≥ 80% (measure on the bigger set).

## Week 6 – Video support
**Doğukan Yeni – Vision**
- [x] Tracking with ByteTrack, every 3rd frame: `app/video_pipeline.py`, tuned tracker `app/bytetrack_anpr.yaml`

**Cemre Gönenç – Builder**
- [x] Plate voting over frames (valid pattern first, then total confidence): `ocr/plate_vote.py`
- [x] Each car saved once: tracks with the same plate are merged; database key = (video, plate)
- [x] Video tab in the app: upload, progress, table, CSV, annotated video download
- [x] Tests: `ocr/test_week6.py`

**Both**
- [ ] Film / find a 1-minute traffic video → Drive `ANPR/videos/traffic.mp4`
- [ ] Run `app/week6_video_test.ipynb`, count real cars vs rows, fill the table

**Done when:** a 1-minute traffic video gives a clean list of cars and plates with no duplicates.

### What was checked here (without the real models)
- Video code ran on a 7-second Turkish street clip: 17 cars tracked; the tuned tracker gave 17 rows vs 21 with default settings (fewer split tracks).
- Saving the same video twice added 0 new rows.
- Real accuracy must be measured in Colab with plate_best.pt, cls_best.pt and PaddleOCR.
