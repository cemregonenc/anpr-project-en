# Models (Vision – Doğukan Yeni)

Model weights are **not** on GitHub (`*.pt` is ignored). Keep them in Google Drive `ANPR/models/` and copy here to run the app:

| File | What | Result |
|---|---|---|
| `models/plate_best.pt` | plate_v1 – YOLOv8n plate detector | mAP50 0.818 (test) |
| `models/cls_best.pt` | cls_v1 – YOLOv8n-cls vehicle type (car, suv, van, truck, bus, motorbike) | top-1 0.937 |

Vehicles are found with the COCO-pretrained `yolov8n.pt` (downloads automatically).

| File | Week | Purpose |
|---|---|---|
| `crop_plates.py` | 3 | Crop plates from any image/folder → crops + `crops.csv` |
| `week3_plate_crops.ipynb` | 2–3 | Test on 20 new images, make auto crops for OCR, optional yolov8s retrain |
| `merge_datasets.py` | 5 | Merge YOLO datasets into one plate-only set, drop duplicates |
| `week5_data_and_retrain.ipynb` | 5 | Download Turkish plate sets + Open Images hard cases, retrain plate_v2, old vs new |
