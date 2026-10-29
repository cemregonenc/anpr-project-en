# Experiments Log

Add one row for every training run. The final report will be written from this table.

| Date | Who | Model | Epochs | imgsz | mAP50 | Notes |
|---|---|---|---|---|---|---|
| 2026-10-23 | A | YOLOv8n (detect) | 50 | 640 | 0.80 | plate_v1. Plate detector, 1,100-image plate set. P=0.89, R=0.75, mAP50-95=0.44. Misses some small plates. |
| 2026-10-24 | A | YOLOv8n-cls | 30 | 224 | top1 = 0.937 | cls_v1. Vehicle type classifier, crops from vehicles.zip + suv.zip, max 1,500/class. Main confusion: car↔van. SUV from single source, possible shortcut learning. |
