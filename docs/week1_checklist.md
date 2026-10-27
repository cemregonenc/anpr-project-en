# Week 1 Checklist

## Cemre Gönenç – Builder
- [x] GitHub repo created with folders: data/ models/ ocr/ app/ docs/
- [ ] Doğukan added as a collaborator
- [x] Colab notebook running with GPU (`!nvidia-smi` works)
- [x] Google Drive mounted, `ANPR/` folder created
- [x] OpenCV basics practiced: open, crop, grayscale, resize
- [x] 50 plate crops in `data/plates/` (selected from the Roboflow export, Turkish plates only)
- [x] `labels.csv` filled with the correct text for all 50 plates

**Plate set mix:** 30 clear, 6 blurry, 5 small, 3 flipped (fixed), 2 angled, 1 dark, 1 two-line, 1 yellow, 1 sticker.

## Doğukan Yeni – Vision
- [ ] Images uploaded to Roboflow; every label checked (tight boxes, same class names, no unlabeled images)
- [ ] Missing plates labeled
- [ ] Classes fixed: car, suv, van, truck, bus, motorbike, plate
- [ ] Exported in YOLOv8 format with a 70/20/10 split
- [ ] Table of images per class (flag any class with fewer than 100)

### Notes for Friday
- The Roboflow export contains non-Turkish (Persian) plates. Decide: keep them for detection, or remove? (OCR test set uses Turkish plates only.)
- Some images are mirrored by augmentation; plate text is unreadable on those, which is fine for detection but not for OCR.
- Project plan says plate format "ABC-123"; Turkish format is 2 digits + 1–3 letters + 2–4 digits. Confirm with the instructor.

**Done when:** Dataset exported, repo ready, Builder has 50 plate crops.
