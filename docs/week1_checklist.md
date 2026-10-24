# Week 1 Checklist

## Student B – Builder
- [ ] GitHub repo created with folders: data/ models/ ocr/ app/ docs/
- [ ] Partner (A) added as a collaborator
- [ ] Colab notebook running with GPU (`!nvidia-smi` works)
- [ ] Google Drive mounted, `ANPR/` folder created
- [ ] OpenCV basics practiced: open, crop, grayscale, resize
- [ ] 50 hand-cropped plate images in `data/plates/`
- [ ] `labels.csv` filled with the correct text for all 50 plates

## Student A – Vision
- [ ] Images uploaded to Roboflow; every label checked (tight boxes, same class names, no unlabeled images)
- [ ] Missing plates labeled
- [ ] Classes fixed: car, suv, van, truck, bus, motorbike, plate
- [ ] Exported in YOLOv8 format with a 70/20/10 split
- [ ] Table of images per class (flag any class with fewer than 100)

**Done when:** Dataset exported, repo ready, B has 50 plate crops.
