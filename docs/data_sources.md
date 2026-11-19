# Data sources (Week 5)

Downloaded automatically by the notebooks; not stored on GitHub (except our own 50 plates).

| Dataset | Used for | Where it is downloaded | License |
|---|---|---|---|
| Our 50 hand-checked plates (`data/plates/`) | OCR test (Weeks 2–3) | in repo | – |
| [Plate Detection – guler-kandeger/plate-detection-vh2rk](https://universe.roboflow.com/guler-kandeger/plate-detection-vh2rk) | Turkish plate boxes → plate_v2 training | `models/week5_data_and_retrain.ipynb` (Roboflow API) | see Roboflow page |
| [Turkish Number Plates – plakatanima-vnt3k](https://universe.roboflow.com/plakatanima-vnt3k/turkish-number-plates) | Turkish plate boxes → plate_v2 training | same | CC BY 4.0 |
| [Open Images V7 – Vehicle registration plate](https://storage.googleapis.com/openimages/web/index.html) | Hard cases (far, angled, blurry, motorbikes) + separate hard test set | same (FiftyOne) | CC BY 4.0 (annotations), images CC BY 2.0 |
| [plaka – metehan-yasar/plaka-cript](https://universe.roboflow.com/metehan-yasar/plaka-cript) | Character boxes → true plate text → bigger OCR test set + hard-30 | `ocr/week5_hard_plates.ipynb` (Roboflow API) | CC BY 4.0 |
| [plaka-tanıma-all – metehan-yasar](https://universe.roboflow.com/metehan-yasar/plaka-tanima-all) | same, extra plates | same | CC BY 4.0 |
| [Sample street video (araba.mp4)](https://github.com/Semihocakli/turkish-plate-recognition-w-yolov8-onnx-to-engine-cpp) | Only to check that the video code runs (it already has boxes drawn on it) | `app/week6_video_test.ipynb` fallback | Apache-2.0 |

**Why these:** plate_v1 missed small plates (recall 0.73) and the old data had non-Turkish plates.
More Turkish plates + hard cases target exactly that. For OCR, character-labelled plates give the
true text for free, so the test set grows from 50 to hundreds of plates.
