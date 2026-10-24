# Number Plate Recognition & Vehicle Classification (ANPR)

Upload an image or video; the system finds each vehicle, names its type,
finds the number plate, reads the plate text, and saves the result.

**Pipeline:** Image/Video → Detect vehicle → Classify type → Detect plate → Read plate (OCR) → Clean text → Save & display

## Folders
| Folder | Contents |
|---|---|
| `data/` | Dataset config and hand-cropped plates for OCR testing (`data/plates/`) |
| `models/` | Training scripts (`best.pt` is stored in Google Drive) |
| `ocr/` | OCR and text-cleaning code |
| `app/` | Streamlit app |
| `docs/` | Report notes and `experiments.md` |

## Team
- **A – Vision:** dataset, labeling, YOLO training, model testing
- **B – Builder:** OCR, pipeline, app, database

## Tools
Python, Google Colab (GPU), Roboflow, YOLOv8 (Ultralytics), EasyOCR / PaddleOCR, OpenCV, Streamlit, SQLite

## Setup
```bash
git clone https://github.com/cemregonenc/anpr-project.git
cd anpr-project
pip install -r requirements.txt
```
