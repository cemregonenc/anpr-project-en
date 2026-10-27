# Number Plate Recognition & Vehicle Classification (ANPR)

Upload an image or video; the system finds each vehicle, names its type,
finds the number plate, reads the plate text, and saves the result.

**Pipeline:** Image/Video → Detect vehicle → Classify type → Detect plate → Read plate (OCR) → Clean text → Save & display

## Folders
| Folder | Contents |
|---|---|
| `data/` | Dataset config; `data/plates/` holds 50 labeled plate crops for OCR testing (`labels.csv` = ground truth) |
| `models/` | Training scripts (`best.pt` is stored in Google Drive) |
| `ocr/` | OCR and text-cleaning code |
| `app/` | Streamlit app |
| `docs/` | Report notes and `experiments.md` |

## Team
| Name | Role | Responsibilities |
|---|---|---|
| Doğukan Yeni | Vision | Dataset, labeling, YOLO training, model testing |
| Cemre Gönenç | Builder | OCR, pipeline, Streamlit app, database |

Both: testing, documentation, demos. We work separately during the week and combine work every Friday.

## Tools
Python, Google Colab (GPU), Roboflow, YOLOv8 (Ultralytics), EasyOCR / PaddleOCR, OpenCV, Streamlit, SQLite

## Setup
```bash
git clone https://github.com/cemregonenc/anpr-project-en.git
cd anpr-project-en
pip install -r requirements.txt
```
