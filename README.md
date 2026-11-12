# Number Plate Recognition & Vehicle Classification (ANPR)

Upload an image or video; the system finds each vehicle, names its type,
finds the number plate, reads the plate text, and saves the result.

**Pipeline:** Image/Video → Detect vehicle → Classify type → Detect plate → Read plate (OCR) → Clean text → Save & display

## Folders
| Folder | Contents |
|---|---|
| `data/` | Dataset config; `data/plates/` holds 50 labeled plate crops for OCR testing (`labels.csv` = ground truth) |
| `models/` | Plate cropping script and notebooks (`*.pt` weights live in Google Drive) |
| `ocr/` | PaddleOCR reader, text cleanup + Turkish plate pattern, notebooks |
| `app/` | `pipeline.py` (image → results) and the Streamlit app |
| `docs/` | Experiments log, checklists, test log, demo plan, Phase 2 to-do |

## Team
| Name | Role | Responsibilities |
|---|---|---|
| Doğukan Yeni | Vision | Dataset, labeling, YOLO training, model testing |
| Cemre Gönenç | Builder | OCR, pipeline, Streamlit app, database |

Both: testing, documentation, demos. We work separately during the week and combine work every Friday.

## Tools
Python, Google Colab (GPU), Roboflow, YOLOv8 (Ultralytics), EasyOCR / PaddleOCR, OpenCV, Streamlit, SQLite

## Status (MVP phase, weeks 1–4)
| Week | Target | Result |
|---|---|---|
| 1 | Dataset exported, repo ready | ✅ dataset table in `docs/table.md`, 50 labeled plate crops |
| 2 | mAP ≥ 0.6, pick OCR tool | ✅ plate mAP50 0.818 (test) · PaddleOCR 80% vs EasyOCR 24% |
| 3 | OCR ≥ 70% | 🟡 baseline already 80%; cleanup + Turkish pattern check added; run `ocr/week3_ocr_cleanup.ipynb` for final numbers |
| 4 | MVP app + demo | 🟡 code ready: `app/pipeline.py` + Streamlit app; 10-image test + demo still to do (`docs/demo_plan.md`) |

Details: `docs/experiments.md`, `docs/checklist_weeks2-4.md`.

## Setup
```bash
git clone https://github.com/cemregonenc/anpr-project-en.git
cd anpr-project-en
pip install -r requirements.txt
```
Copy `plate_best.pt` and `cls_best.pt` from Google Drive `ANPR/models/` into `models/`, then:
```bash
streamlit run app/app.py               # web app
python app/pipeline.py photo.jpg       # command line
python ocr/test_plate_text.py          # OCR text-rule tests
```
