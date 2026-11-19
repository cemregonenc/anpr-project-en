# App (Week 4)

| File | Owner | Purpose |
|---|---|---|
| `pipeline.py` | Vision | Image in → list of `{"vehicle", "plate", "confidence", ...}` |
| `app.py` | Builder | Streamlit MVP: upload, boxes, results table, CSV, SQLite history |

```bash
pip install -r requirements.txt
# copy plate_best.pt and cls_best.pt from Drive ANPR/models/ into models/
streamlit run app/app.py
```
Command line test: `python app/pipeline.py some_photo.jpg`

## Video (Week 6)
| File | Owner | Purpose |
|---|---|---|
| `video_pipeline.py` | Vision | Track cars (ByteTrack, every 3rd frame), read plates, one row per car |
| `bytetrack_anpr.yaml` | Vision | Tracker settings tuned for frame step 3 |
| `week6_video_test.ipynb` | Both | 1-minute video test in Colab |

`python app/video_pipeline.py traffic.mp4 annotated.mp4` — or use the **Video** tab in the app.
