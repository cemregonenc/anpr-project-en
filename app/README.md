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
