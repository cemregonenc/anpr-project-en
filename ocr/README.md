# OCR (Builder – Cemre Gönenç)

| File | Week | Purpose |
|---|---|---|
| `week2_ocr_compare.ipynb` | 2 | EasyOCR vs PaddleOCR on the 50 plates → PaddleOCR (80%) |
| `plate_text.py` | 3 | Clean text, fix O→0 / I→1 / S→5 by position, Turkish pattern check |
| `plate_reader.py` | 3 | PaddleOCR + grayscale + 2x enlarge; joins two-line plates |
| `test_plate_text.py` | 3 | Unit tests: `python ocr/test_plate_text.py` |
| `week3_ocr_cleanup.ipynb` | 3 | Measures each cleanup step + accuracy on real auto crops |
| `build_text_testset.py` | 5 | Character-labelled dataset → plate crops + true text (bigger test set) |
| `test_week5.py` | 5 | Tests for the test-set builder and enhancement modes |
| `week5_hard_plates.ipynb` | 5 | Bigger test set, hard-30 set, sharpening / thresholding comparison |
| `plate_vote.py` | 6 | Best plate per car from many frames; merge duplicate tracks |
| `test_week6.py` | 6 | Tests for voting and de-duplication |
