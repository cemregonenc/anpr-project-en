"""ANPR MVP app (Week 4, Builder).

Run:   streamlit run app/app.py
Upload a photo -> boxes drawn on the image -> table of vehicle type + plate text.
Results are also saved to app/results.db (SQLite) so they are not lost.
"""
import os
import sqlite3
import sys
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DB_PATH = os.path.join(HERE, "results.db")

VEHICLE_COLOR = (60, 180, 75)    # green (BGR)
PLATE_COLOR = (0, 140, 255)      # orange (BGR)


# ---------- drawing -----------------------------------------------------------
def _label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y = max(y, th + 6)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_results(img, results):
    out = img.copy()
    for r in results:
        if r.get("vehicle_box"):
            x1, y1, x2, y2 = r["vehicle_box"]
            cv2.rectangle(out, (x1, y1), (x2, y2), VEHICLE_COLOR, 2)
            conf = f" {r['confidence']:.2f}" if r.get("confidence") is not None else ""
            _label(out, f"{r['vehicle']}{conf}", x1, y1, VEHICLE_COLOR)
        if r.get("plate_box"):
            x1, y1, x2, y2 = r["plate_box"]
            cv2.rectangle(out, (x1, y1), (x2, y2), PLATE_COLOR, 2)
            _label(out, r["plate"] or "plate ?", x1, y2 + 22, PLATE_COLOR)
    return out


# ---------- database ----------------------------------------------------------
def save_results(image_name, results, db_path=DB_PATH):
    with sqlite3.connect(db_path) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, image TEXT,
            vehicle TEXT, plate TEXT, confidence REAL, plate_valid INTEGER)""")
        now = datetime.now().isoformat(timespec="seconds")
        con.executemany(
            "INSERT INTO detections (time, image, vehicle, plate, confidence, plate_valid) VALUES (?,?,?,?,?,?)",
            [(now, image_name, r["vehicle"], r["plate"], r["confidence"],
              None if r["plate_valid"] is None else int(r["plate_valid"])) for r in results])


def load_history(limit=50, db_path=DB_PATH):
    if not os.path.exists(db_path):
        return pd.DataFrame()
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(f"SELECT * FROM detections ORDER BY id DESC LIMIT {int(limit)}", con)


# ---------- app ---------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models (first time ~1 min)...")
def get_pipeline():
    from pipeline import ANPRPipeline
    return ANPRPipeline()


def main():
    st.set_page_config(page_title="ANPR MVP", page_icon="🚗", layout="wide")
    st.title("🚗 Number Plate Recognition & Vehicle Classification")
    st.caption("Upload a photo – the app finds each vehicle, names its type and reads the plate.")

    with st.sidebar:
        st.header("Settings")
        conf = st.slider("Detection confidence", 0.10, 0.90, 0.25, 0.05)
        save = st.checkbox("Save results to database", value=True)

    pipe = get_pipeline()
    pipe.conf = conf
    for w in pipe.warnings:
        st.sidebar.warning(w)

    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if up is None:
        st.info("Upload a JPG or PNG photo of a street or a car to start.")
    else:
        img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            st.error("Could not read this image.")
            return
        with st.spinner("Detecting vehicles and reading plates..."):
            results = pipe.run(img)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.image(cv2.cvtColor(draw_results(img, results), cv2.COLOR_BGR2RGB), width="stretch")
        with col2:
            st.subheader(f"Results ({len(results)})")
            if not results:
                st.warning("No vehicles or plates found. Try a lower confidence in the sidebar.")
            else:
                table = pd.DataFrame(results)[["vehicle", "plate", "confidence", "plate_valid",
                                               "type_confidence", "ocr_confidence"]]
                st.dataframe(table, hide_index=True, width="stretch")
                st.download_button("Download CSV", table.to_csv(index=False), "results.csv", "text/csv")
                key = (up.file_id, conf)
                if save and st.session_state.get("saved_key") != key:   # save once per upload
                    save_results(up.name, results)
                    st.session_state["saved_key"] = key
                if st.session_state.get("saved_key") == key:
                    st.success("Saved to database.")

    with st.expander("History (last 50 saved detections)"):
        hist = load_history()
        if hist.empty:
            st.write("Nothing saved yet.")
        else:
            st.dataframe(hist, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
