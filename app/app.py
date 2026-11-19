"""ANPR app (Week 4 MVP + Week 6 video, Builder).

Run:   streamlit run app/app.py
Image tab: upload a photo -> boxes drawn on the image -> table of vehicle type + plate text.
Video tab: upload a traffic video -> one row per car (tracked), plate chosen by voting.
Results are saved to app/results.db (SQLite); each car of a video is saved only once.
"""
import os
import sqlite3
import sys
import tempfile
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


def save_video_results(video_name, rows, db_path=DB_PATH):
    """Save one row per car. A car with a valid plate is unique per (video, plate),
    other cars per (video, track id), so saving the same video twice adds nothing.
    Returns how many new rows were added."""
    with sqlite3.connect(db_path) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS video_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, video TEXT, track_id INTEGER,
            vehicle TEXT, plate TEXT, plate_valid INTEGER, votes INTEGER, frames_seen INTEGER,
            first_seen_s REAL, last_seen_s REAL, uid TEXT UNIQUE)""")
        now = datetime.now().isoformat(timespec="seconds")
        before = con.total_changes
        con.executemany(
            """INSERT OR IGNORE INTO video_detections (time, video, track_id, vehicle, plate, plate_valid,
               votes, frames_seen, first_seen_s, last_seen_s, uid) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            [(now, video_name, r["track_id"], r["vehicle"], r["plate"],
              None if r["plate_valid"] is None else int(r["plate_valid"]), r["votes"], r["frames_seen"],
              r["first_seen_s"], r["last_seen_s"],
              f"{video_name}|{r['plate']}" if r["plate_valid"] else f"{video_name}|track{r['track_id']}")
             for r in rows])
        return con.total_changes - before


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
    st.set_page_config(page_title="ANPR", page_icon="🚗", layout="wide")
    st.title("🚗 Number Plate Recognition & Vehicle Classification")
    st.caption("Upload a photo or a traffic video – the app finds each vehicle, names its type and reads the plate.")

    with st.sidebar:
        st.header("Settings")
        conf = st.slider("Detection confidence", 0.10, 0.90, 0.25, 0.05)
        save = st.checkbox("Save results to database", value=True)
        stride = st.slider("Video: process every Nth frame", 1, 10, 3)

    pipe = get_pipeline()
    pipe.conf = conf
    for w in pipe.warnings:
        st.sidebar.warning(w)

    tab_img, tab_vid = st.tabs(["📷 Image", "🎞️ Video"])
    with tab_img:
        image_tab(pipe, conf, save)
    with tab_vid:
        video_tab(pipe, stride, save)


def video_tab(pipe, stride, save):
    from video_pipeline import VideoANPR
    up = st.file_uploader("Upload a traffic video", type=["mp4", "mov", "avi", "mkv"], key="video")
    if up is None:
        st.info("Upload a short traffic video (about 1 minute). Every car is tracked and reported once.")
        return
    key = (up.file_id, stride, pipe.conf)
    if st.button("Process video", type="primary") and st.session_state.get("video_key") != key:
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, up.name)
            with open(src, "wb") as f:
                f.write(up.getvalue())
            bar = st.progress(0.0, text="Tracking cars and reading plates...")
            out_path = os.path.join(tmp, "annotated.mp4")
            rows = VideoANPR(pipe, stride=stride).process(src, out_video=out_path,
                                                          progress=lambda p: bar.progress(p, text=f"Processing... {p:.0%}"))
            bar.empty()
            with open(out_path, "rb") as f:
                st.session_state["video_annotated"] = f.read()
        st.session_state.update(video_key=key, video_rows=rows, video_saved=None)
    if st.session_state.get("video_key") != key:
        return
    rows = st.session_state["video_rows"]
    st.subheader(f"Cars found: {len(rows)}  ·  with a plate: {sum(1 for r in rows if r['plate'])}")
    if rows:
        table = pd.DataFrame(rows)[["track_id", "vehicle", "plate", "plate_valid", "votes", "readings",
                                    "frames_seen", "first_seen_s", "last_seen_s"]]
        st.dataframe(table, hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        c1.download_button("Download CSV", table.to_csv(index=False), "video_results.csv", "text/csv")
        c2.download_button("Download annotated video", st.session_state["video_annotated"],
                           "annotated.mp4", "video/mp4")
        if save and st.session_state.get("video_saved") is None:
            st.session_state["video_saved"] = save_video_results(up.name, rows)
        if st.session_state.get("video_saved") is not None:
            st.success(f"Saved {st.session_state['video_saved']} new cars to the database "
                       "(cars already saved from this video are skipped).")
    else:
        st.warning("No cars found. Try a lower confidence or a smaller frame step.")


def image_tab(pipe, conf, save):
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

    with st.expander("History (last 50 saved image detections)"):
        hist = load_history()
        if hist.empty:
            st.write("Nothing saved yet.")
        else:
            st.dataframe(hist, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
