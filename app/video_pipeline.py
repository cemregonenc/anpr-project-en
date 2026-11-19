"""Video support (Week 6): track each car across frames and read its plate once.

    from pipeline import ANPRPipeline
    from video_pipeline import VideoANPR
    v = VideoANPR(ANPRPipeline(), stride=3)
    rows = v.process("traffic.mp4", out_video="traffic_annotated.mp4")
    # one row per car: track_id, vehicle, plate, plate_valid, votes, frames_seen, first_seen_s, last_seen_s

Vision part: YOLO tracking with ByteTrack (`model.track(..., tracker="bytetrack.yaml", persist=True)`),
only every 3rd frame is processed to keep it fast.
Builder part: every plate reading of a car is a vote (ocr/plate_vote.py); the best one is kept and
each car is reported once (tracks with the same plate are merged).

Command line:  python app/video_pipeline.py traffic.mp4 [annotated.mp4]
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ocr"))
sys.path.insert(0, os.path.join(ROOT, "models"))
from crop_plates import plate_boxes  # noqa: E402
from pipeline import COCO_VEHICLES, match_plates  # noqa: E402
from plate_text import format_plate  # noqa: E402
from plate_vote import Voter, dedupe_tracks  # noqa: E402


class VideoANPR:
    def __init__(self, pipe, stride=3, min_frames=3, type_checks=3,
                 tracker=os.path.join(HERE, "bytetrack_anpr.yaml")):
        """pipe: an ANPRPipeline (its models are reused).
        stride: process every Nth frame. min_frames: ignore tracks seen in fewer processed frames.
        type_checks: run the type classifier on at most this many frames per car."""
        self.pipe, self.stride, self.min_frames = pipe, stride, min_frames
        self.type_checks, self.tracker = type_checks, tracker

    def _new_tracker_model(self):
        """A separate model object for tracking: a fresh tracker per video, and image
        predictions in the app are not affected by the tracking callbacks."""
        from ultralytics import YOLO
        return YOLO(getattr(self.pipe.vehicle_model, "ckpt_path", None) or "yolov8n.pt")

    def _track(self, frame):
        r = self.track_model.track(frame, persist=True, tracker=self.tracker, conf=self.pipe.conf,
                                          classes=list(COCO_VEHICLES), verbose=False)[0]
        out = []
        if r.boxes is None or r.boxes.id is None:
            return out
        for b, tid in zip(r.boxes, r.boxes.id.int().tolist()):
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            out.append({"track_id": tid, "box": [x1, y1, x2, y2],
                        "vehicle": COCO_VEHICLES[int(b.cls)], "confidence": float(b.conf)})
        return out

    def process(self, video_path, out_video=None, progress=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        writer = None
        plates_vote, types = Voter(), defaultdict(Counter)
        seen = defaultdict(lambda: {"frames": 0, "first": None, "last": None, "conf": []})
        type_runs = Counter()
        label = {}
        self.track_model = self._new_tracker_model()
        last_draw = ([], {})

        i = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % self.stride:
                if writer is not None:
                    writer.write(self._draw(frame, last_draw))
                continue
            t = i / fps
            vehicles = self._track(frame)
            for v in vehicles:
                s = seen[v["track_id"]]
                s["frames"] += 1
                s["first"] = t if s["first"] is None else s["first"]
                s["last"] = t
                s["conf"].append(v["confidence"])
                if type_runs[v["track_id"]] < self.type_checks:
                    vtype, tconf = self.pipe.classify_type(frame, v)
                    types[v["track_id"]][vtype] += tconf if tconf is not None else v["confidence"]
                    type_runs[v["track_id"]] += 1
            plates = plate_boxes(self.pipe.plate_model, frame, self.pipe.conf) if self.pipe.plate_model else []
            matched, _ = match_plates(vehicles, plates)
            for vi, pi in matched.items():
                text, score, valid = self.pipe.read_plate(frame, plates[pi])
                tid = vehicles[vi]["track_id"]
                plates_vote.add(tid, text, score, valid)
                vehicles[vi]["plate_box"] = [int(x) for x in plates[pi][:4]]
            for v in vehicles:
                b = plates_vote.best(v["track_id"])
                label[v["track_id"]] = f"#{v['track_id']} {types[v['track_id']].most_common(1)[0][0] if types[v['track_id']] else v['vehicle']}" + \
                                       (f" {format_plate(b['text'])}" if b["text"] else "")
            last_draw = (vehicles, dict(label))
            if out_video:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(self._draw(frame, last_draw))
            if progress and total:
                progress(min(1.0, (i + 1) / total))
        cap.release()
        if writer is not None:
            writer.release()

        rows = []
        for tid, s in seen.items():
            if s["frames"] < self.min_frames:
                continue
            b = plates_vote.best(tid)
            rows.append({"track_id": tid,
                         "vehicle": types[tid].most_common(1)[0][0] if types[tid] else "vehicle",
                         "plate": format_plate(b["text"]) if b["text"] else None,
                         "plate_valid": b["valid"], "votes": b["votes"], "readings": b["readings"],
                         "ocr_confidence": b["confidence"],
                         "confidence": round(sum(s["conf"]) / len(s["conf"]), 3),
                         "frames_seen": s["frames"],
                         "first_seen_s": round(s["first"], 2), "last_seen_s": round(s["last"], 2)})
        return dedupe_tracks(rows)

    @staticmethod
    def _draw(frame, state):
        vehicles, label = state
        out = frame.copy()
        for v in vehicles:
            x1, y1, x2, y2 = v["box"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (60, 180, 75), 2)
            cv2.putText(out, label.get(v["track_id"], ""), (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 180, 75), 2)
            if v.get("plate_box"):
                a, b, c, d = v["plate_box"]
                cv2.rectangle(out, (a, b), (c, d), (0, 140, 255), 2)
        return out


def save_csv(rows, path):
    keys = ["track_id", "vehicle", "plate", "plate_valid", "votes", "readings", "ocr_confidence",
            "confidence", "frames_seen", "first_seen_s", "last_seen_s"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python app/video_pipeline.py video.mp4 [annotated_out.mp4]")
    from pipeline import ANPRPipeline
    pipe = ANPRPipeline()
    for w in pipe.warnings:
        print("WARNING:", w, file=sys.stderr)
    rows = VideoANPR(pipe).process(sys.argv[1], out_video=sys.argv[2] if len(sys.argv) > 2 else None)
    print(json.dumps(rows, indent=2))
