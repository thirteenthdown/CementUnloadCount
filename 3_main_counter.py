import sys
import os
import cv2
import numpy as np
import supervision as sv
from supervision.geometry.core import Point
from collections import defaultdict
from ultralytics import YOLO

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
VIDEO_IN = "input/cement_ref_video1.mp4"
VIDEO_OUT = "output/result2.mp4"
MODEL_PATH = "best.pt" if os.path.exists("best.pt") else ("runs/detect/train/weights/best.pt" if os.path.exists("runs/detect/train/weights/best.pt") else "yolo11n.pt")
TARGET_COUNT = 36

if not os.path.exists(VIDEO_IN):
    sys.exit(f"ERROR: Video file not found: {VIDEO_IN}")
if not os.path.exists(MODEL_PATH):
    sys.exit(f"ERROR: Model file not found: {MODEL_PATH}")

# ================================================================
# INITIALIZATION
# ================================================================
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_IN)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video Resolution: {W}x{H} @ {fps}fps")

# ================================================================
# PHASE 1 — QUALITY GATEKEEPER
# ================================================================
print("\n[PHASE 1] Quality Gatekeeper — checking video clarity...")
ret, frame = cap.read()
if not ret:
    sys.exit("ERROR: Could not read first frame.")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
print(f"  Blur score (Laplacian variance): {laplacian_var:.2f}")

if laplacian_var < 100:
    print("  ⚠ WARNING: Video is extremely blurry. Results may be poor.")
else:
    print("  ✓ Video quality is acceptable.\n")

# ================================================================
# SETUP DIAGONAL LINES & ZONES
# ================================================================
RED_LINE_START = Point(x=410, y=50)
RED_LINE_END = Point(x=530, y=700)
BLUE_LINE_START = Point(x=440, y=50)
BLUE_LINE_END = Point(x=560, y=700)

red_zone = sv.LineZone(start=RED_LINE_START, end=RED_LINE_END)
blue_zone = sv.LineZone(start=BLUE_LINE_START, end=BLUE_LINE_END)

# Reset video to frame 0 for the counting pass
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ================================================================
# PHASE 3 — COUNTING & UI OVERLAY
# ================================================================
print("[PHASE 3] Counting & UI Overlay — processing full video...\n")

# --- Output video writer ---
os.makedirs("output", exist_ok=True)
writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

# --- Annotators ---
box_annotator = sv.BoxAnnotator(color=sv.Color(0, 255, 0), thickness=2)

# Global tracking state for bags (to avoid double-counting)
track_state = {}
counted_ids = set()

def get_line_x(y, p1, p2):
    """Returns the X coordinate of a line at a given Y."""
    if p2.y == p1.y: return p1.x
    return p1.x + (p2.x - p1.x) * (y - p1.y) / (p2.y - p1.y)

# --- Pre-compute line coordinates for cv2.line drawing ---
red_pt1 = (RED_LINE_START.x, RED_LINE_START.y)
red_pt2 = (RED_LINE_END.x, RED_LINE_END.y)
blue_pt1 = (BLUE_LINE_START.x, BLUE_LINE_START.y)
blue_pt2 = (BLUE_LINE_END.x, BLUE_LINE_END.y)

frame_num = 0
MAX_BBOX_AREA = W * H * 0.10

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1

    # --- Detect + Track ---
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", device="mps", verbose=False, classes=[0], conf=0.1)[0]
    detections = sv.Detections.from_ultralytics(results)

    # --- Filter detections ---
    if detections.tracker_id is not None and len(detections) > 0:
        valid_id = detections.tracker_id != None
        areas = detections.area
        valid_area = areas < MAX_BBOX_AREA
        mask = valid_id & valid_area
        detections = detections[mask]

        # Use our own bulletproof spatial state machine instead of sv.LineZone.trigger
        # because the fallback YOLO model is noisy and drops frames.
        for box, tid in zip(detections.xyxy, detections.tracker_id):
            tid = int(tid)
            cx = (box[0] + box[2]) / 2
            cy = box[3].item() # Bottom Center

            # Find where the lines are at this exact Y coordinate
            red_x = get_line_x(cy, RED_LINE_START, RED_LINE_END)
            blue_x = get_line_x(cy, BLUE_LINE_START, BLUE_LINE_END)

            # Initialize state for new tracks
            if tid not in track_state:
                if cx > red_x:
                    track_state[tid] = 'right' # Started on the truck side
                else:
                    track_state[tid] = 'left'  # Started on the dropoff side

            # Transition: Right -> Crossed Red
            if track_state[tid] == 'right' and cx < red_x:
                track_state[tid] = 'crossed_red'

            # Transition: Crossed Red -> Crossed Blue -> Counted
            if track_state[tid] == 'crossed_red' and cx < blue_x:
                track_state[tid] = 'counted'
                counted_ids.add(tid)

        # Keep trigger running ONLY for LineZoneAnnotator to visually light up the lines
        red_zone.trigger(detections)
        blue_zone.trigger(detections)

    # ────────────────────────────────────────
    # VISUAL ANNOTATIONS
    # ────────────────────────────────────────

    # 1. Draw dynamic fence lines
    cv2.line(frame, red_pt1, red_pt2, (0, 0, 255), 3, cv2.LINE_AA)   # Red
    cv2.line(frame, blue_pt1, blue_pt2, (255, 0, 0), 3, cv2.LINE_AA)  # Blue

    # 2. Annotate bounding boxes (green)
    frame = box_annotator.annotate(scene=frame, detections=detections)

    # 3. Yellow dot at bbox center
    if len(detections) > 0:
        cx = ((detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2).astype(int)
        cy = ((detections.xyxy[:, 1] + detections.xyxy[:, 3]) / 2).astype(int)
        for x, y in zip(cx, cy):
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1, cv2.LINE_AA)

    # ────────────────────────────────────────
    # TOP-LEFT DASHBOARD (3 metric boxes)
    # ────────────────────────────────────────
    counted_now = len(counted_ids)
    remaining = max(0, TARGET_COUNT - counted_now)

    box_w, box_h = 85, 55
    box_y = 10
    gap = 8
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Box 1: TARGET
    x1 = 10
    cv2.rectangle(frame, (x1, box_y), (x1 + box_w, box_y + box_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x1, box_y), (x1 + box_w, box_y + box_h), (100, 100, 100), 1)
    cv2.putText(frame, "TARGET", (x1 + 10, box_y + 18), font, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, str(TARGET_COUNT), (x1 + 20, box_y + 45), font, 0.80, (255, 255, 255), 2, cv2.LINE_AA)

    # Box 2: COUNTED
    x2 = x1 + box_w + gap
    cv2.rectangle(frame, (x2, box_y), (x2 + box_w, box_y + box_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x2, box_y), (x2 + box_w, box_y + box_h), (100, 100, 100), 1)
    cv2.putText(frame, "COUNTED", (x2 + 5, box_y + 18), font, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, str(counted_now), (x2 + 20, box_y + 45), font, 0.80, (0, 255, 0), 2, cv2.LINE_AA)

    # Box 3: REMAINING
    x3 = x2 + box_w + gap
    cv2.rectangle(frame, (x3, box_y), (x3 + box_w + 15, box_y + box_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x3, box_y), (x3 + box_w + 15, box_y + box_h), (100, 100, 100), 1)
    cv2.putText(frame, "REMAINING", (x3 + 2, box_y + 18), font, 0.36, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, str(remaining), (x3 + 25, box_y + 45), font, 0.80, (0, 165, 255), 2, cv2.LINE_AA)

    # ────────────────────────────────────────
    # BOTTOM-LEFT DASHBOARD (line crossing stats)
    # ────────────────────────────────────────
    panel_w, panel_h = 420, 120
    panel_x, panel_y = 20, H - panel_h - 20

    # Black background with white border
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (255, 255, 255), 2)

    line_font = cv2.FONT_HERSHEY_SIMPLEX
    line_scale = 0.75
    line_thick = 2
    tx = panel_x + 20

    # Line 1: RED count
    red_crossed_count = sum(1 for state in track_state.values() if state in ['crossed_red', 'counted'])
    blue_crossed_count = len(counted_ids)

    cv2.putText(frame, f"LINE 1 (RED) : {red_crossed_count}", (tx, panel_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"LINE 2 (BLUE): {blue_crossed_count}", (tx, panel_y + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    # Line 3: TARGET
    cv2.putText(frame, f"TARGET: {TARGET_COUNT}", (tx, panel_y + 105),
                line_font, line_scale, (0, 255, 0), line_thick, cv2.LINE_AA)

    # --- Write frame ---
    writer.write(frame)

    if frame_num % 100 == 0:
        print(f"  Frame {frame_num:>5d}  |  Counted: {counted_now}  |  Remaining: {remaining}")

# ──────────────────────────────────────────────
# CLEANUP
# ──────────────────────────────────────────────
cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"\n{'='*50}")
print(f"  FINAL COUNT : {len(counted_ids)}")
print(f"  TARGET      : {TARGET_COUNT}")
print(f"  Output saved: {VIDEO_OUT}")
print(f"{'='*50}")
