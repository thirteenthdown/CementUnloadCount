import sys
import os
import cv2
import numpy as np
import supervision as sv
from supervision.geometry.core import Point
from ultralytics import YOLO
import argparse

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='input/cement_v1.mp4', help='Path to input video')
args = parser.parse_args()

VIDEO_IN = args.input
VIDEO_OUT = f"output/result_{os.path.basename(VIDEO_IN)}"
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
print(f"  Blur score: {laplacian_var:.2f}")

if laplacian_var < 100:
    print("  ⚠ WARNING: Video is blurry. Results may be poor.")
else:
    print("  ✓ Video quality is acceptable.\n")

# ================================================================
# PHASE 2 — AUTO CALIBRATION (DYNAMIC CORRIDOR DETECTION)
# ================================================================
print("\n[PHASE 2] Auto-Calibration — analyzing motion path...")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
track_points = {}
calib_frames = min(int(fps * 3), total_frames) # Use first 3 seconds

for i in range(calib_frames):
    ret, frame = cap.read()
    if not ret: break
    
    # Conf=0.1 to aggressively track bags/workers during calibration
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", device="mps", verbose=False, classes=[0], conf=0.1)[0]
    if results.boxes is not None and results.boxes.id is not None:
        for box, tid in zip(results.boxes.xyxy, results.boxes.id):
            tid = int(tid)
            cx, cy = float((box[0] + box[2]) / 2), float(box[3].item())
            if tid not in track_points: track_points[tid] = []
            track_points[tid].append((cx, cy))

valid_vectors = []
all_points = []
for tid, pts in track_points.items():
    if len(pts) > 5:
        p1, p2 = np.array(pts[0]), np.array(pts[-1])
        dist = np.linalg.norm(p2 - p1)
        if dist > 20: # Must have moved to be considered a path
            valid_vectors.append((p2 - p1) / dist)
            all_points.extend(pts)

if not valid_vectors:
    print("  ⚠ Auto-calibration failed (no motion detected). Using default lines.")
    V = np.array([0.0, 1.0]) # Default straight down
    C = np.array([W / 2, H / 2])
else:
    avg_vec = np.mean(valid_vectors, axis=0)
    V = avg_vec / np.linalg.norm(avg_vec)
    C = np.mean(all_points, axis=0)
    print(f"  ✓ Path vector: [{V[0]:.2f}, {V[1]:.2f}], Centroid: ({int(C[0])}, {int(C[1])})")

# Calculate perpendicular line points for drawing
prog_gap = max(10, int(W * 0.02)) # Scale gap to video resolution
prog_red = -prog_gap # Pixels before centroid
prog_blue = prog_gap # Pixels after centroid
U = np.array([-V[1], V[0]]) # Perpendicular vector
line_width = max(300, int(W * 0.3)) # Scale line width too

C_red = C + prog_red * V
red_pt1 = (int(C_red[0] - line_width * U[0]), int(C_red[1] - line_width * U[1]))
red_pt2 = (int(C_red[0] + line_width * U[0]), int(C_red[1] + line_width * U[1]))

C_blue = C + prog_blue * V
blue_pt1 = (int(C_blue[0] - line_width * U[0]), int(C_blue[1] - line_width * U[1]))
blue_pt2 = (int(C_blue[0] + line_width * U[0]), int(C_blue[1] + line_width * U[1]))

# Create supervision line zones purely for visual highlighting
red_zone = sv.LineZone(start=Point(x=red_pt1[0], y=red_pt1[1]), end=Point(x=red_pt2[0], y=red_pt2[1]))
blue_zone = sv.LineZone(start=Point(x=blue_pt1[0], y=blue_pt1[1]), end=Point(x=blue_pt2[0], y=blue_pt2[1]))

# ================================================================
# PHASE 3 — COUNTING & UI OVERLAY
# ================================================================
print("\n[PHASE 3] Counting & UI Overlay — processing full video...")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
os.makedirs("output", exist_ok=True)
writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
box_annotator = sv.BoxAnnotator(color=sv.Color(0, 255, 0), thickness=2)

track_state = {}
counted_ids = set()
frame_num = 0
MAX_BBOX_AREA = W * H * 0.15

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_num += 1

    # Use conf=0.1 to hold onto tracks longer during occlusions
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", device="mps", verbose=False, classes=[0], conf=0.1)[0]
    detections = sv.Detections.from_ultralytics(results)

    if detections.tracker_id is not None and len(detections) > 0:
        valid_id = detections.tracker_id != None
        areas = detections.area
        valid_area = areas < MAX_BBOX_AREA
        mask = valid_id & valid_area
        detections = detections[mask]

        for box, tid in zip(detections.xyxy, detections.tracker_id):
            tid = int(tid)
            cx, cy = float((box[0] + box[2]) / 2), float(box[3].item())
            
            # Projection along the motion path vector V relative to Centroid C
            prog = (cx - C[0]) * V[0] + (cy - C[1]) * V[1]

            if tid not in track_state:
                if prog < prog_red:
                    track_state[tid] = 'before_red'
                else:
                    track_state[tid] = 'past_red'

            if track_state[tid] == 'before_red' and prog > prog_red:
                track_state[tid] = 'crossed_red'

            if track_state[tid] == 'crossed_red' and prog > prog_blue:
                track_state[tid] = 'counted'
                counted_ids.add(tid)

        # Trigger for visual effects only
        red_zone.trigger(detections)
        blue_zone.trigger(detections)

    # ────────────────────────────────────────
    # VISUAL ANNOTATIONS
    # ────────────────────────────────────────
    cv2.line(frame, red_pt1, red_pt2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.line(frame, blue_pt1, blue_pt2, (255, 0, 0), 3, cv2.LINE_AA)
    frame = box_annotator.annotate(scene=frame, detections=detections)

    if len(detections) > 0:
        for box in detections.xyxy:
            cx, cy = int((box[0] + box[2]) / 2), int(box[3].item())
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1, cv2.LINE_AA)

    # UI Dashboard
    counted_now = len(counted_ids)
    remaining = max(0, TARGET_COUNT - counted_now)

    box_w, box_h, box_y, gap = 85, 55, 10, 8
    font = cv2.FONT_HERSHEY_SIMPLEX

    # TARGET Box
    cv2.rectangle(frame, (10, box_y), (10 + box_w, box_y + box_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, box_y), (10 + box_w, box_y + box_h), (100, 100, 100), 1)
    cv2.putText(frame, "TARGET", (20, box_y + 18), font, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, str(TARGET_COUNT), (30, box_y + 45), font, 0.80, (255, 255, 255), 2, cv2.LINE_AA)

    # COUNTED Box
    x2 = 10 + box_w + gap
    cv2.rectangle(frame, (x2, box_y), (x2 + box_w, box_y + box_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x2, box_y), (x2 + box_w, box_y + box_h), (100, 100, 100), 1)
    cv2.putText(frame, "COUNTED", (x2 + 5, box_y + 18), font, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, str(counted_now), (x2 + 20, box_y + 45), font, 0.80, (0, 255, 0), 2, cv2.LINE_AA)

    # REMAINING Box
    x3 = x2 + box_w + gap
    cv2.rectangle(frame, (x3, box_y), (x3 + box_w + 15, box_y + box_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x3, box_y), (x3 + box_w + 15, box_y + box_h), (100, 100, 100), 1)
    cv2.putText(frame, "REMAINING", (x3 + 2, box_y + 18), font, 0.36, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, str(remaining), (x3 + 25, box_y + 45), font, 0.80, (0, 165, 255), 2, cv2.LINE_AA)

    # BOTTOM LEFT PANEL
    panel_w, panel_h, panel_x, panel_y = 420, 120, 20, H - 140
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (255, 255, 255), 2)

    red_crossed_count = sum(1 for state in track_state.values() if state in ['crossed_red', 'counted'])
    blue_crossed_count = len(counted_ids)

    cv2.putText(frame, f"LINE 1 (RED) : {red_crossed_count}", (panel_x + 20, panel_y + 35), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"LINE 2 (BLUE): {blue_crossed_count}", (panel_x + 20, panel_y + 65), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"TARGET: {TARGET_COUNT}", (panel_x + 20, panel_y + 105), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    writer.write(frame)
    if frame_num % 100 == 0:
        print(f"  Frame {frame_num:>5d}  |  Counted: {counted_now}")

cap.release()
writer.release()

print(f"\n{'='*50}")
print(f"  FINAL COUNT : {len(counted_ids)}")
print(f"  Output saved: {VIDEO_OUT}")
print(f"{'='*50}")
