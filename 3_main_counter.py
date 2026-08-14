import sys
import os
import cv2
import numpy as np
import supervision as sv
from supervision.geometry.core import Point
from ultralytics import YOLO
import argparse
import json

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='input/cement_v1.mp4', help='Path to input video')
parser.add_argument('--recalibrate', action='store_true', help='Force redrawing the line by ignoring saved config')
args = parser.parse_args()

VIDEO_IN = args.input
VIDEO_OUT = f"output/result_{os.path.basename(VIDEO_IN)}"
MODEL_PATH = "best.pt" if os.path.exists("best.pt") else ("runs/detect/train/weights/best.pt" if os.path.exists("runs/detect/train/weights/best.pt") else "yolo11n.pt")
TARGET_COUNT = 36

# Generate a config file path based on the video name
CONFIG_PATH = f"{os.path.splitext(VIDEO_IN)[0]}_config.json"

if not os.path.exists(VIDEO_IN):
    sys.exit(f"ERROR: Video file not found: {VIDEO_IN}")
if not os.path.exists(MODEL_PATH):
    sys.exit(f"ERROR: Model file not found: {MODEL_PATH}")

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_IN)
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video Resolution: {W}x{H} @ {fps}fps")

ret, first_frame = cap.read()
if not ret: sys.exit("ERROR: Could not read first frame.")

# ================================================================
# PHASE 1 — INTERACTIVE UI (OR LOAD CONFIG)
# ================================================================
line_pts = []

if os.path.exists(CONFIG_PATH) and not args.recalibrate:
    print(f"\n[PHASE 1] Loading saved line configuration from {CONFIG_PATH}...")
    with open(CONFIG_PATH, 'r') as f:
        line_pts = json.load(f)
else:
    print("\n[PHASE 1] Interactive Line Calibration...")
    print("  -> A window will open. CLICK AND DRAG to draw a line across the corridor.")
    print("  -> Press ENTER to confirm and save.")
    
    clone = first_frame.copy()
    drawing = False
    temp_pt1 = None
    temp_pt2 = None
    
    def draw_line(event, x, y, flags, param):
        global line_pts, drawing, temp_pt1, temp_pt2
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            temp_pt1 = (x, y)
            temp_pt2 = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                temp_pt2 = (x, y)
                temp_clone = clone.copy()
                cv2.line(temp_clone, temp_pt1, temp_pt2, (0, 255, 0), 2)
                cv2.imshow("Calibration", temp_clone)
                
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            temp_pt2 = (x, y)
            line_pts = [temp_pt1, temp_pt2]
            temp_clone = clone.copy()
            cv2.line(temp_clone, temp_pt1, temp_pt2, (0, 255, 0), 2)
            cv2.imshow("Calibration", temp_clone)

    cv2.imshow("Calibration", clone)
    cv2.setMouseCallback("Calibration", draw_line)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13: # ENTER key
            if len(line_pts) == 2:
                # Validate line isn't too small (prevents NaN error)
                dist = np.linalg.norm(np.array(line_pts[1]) - np.array(line_pts[0]))
                if dist < 10:
                    print("  ⚠ Line is too short! Please click and drag to draw a longer line.")
                    line_pts = [] # Reset
                    cv2.imshow("Calibration", clone)
                else:
                    break
            else:
                print("  ⚠ Please click and drag to draw the line first!")
    cv2.destroyAllWindows()
    
    with open(CONFIG_PATH, 'w') as f:
        json.dump(line_pts, f)
    print(f"  ✓ Line saved to {CONFIG_PATH}")

# Calculate geometry from the user-drawn line
A = np.array(line_pts[0], dtype=float)
B = np.array(line_pts[1], dtype=float)

if np.linalg.norm(B - A) < 10:
    if os.path.exists(CONFIG_PATH):
        os.remove(CONFIG_PATH)
    sys.exit(f"ERROR: The loaded line is too short! Corrupted config {CONFIG_PATH} has been deleted. Please re-run the script and drag to draw a new line.")

# Vector along the drawn line
u = (B - A)
u = u / np.linalg.norm(u)

# Perpendicular vector (the motion path)
v = np.array([-u[1], u[0]])

# Offset thresholds for Red and Blue lines (dynamic based on resolution)
gap = max(10, int(W * 0.02))
prog_red = -gap
prog_blue = gap

# Points for drawing the dual lines
# Red line: Shift A and B backwards along v
red_A = A + prog_red * v
red_B = B + prog_red * v
# Blue line: Shift A and B forwards along v
blue_A = A + prog_blue * v
blue_B = B + prog_blue * v

red_pt1 = (int(red_A[0]), int(red_A[1]))
red_pt2 = (int(red_B[0]), int(red_B[1]))
blue_pt1 = (int(blue_A[0]), int(blue_A[1]))
blue_pt2 = (int(blue_B[0]), int(blue_B[1]))

# Create supervision line zones (for visual effects only)
red_zone = sv.LineZone(start=Point(x=red_pt1[0], y=red_pt1[1]), end=Point(x=red_pt2[0], y=red_pt2[1]))
blue_zone = sv.LineZone(start=Point(x=blue_pt1[0], y=blue_pt1[1]), end=Point(x=blue_pt2[0], y=blue_pt2[1]))

# ================================================================
# PHASE 2 — COUNTING & UI OVERLAY
# ================================================================
print("\n[PHASE 2] Counting & UI Overlay — processing full video...")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
os.makedirs("output", exist_ok=True)
writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
box_annotator = sv.BoxAnnotator(color=sv.Color(0, 255, 0), thickness=2)

track_state = {}
counted_ids = set()
frame_num = 0
MAX_BBOX_AREA = W * H * 0.80

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_num += 1

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
            P = np.array([cx, cy])
            
            # Project point P onto the motion vector v relative to point A
            prog = np.dot(P - A, v)

            # State Machine: Tracks progress along the v vector
            if tid not in track_state:
                if prog < prog_red:
                    track_state[tid] = 'before_red'
                elif prog > prog_blue:
                    track_state[tid] = 'past_blue'
                else:
                    track_state[tid] = 'middle'

            # Forward Direction
            if track_state[tid] == 'before_red' and prog > prog_red:
                track_state[tid] = 'crossed_red'
            if track_state[tid] == 'crossed_red' and prog > prog_blue:
                track_state[tid] = 'counted'
                counted_ids.add(tid)

            # Reverse Direction
            if track_state[tid] == 'past_blue' and prog < prog_blue:
                track_state[tid] = 'crossed_blue'
            if track_state[tid] == 'crossed_blue' and prog < prog_red:
                track_state[tid] = 'counted'
                counted_ids.add(tid)

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

    box_w, box_h, box_y, gap_ui = 85, 55, 10, 8
    font = cv2.FONT_HERSHEY_SIMPLEX

    # TARGET Box
    cv2.rectangle(frame, (10, box_y), (10 + box_w, box_y + box_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, box_y), (10 + box_w, box_y + box_h), (100, 100, 100), 1)
    cv2.putText(frame, "TARGET", (20, box_y + 18), font, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, str(TARGET_COUNT), (30, box_y + 45), font, 0.80, (255, 255, 255), 2, cv2.LINE_AA)

    # COUNTED Box
    x2 = 10 + box_w + gap_ui
    cv2.rectangle(frame, (x2, box_y), (x2 + box_w, box_y + box_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x2, box_y), (x2 + box_w, box_y + box_h), (100, 100, 100), 1)
    cv2.putText(frame, "COUNTED", (x2 + 5, box_y + 18), font, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, str(counted_now), (x2 + 20, box_y + 45), font, 0.80, (0, 255, 0), 2, cv2.LINE_AA)

    # REMAINING Box
    x3 = x2 + box_w + gap_ui
    cv2.rectangle(frame, (x3, box_y), (x3 + box_w + 15, box_y + box_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x3, box_y), (x3 + box_w + 15, box_y + box_h), (100, 100, 100), 1)
    cv2.putText(frame, "REMAINING", (x3 + 2, box_y + 18), font, 0.36, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, str(remaining), (x3 + 25, box_y + 45), font, 0.80, (0, 165, 255), 2, cv2.LINE_AA)

    # BOTTOM LEFT PANEL
    panel_w, panel_h, panel_x, panel_y = 420, 120, 20, H - 140
    # Keep it at bottom regardless of height, but make sure it doesn't clip
    if panel_y < 0: panel_y = H - 50
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (255, 255, 255), 2)

    red_crossed_count = sum(1 for state in track_state.values() if state in ['crossed_red', 'counted'])
    blue_crossed_count = sum(1 for state in track_state.values() if state in ['crossed_blue', 'counted'])

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
