"""Extract 1 frame per second from CCTV footage for dataset labeling."""
import os
import cv2

VIDEO_PATH = "input/cement_ref_video1.mp4"
OUTPUT_DIR = "frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_idx = 0
saved = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % fps == 0:
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{saved:05d}.jpg"), frame)
        saved += 1
    frame_idx += 1

cap.release()
print(f"Done — saved {saved} frames to '{OUTPUT_DIR}/'")
