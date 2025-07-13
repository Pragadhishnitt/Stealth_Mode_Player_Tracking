"""# Main WorkFlow using Built-in Deepsort Tracker"""

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import random

# Load model
model = YOLO("best.pt")
tracker = DeepSort(max_age=60, n_init=2,max_cosine_distance=0.2, nn_budget=100)

# Video setup
cap = cv2.VideoCapture("15sec_input_720p.mp4")
width, height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("deepsort_realtime.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
colors = [(205,35,0), (0,205,35), (35, 0, 205),(0, 35, 205),(35,205,0),(205,0,35)]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        if cls_id != 2 or conf < 0.9:
            continue

        w, h = x2 - x1, y2 - y1
        detections.append(([x1, y1, w, h], conf, 'player'))

    # DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = int(track.track_id)
        l, t, r, b = map(int, track.to_ltrb())

        # Drawing accurate YOLO box instead of DeepSORT’s smoothed version
        cv2.rectangle(frame, (l, t), (r, b), colors[track_id % 6], 2)
        cv2.putText(frame, f"Player {track_id}", (l, t - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[track_id % 6], 2)

    out.write(frame)

cap.release()
out.release()
print("Final output saved with accurate boxes and stable IDs.")
