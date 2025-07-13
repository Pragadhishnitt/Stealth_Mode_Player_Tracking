"""# Main Workflow"""

from ultralytics import YOLO
import cv2
import os
from tracker import Tracker

# Load model and video
model = YOLO("best.pt")
video_path = "/content/15sec_input_720p.mp4"
out_path = "/content/deepsort_custom.mp4"

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

tracker = Tracker()
colors = [(225,5,0), (0,225,5), (5, 0, 225)]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = []
    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        if conf > 0.6:
            detections.append([x1, y1, x2, y2, conf])

    tracker.update(frame, detections)

    for track in tracker.tracks:
        x1, y1, x2, y2 = map(int, track.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[track.track_id % 3], 2)
        cv2.putText(frame, f"ID: {track.track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[track.track_id % 3], 2)

    out.write(frame)

cap.release()
out.release()
print("Tracking complete. Video saved to:", out_path)