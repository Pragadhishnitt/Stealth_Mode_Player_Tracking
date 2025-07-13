import cv2
import numpy as np
import uuid
from ultralytics import YOLO

"""# Main Workflow For IOU Based Trackers"""

# --- Init ---

# Load YOLOv8 model trained to detect players
model = YOLO("best.pt")

# Load video
cap = cv2.VideoCapture("15sec_input_720p.mp4")

# Dictionary to hold active tracked players and lost ones
tracked_players = {}        # player_id: bbox
lost_players = {}           # player_id: {'last_position': (x,y), 'last_seen_frame': idx}

next_player_id = 0
max_lost_frames = 25
frame_idx = 0

# Setup output video writer
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_iou.mp4", fourcc, fps, (width, height))

# --- Utility Functions ---

# Get center of a bounding box
def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

#Calculate IoU between two bounding boxes
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou



while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object Detection using YOLO
    detections = []
    results = model(frame)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append([x1, y1, x2, y2])

    updated_players = {}
    used_ids = set()
    assigned_ids = set()

    # Match current detections with previous ones using IoU
    for det in detections:
        best_iou = 0
        best_id = None
        for pid, prev_bbox in tracked_players.items():
            if pid in used_ids:
                continue
            score = iou(det, prev_bbox)
            if score > best_iou:
                best_iou = score
                best_id = pid

        if best_iou > 0.75:
            updated_players[best_id] = det
            used_ids.add(best_id)
            assigned_ids.add(best_id)

        # Try re-identifying from lost_players based on distance
        else:
            center = get_center(det)
            reid_success = False
            for pid, info in list(lost_players.items()):
                if frame_idx - info["last_seen_frame"] > max_lost_frames:
                    continue
                lost_center = info["last_position"]
                dist = ((center[0] - lost_center[0])**2 + (center[1] - lost_center[1])**2) ** 0.5
                if dist < 40:
                    updated_players[pid] = det
                    assigned_ids.add(pid)
                    del lost_players[pid]
                    reid_success = True
                    break
            # New player ID
            if not reid_success:
                updated_players[next_player_id] = det
                assigned_ids.add(next_player_id)
                next_player_id += 1

    # Add unmatched tracked players to lost_players
    for pid in tracked_players:
        if pid not in assigned_ids:
            prev_bbox = tracked_players[pid]
            lost_players[pid] = {
                "last_position": get_center(prev_bbox),
                "last_seen_frame": frame_idx
            }

    tracked_players = updated_players

    # Draw boxes
    for pid, bbox in tracked_players.items():
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID: {pid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print("Output saved to output_iou.mp4")

"""# Main Workflow For Color Histogram Based Trackers"""

# --- Init ---
model = YOLO("best.pt")
cap = cv2.VideoCapture("15sec_input_720p.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_histogram.mp4", fourcc, fps, (width, height))

tracked_players = {}
lost_players = {}  # {id: {'hist': histogram, 'last_seen': frame_idx}}
next_id = 0
max_lost_frames = 30
frame_idx = 0

# --- Utility Functions ---

# Compute color histogram for a detection ROI
def compute_histogram(image, bbox):
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

# Compare histograms
def histogram_similarity(h1, h2):
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = []
    histograms = []
    results = model(frame)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append([x1, y1, x2, y2])
        histograms.append(compute_histogram(frame, [x1, y1, x2, y2]))

    updated_players = {}
    assigned_ids = set()

    for det, hist in zip(detections, histograms):
        matched = False

        # Try matching with lost players
        for pid, info in list(lost_players.items()):
            if frame_idx - info["last_seen"] > max_lost_frames:
                continue
            sim = histogram_similarity(hist, info["hist"])
            if sim > 0.8:
                updated_players[pid] = det
                del lost_players[pid]
                assigned_ids.add(pid)
                matched = True
                break

        if not matched:
            updated_players[next_id] = det
            assigned_ids.add(next_id)
            next_id += 1

    # Track unassigned tracked_players to lost
    for pid, bbox in tracked_players.items():
        if pid not in assigned_ids:
            lost_players[pid] = {
                "hist": compute_histogram(frame, bbox),
                "last_seen": frame_idx
            }

    tracked_players = updated_players

    # Draw on frame
    for pid, bbox in tracked_players.items():
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"ID: {pid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
