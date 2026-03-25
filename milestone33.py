import cv2
import csv
from datetime import datetime
from ultralytics import YOLO

# ===============================
# LOAD YOLO MODEL
# ===============================
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# ===============================
# LINE SETTINGS
# ===============================
LINE_Y = 300
OFFSET = 20   # increase tolerance for stability

# ===============================
# COUNTERS
# ===============================
entry_count = 0
exit_count = 0

# ===============================
# TRACK STORAGE
# ===============================
track_positions = {}
entry_ids = set()
exit_ids = set()

zone_1_ids = set()
zone_2_ids = set()

# ===============================
# CSV FILE
# ===============================
csv_file = open("count_data.csv", "a", newline="")
writer = csv.writer(csv_file)
writer.writerow(["Time", "Entry", "Exit"])
csv_file.flush()

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=[0])

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw detection
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)
            cv2.putText(frame, f"ID {track_id}", (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            # ===============================
            # INITIAL POSITION STORE
            # ===============================
            if track_id not in track_positions:
                track_positions[track_id] = cy

            prev_y = track_positions[track_id]

            # ===============================
            # ENTRY (Top → Bottom)
            # ===============================
            if prev_y < LINE_Y - OFFSET and cy > LINE_Y + OFFSET:
                if track_id not in entry_ids:
                    entry_count += 1
                    entry_ids.add(track_id)
                    print("ENTRY DETECTED")
                    writer.writerow([datetime.now(), entry_count, exit_count])

            # ===============================
            # EXIT (Bottom → Top)
            # ===============================
            elif prev_y > LINE_Y + OFFSET and cy < LINE_Y - OFFSET:
                if track_id not in exit_ids:
                    exit_count += 1
                    exit_ids.add(track_id)
                    print("EXIT DETECTED")
                    writer.writerow([datetime.now(), entry_count, exit_count])

            # ===============================
            # UPDATE POSITION
            # ===============================
            track_positions[track_id] = cy

            # ===============================
            # ZONE LOGIC (NO DOUBLE COUNTING)
            # ===============================
            if cy < LINE_Y:
                zone_1_ids.add(track_id)
                zone_2_ids.discard(track_id)
            else:
                zone_2_ids.add(track_id)
                zone_1_ids.discard(track_id)

    # ===============================
    # DRAW LINE
    # ===============================
    cv2.line(frame, (0, LINE_Y),
             (frame.shape[1], LINE_Y),
             (255,0,0), 3)

    # ===============================
    # DASHBOARD PANEL
    # ===============================
    cv2.rectangle(frame, (10, 10), (420, 200), (0,0,0), -1)

    cv2.putText(frame, f"Total People: {len(track_positions)}",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.putText(frame, f"Entry Count: {entry_count}",
                (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(frame, f"Exit Count: {exit_count}",
                (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.putText(frame, f"Zone 1 Count: {len(zone_1_ids)}",
                (20,130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.putText(frame, f"Zone 2 Count: {len(zone_2_ids)}",
                (20,160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("People Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()