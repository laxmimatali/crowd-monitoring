import cv2
import json
import os

# ---------- VIDEO INPUT ----------
cap = cv2.VideoCapture(0)   # 0 = webcam
# cap = cv2.VideoCapture("video.mp4")  # use this for video file

ZONE_FILE = "zone.json"

drawing = False
start_point = (0, 0)
end_point = (0, 0)
rect = None

# Load saved zone
if os.path.exists(ZONE_FILE):
    with open(ZONE_FILE, "r") as f:
        rect = json.load(f)

# Mouse callback
def draw_zone(event, x, y, flags, param):
    global drawing, start_point, end_point, rect

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        rect = [start_point[0], start_point[1], end_point[0], end_point[1]]
        with open(ZONE_FILE, "w") as f:
            json.dump(rect, f)

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", draw_zone)

# Background subtractor (for object detection)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if rect is not None:
        x1, y1, x2, y2 = rect

        roi = frame[y1:y2, x1:x2]

        # Apply background subtraction
        fgmask = fgbg.apply(roi)

        # Find contours (objects)
        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        object_count = 0

        for cnt in contours:
            if cv2.contourArea(cnt) > 500:  # ignore noise
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 2)
                object_count += 1

        # Draw ROI rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show count
        cv2.putText(
            frame,
            f"Objects detected: {object_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()