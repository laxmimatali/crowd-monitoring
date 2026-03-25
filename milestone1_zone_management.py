import cv2
import json
import os

# ==================================================
# FILE PATH
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZONE_FILE = os.path.join(BASE_DIR, "zones.json")

# ==================================================
# COLORS
# ==================================================
colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255)
]

# ==================================================
# VARIABLES (START EMPTY)
# ==================================================
zones = []
start_point = None
drawing = False
frame = None

print("Starting fresh - no zones loaded")

# ==================================================
# MOUSE FUNCTION
# ==================================================
def draw_zone(event, x, y, flags, param):
    global start_point, drawing, zones, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp = frame.copy()
        cv2.rectangle(temp, start_point, (x, y), (0, 255, 255), 2)
        cv2.imshow("Milestone 1 - Zone Management", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)

        # Add zone
        zones.append([start_point, end_point])

        # Auto save
        with open(ZONE_FILE, "w") as f:
            json.dump(zones, f)

        print(f"Zone {len(zones)} saved")

# ==================================================
# CAMERA SETUP (STABLE)
# ==================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera not opening")
    exit()

cv2.namedWindow("Milestone 1 - Zone Management")
cv2.setMouseCallback("Milestone 1 - Zone Management", draw_zone)

# ==================================================
# MAIN LOOP
# ==================================================
while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Frame error")
        break

    frame = cv2.flip(frame, 1)

    # Draw zones
    for i, zone in enumerate(zones):
        color = colors[i % len(colors)]
        pt1 = tuple(zone[0])
        pt2 = tuple(zone[1])

        cv2.rectangle(frame, pt1, pt2, color, 2)

        cv2.putText(
            frame,
            f"Zone {i+1}",
            (pt1[0], pt1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imshow("Milestone 1 - Zone Management", frame)

    key = cv2.waitKey(1) & 0xFF

    # Delete last zone
    if key == ord('d') and zones:
        zones.pop()
        with open(ZONE_FILE, "w") as f:
            json.dump(zones, f)
        print("Last zone deleted")

    # Clear all zones
    elif key == ord('c'):
        zones.clear()
        with open(ZONE_FILE, "w") as f:
            json.dump(zones, f)
        print("All zones cleared")

    # Quit
    elif key == ord('q'):
        break

# ==================================================
# CLEAN EXIT
# ==================================================
cap.release()
cv2.destroyAllWindows()