import cv2
import json
import os
from datetime import datetime

ZONE_FILE = "zones.json"

zones = []
drawing = False
start_x, start_y = -1, -1
fullscreen = False


# ---------------- LOAD SAVED ZONES ----------------
if os.path.exists(ZONE_FILE):
    try:
        with open(ZONE_FILE, "r") as f:
            loaded_data = json.load(f)

        # Convert old list format to new dictionary format
        for item in loaded_data:
            if isinstance(item, list):
                zone = {
                    "x1": item[0],
                    "y1": item[1],
                    "x2": item[2],
                    "y2": item[3],
                    "created_at": "converted_from_milestone1"
                }
                zones.append(zone)
            else:
                zones.append(item)

        # Save converted format back to JSON
        with open(ZONE_FILE, "w") as f:
            json.dump(zones, f, indent=4)

    except:
        zones = []


# ---------------- SAVE ZONES FUNCTION ----------------
def save_zones():
    with open(ZONE_FILE, "w") as f:
        json.dump(zones, f, indent=4)


# ---------------- MOUSE FUNCTION ----------------
def draw_rectangle(event, x, y, flags, param):
    global start_x, start_y, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        zone = {
            "x1": start_x,
            "y1": start_y,
            "x2": x,
            "y2": y,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        zones.append(zone)
        save_zones()


# ---------------- CAMERA SETUP ----------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Camera failed to open.")
    exit()

cv2.namedWindow("Zone Manager", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Zone Manager", draw_rectangle)


# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ ERROR: Failed to read frame.")
        break

    # -------- DRAW ZONES --------
    for i, zone in enumerate(zones):

        x1 = zone["x1"]
        y1 = zone["y1"]
        x2 = zone["x2"]
        y2 = zone["y2"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"Zone {i+1}"
        count_text = "Count: 0"

        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, count_text, (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # -------- INSTRUCTION OVERLAY --------
    instructions = [
        "Mouse Drag : Draw Zone",
        "d : Delete Last Zone",
        "r : Reset All Zones",
        "p : Save Screenshot",
        "f : Toggle Fullscreen",
        "q : Quit"
    ]

    for idx, text in enumerate(instructions):
        cv2.putText(frame, text, (10, 30 + idx * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Zone Manager", frame)

    key = cv2.waitKey(1) & 0xFF

    # -------- DELETE LAST ZONE --------
    if key == ord('d'):
        if zones:
            zones.pop()
            save_zones()

    # -------- RESET ALL ZONES --------
    elif key == ord('r'):
        zones.clear()
        save_zones()

    # -------- SAVE SCREENSHOT --------
    elif key == ord('p'):
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")

    # -------- FULLSCREEN TOGGLE --------
    elif key == ord('f'):
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty("Zone Manager",
                                  cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("Zone Manager",
                                  cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_NORMAL)

    # -------- QUIT --------
    elif key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()