import cv2
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from ultralytics import YOLO

# ================================
# LOAD YOLO MODEL
# ================================
model = YOLO("yolov8n.pt")

# ================================
# CAMERA
# ================================
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

# ================================
# LINE POSITION
# ================================
LINE_Y = 360

# ================================
# ZONES (AUTO DIVIDED)
# ================================
ZONE1 = (0,0,1280,LINE_Y)
ZONE2 = (0,LINE_Y,1280,720)

zone1_count = 0
zone2_count = 0

# ================================
# COUNTERS
# ================================
entry_ids=set()
exit_ids=set()
total_people=set()

tracker={}
CROWD_LIMIT=5

# ================================
# DATA FILE
# ================================
DATA_FILE="crowd_data.csv"

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE,'w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(["time","zone","entry","exit","total_people"])

# ================================
# MAIN LOOP
# ================================
while True:

    ret,frame=cap.read()
    if not ret:
        break

    zone1_count=0
    zone2_count=0

    results=model.track(frame,persist=True,classes=[0])

    if results[0].boxes.id is not None:

        boxes=results[0].boxes.xyxy.cpu().numpy()
        ids=results[0].boxes.id.cpu().numpy()

        for box,track_id in zip(boxes,ids):

            x1,y1,x2,y2=map(int,box)

            cx=int((x1+x2)/2)
            cy=int((y1+y2)/2)

            track_id=int(track_id)

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)

            total_people.add(track_id)

            # ======================
            # TRACK HISTORY
            # ======================
            if track_id not in tracker:
                tracker[track_id]=[]

            tracker[track_id].append((cx,cy))

            if len(tracker[track_id])>=2:

                prev_y=tracker[track_id][-2][1]

                if prev_y < LINE_Y and cy > LINE_Y:
                    entry_ids.add(track_id)

                if prev_y > LINE_Y and cy < LINE_Y:
                    exit_ids.add(track_id)

            # ======================
            # ZONE CHECK
            # ======================
            if cy < LINE_Y:
                zone1_count+=1
                zone="Zone1"
            else:
                zone2_count+=1
                zone="Zone2"

            with open(DATA_FILE,'a',newline='') as f:
                writer=csv.writer(f)
                writer.writerow([
                    datetime.now(),
                    zone,
                    len(entry_ids),
                    len(exit_ids),
                    len(total_people)
                ])

    # ================================
    # DRAW LINE
    # ================================
    cv2.line(frame,(0,LINE_Y),(1280,LINE_Y),(255,0,0),3)

    # ================================
    # DRAW ZONES
    # ================================
    cv2.rectangle(frame,(0,0),(1280,LINE_Y),(255,255,0),2)
    cv2.rectangle(frame,(0,LINE_Y),(1280,720),(255,255,0),2)

    # ================================
    # DISPLAY STATS
    # ================================
    cv2.putText(frame,f"Entry : {len(entry_ids)}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.putText(frame,f"Exit : {len(exit_ids)}",(20,80),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.putText(frame,f"Total : {len(total_people)}",(20,120),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.putText(frame,f"Zone1 : {zone1_count}",(20,160),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

    cv2.putText(frame,f"Zone2 : {zone2_count}",(20,200),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

    # ================================
    # OVERCROWD ALERT
    # ================================
    if zone1_count > CROWD_LIMIT or zone2_count > CROWD_LIMIT:

        cv2.putText(frame,"OVER CROWD ALERT!",
                    (400,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0,0,255),
                    3)

    cv2.imshow("Crowd Monitoring System",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()

# ================================
# GRAPH GENERATION
# ================================
data=pd.read_csv(DATA_FILE)

plt.figure()
data["total_people"].plot()
plt.title("Crowd Trend")
plt.xlabel("Time Index")
plt.ylabel("Total People")
plt.show()

zone_data=data.groupby("zone").count()

plt.figure()
zone_data["total_people"].plot(kind='bar')
plt.title("Zone Usage")
plt.show()