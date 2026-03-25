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
# DEFINE 4 ZONES
# ================================
ZONE1 = (500,300,650,520)
ZONE2 = (300,300,450,520)
ZONE3 = (420,150,520,300)
ZONE4 = (80,300,150,550)

# ================================
# COUNTERS
# ================================
zone1_count=0
zone2_count=0
zone3_count=0
zone4_count=0

entry_ids=set()
exit_ids=set()
total_people=set()

tracker={}

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
    zone3_count=0
    zone4_count=0

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

            # =========================
            # TRACK HISTORY
            # =========================
            if track_id not in tracker:
                tracker[track_id]=[]

            tracker[track_id].append((cx,cy))

            if len(tracker[track_id])>=2:

                prev_y=tracker[track_id][-2][1]

                if prev_y < LINE_Y and cy > LINE_Y:
                    entry_ids.add(track_id)

                if prev_y > LINE_Y and cy < LINE_Y:
                    exit_ids.add(track_id)

            # =========================
            # ZONE CHECK
            # =========================
            zone="None"

            if ZONE1[0] < cx < ZONE1[2] and ZONE1[1] < cy < ZONE1[3]:
                zone1_count+=1
                zone="Zone1"

            elif ZONE2[0] < cx < ZONE2[2] and ZONE2[1] < cy < ZONE2[3]:
                zone2_count+=1
                zone="Zone2"

            elif ZONE3[0] < cx < ZONE3[2] and ZONE3[1] < cy < ZONE3[3]:
                zone3_count+=1
                zone="Zone3"

            elif ZONE4[0] < cx < ZONE4[2] and ZONE4[1] < cy < ZONE4[3]:
                zone4_count+=1
                zone="Zone4"

            # =========================
            # SAVE DATA
            # =========================
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
    # DRAW ENTRY LINE
    # ================================
    cv2.line(frame,(0,LINE_Y),(1280,LINE_Y),(255,0,0),3)

    # ================================
    # DRAW ZONES
    # ================================
    cv2.rectangle(frame,(ZONE1[0],ZONE1[1]),(ZONE1[2],ZONE1[3]),(255,255,0),2)
    cv2.rectangle(frame,(ZONE2[0],ZONE2[1]),(ZONE2[2],ZONE2[3]),(255,255,0),2)
    cv2.rectangle(frame,(ZONE3[0],ZONE3[1]),(ZONE3[2],ZONE3[3]),(255,255,0),2)
    cv2.rectangle(frame,(ZONE4[0],ZONE4[1]),(ZONE4[2],ZONE4[3]),(255,255,0),2)

    # ================================
    # DISPLAY STATS
    # ================================
    cv2.putText(frame,f"Entry : {len(entry_ids)}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.putText(frame,f"Exit : {len(exit_ids)}",(20,80),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.putText(frame,f"Total : {len(total_people)}",(20,120),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.putText(frame,f"Zone 1 : {zone1_count}",(550,360),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

    cv2.putText(frame,f"Zone 2 : {zone2_count}",(300,360),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

    cv2.putText(frame,f"Zone 3 : {zone3_count}",(420,200),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

    cv2.putText(frame,f"Zone 4 : {zone4_count}",(60,360),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

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

zone_data=data["zone"].value_counts()

plt.figure()
zone_data.plot(kind='bar')
plt.title("Zone Usage")
plt.xlabel("Zone")
plt.ylabel("Count")
plt.show()