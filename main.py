import cv2
import pandas
from datetime import datetime

frameI = None
status_list=[None,None]
times=[]
df=pandas.DataFrame(columns=["Start","End"])
video = cv2.VideoCapture(0)

while True:
    Check, frame = video.read()
    status=0
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    if frameI is None:
        frameI = gray
        continue

    delta_frame = cv2.absdiff(frameI,gray)
    thread_delta = cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
    thread_delta = cv2.dilate(thread_delta,None,iterations =0)
    _, cnts, _ = cv2.findContours(thread_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        status=1
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    status_list.append(status)
    status_list=status_list[-2:]

    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())
    cv2.imshow('frame',gray)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

print(status_list)
print(times)

for i in range(0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)
df.to_csv("Times.csv")
video.release()
cv2.destroyAllWindows()
