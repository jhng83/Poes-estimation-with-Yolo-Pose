
import tkinter as tk
from tkinter import filedialog
import time

import cv2
from ultralytics import YOLO
import numpy as np

root = tk.Tk()
root.withdraw()

model = YOLO('yolov8n-pose.pt')
video_path = filedialog.askopenfilename()

cap = cv2.VideoCapture(video_path)
#writer = imageio.get_writer("results/output23.mp4", mode="I")
inference_fps = 1

right_shoulder = 6
left_shoulder = 5
right_hip = 11
left_hip = 12
right_feet = 15
left_feet = 16
right_knee = 13
left_knee = 14

frame_counter = 0

# Initialize centroids and object ID lists
centroids = []
object_ids = []

# Initialize object ID counter
object_id_count = 0

def draw_pt(ky_pts,label):
    pts = ky_pts[i][label].reshape(1, 3)
    x,y,z = pts[0]
    #cv2.circle(frame, (int(x), int(y)), 4, (0,0, 255), -1)
    
    return x,y
    

while cap.isOpened():
# Read a frame from the video
    success, frame = cap.read()
    
    if not success:
        # End of video file
        break
    frame_counter += 1
    
    if success:
        frame = cv2.resize(frame , (480, 640))
        if frame_counter %  inference_fps == 0 or frame_counter ==0:
            start_time = time.time()
            # Run YOLOv8 inference on the frame
            results = model(frame)
            
          
            ky_pts = results[0].keypoints.numpy()
            bbox = results[0].boxes.xyxy.numpy()
            for i in range(len(ky_pts)):
                
                #To store labels in list for iterating (TBD) drawing circle
                #will put it togather with draw_pts in next vr
                
                right_shoulder_x, right_shoulder_y = draw_pt(ky_pts,right_shoulder)
                left_shoulder_x, left_shoulder_y = draw_pt(ky_pts,left_shoulder)
                right_hip_x, right_hip_y = draw_pt(ky_pts,right_hip)
                left_hip_x, left_hip_y = draw_pt(ky_pts,left_hip)
                right_feet_x, right_feet_y = draw_pt(ky_pts,right_feet)
                left_feet_x, left_feet_y = draw_pt(ky_pts,left_feet)
                left_knee_x, left_knee_y = draw_pt(ky_pts,left_knee)
                right_knee_x, right_knee_y = draw_pt(ky_pts,right_knee)
                
                if right_hip_x and left_hip_x and right_shoulder_x and left_shoulder_x:
                    
                    COM = (int(0.5*(right_hip_x+left_hip_x)),int(0.5*(right_hip_y+left_hip_y)))
                    mid_shoulder = (int(0.5*(right_shoulder_x+left_shoulder_x)),int(0.5*(right_shoulder_y+left_shoulder_y)))
                    cv2.circle(frame, COM, 6, (0,255, 0), -1)
                    cv2.circle(frame, (int(left_knee_x), int(left_knee_y)), 4, (0,0, 255), -1)
                    cv2.circle(frame, (int(right_knee_x), int(right_knee_y)), 4, (0,0, 255), -1)
                    cv2.circle(frame, (int(right_shoulder_x), int(right_shoulder_y)), 4, (0,0, 255), -1)
                    cv2.circle(frame, (int(left_shoulder_x), int(left_shoulder_y)), 4, (0,0, 255), -1)
                    cv2.circle(frame, (int(right_feet_x), int(right_feet_y)), 4, (0,0, 255), -1)
                    cv2.circle(frame, (int(left_feet_x), int(left_feet_y)), 4, (0,0, 255), -1)
                    cv2.circle(frame, (int(left_hip_x), int(left_hip_y)), 4, (0,0, 255), -1)
                    cv2.circle(frame, (int(right_hip_x), int(right_hip_y)), 4, (0,0, 255), -1)
                    frame = cv2.line(frame, COM, mid_shoulder, (0,0,255), 2)
                    frame = cv2.line(frame, (int(left_shoulder_x), int(left_shoulder_y)), (int(right_shoulder_x), int(right_shoulder_y)), (0,0,255), 2)
                    frame = cv2.line(frame, (int(right_hip_x), int(right_hip_y)), (int(left_hip_x), int(left_hip_y)), (0,0,255), 2)
                    frame = cv2.line(frame, (int(left_hip_x), int(left_hip_y)), (int(left_knee_x), int(left_knee_y)), (0,0,255), 2)
                    frame = cv2.line(frame, (int(right_hip_x), int(right_hip_y)), (int(right_knee_x), int(right_knee_y)), (0,0,255), 2)
                    frame = cv2.line(frame, (int(left_knee_x), int(left_knee_y)), (int(left_feet_x), int(left_feet_y)), (0,0,255), 2)
                    frame = cv2.line(frame, (int(right_hip_x), int(right_hip_y)), (int(right_knee_x), int(right_knee_y)), (0,0,255), 2)
                    frame = cv2.line(frame, (int(right_knee_x), int(right_knee_y)), (int(right_feet_x), int(right_feet_y)), (0,0,255), 2)
                    
                box = bbox[i].reshape(1, 4)
             
                x1, y1, x2, y2 = box[0]
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Calculate centroid of bounding box
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2
                centroid = (centroid_x, centroid_y)
                # Initialize object ID for new object
                object_id = -1
                
                for i in range(len(centroids)):
                    # Calculate distance between centroids
                    dist = np.sqrt((centroids[i][0] - centroid[0])**2 + (centroids[i][1] - centroid[1])**2)

                    # If distance is less than a threshold value, assume it is the same object and update the centroid and object ID
                    if dist < 35:
                        centroids[i] = centroid
                        object_id = object_ids[i]
                        break
                
                # If object is new, add it to the centroids and object IDs lists and update the object ID counter
                if object_id == -1:
                    centroids.append(centroid)
                    object_ids.append(object_id_count)
                    object_id = object_id_count
                    object_id_count += 1

                    # Initialize tracker for new object
                    #tracker.init(frame, (x, y, w, h))
                
                cv2.putText(frame, "ID:"+str(object_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "COM:"+str(COM), (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,cv2.LINE_AA)

            end_time = time.time()
      
        cv2.putText(frame, "YOLO v8 + tracking"+"   "+"fps: " + str(round(1/(end_time - start_time),1)), (20,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,cv2.LINE_AA)
        cv2.imshow("YOLOv8 Pose", frame)
            
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
    else:
        # Break the loop if the end of the video is reached
        break
    
cap.release()
cv2.destroyAllWindows()