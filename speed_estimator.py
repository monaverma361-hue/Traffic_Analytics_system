
from ultralytics import YOLO
import cv2
import cvzone
import math
import torch
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


VIDEO_PATH = "demo/input_short.mp4"
MODEL_PATH = "yolov8n.pt"
MASK_PATH = "mask.png"

cap = cv2.VideoCapture(VIDEO_PATH) # for video


model = YOLO(MODEL_PATH)



class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]






total_count = []
car_crossing = False
previous_positions = {}  # Dictionary to store previous positions of each track ID

fps = cap.get(cv2.CAP_PROP_FPS)

pixels_per_meter = 40  # 🔥 TEMPORARY — we will calibrate properly

while True:
    success, img = cap.read()
    if not success:
        break

    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0])) # resized to match the video frame size
    imgRegion = cv2.bitwise_and(img, img, mask=mask_resized)

    results = model.track(imgRegion, tracker="bytetrack.yaml", persist=True, stream=True)

    # Draw counting line
    limits =[250, 350, 850, 350] # x1, y1, x2, y2
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (255, 0, 255), 5)

    

    # THIS LIST IS RESET EVERY FRAME
    current_frame_ids = []


    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if box.id is None:
                continue  # Skip if no ID is assigned
            track_id = int(box.id[0])
            conf = math.ceil((box.conf[0]*100))/100
            current_class = class_names[cls]

            if current_class in ["car", "truck", "bus", "motorcycle"] and conf > 0.3:
                current_frame_ids.append(track_id)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                # cvzone.putTextRect(img, f'{current_class} ID:{track_id} {conf}', 
                #                    (max(0, x1), max(35, y1-10)), scale=1, thickness=2, offset=3)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))

                cx, cy = x1 + w // 2, y1 + h // 2
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                
                """
================ VEHICLE SPEED CALCULATION LOGIC =================

STEP 1: Measure pixel movement per frame
    pixel_distance = sqrt((cx - prev_x)^2 + (cy - prev_y)^2)

STEP 2: Convert pixel movement (per frame) to pixel movement per second
    pixel_speed_per_second = pixel_distance * fps

STEP 3: Convert pixels to meters. ## "important step" ##
    We must know:
        Real-world distance (meters)
        Corresponding pixel distance in image

    Example:
        Distance between two lane markings = 3 meters
        Pixel distance between those lane markings = 120 pixels

    Then:
        pixels_per_meter = 120 / 3
        pixels_per_meter = 40

STEP 4: Convert pixel speed to meters per second
    meters_per_second = pixel_speed_per_second / pixels_per_meter

STEP 5: Convert meters per second to km/h
    speed_kmh = meters_per_second * 3.6

FINAL FORMULA:
    speed_kmh = ((pixel_distance * fps) / pixels_per_meter) * 3.6

==================================================================
"""
                # calculate pixel movement
                if track_id in previous_positions:
                    prev_cx, prev_cy = previous_positions[track_id]
                    pixel_distance = math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                    pixel_speed_per_second = pixel_distance * fps
                    meters_per_second = pixel_speed_per_second / pixels_per_meter
                    speed_kmh = meters_per_second * 3.6
                    # 🔥 Display ONLY ID + Speed
                    cvzone.putTextRect(img,f'ID {track_id}  {speed_kmh:.1f} km/h',(max(0, x1), max(35, y1 - 10)),
                                       scale=1,thickness=2,offset=5)
                        
                        
                        
                        
                        
                        
                    

                    # print(f'ID {track_id}  speed {speed_kmh:.2f} km/h')
                previous_positions[track_id] = (cx, cy)  # Update the previous position for this track ID
                    

                # Check if the center point is crossing the line
                if limits[0] < cx < limits[2] and limits[1] - 10 < cy < limits[1] + 10:
                    car_crossing = True  # This car is crossing
                    if track_id not in total_count:
                        total_count.append(track_id)
                        

                        # print(f'ID {track_id} crossed the line')
            cv2.putText(img, f'Total Count: {len(total_count)}', (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Draw counting line with temporary color if a car is crossing
    line_color = (0, 0, 255) if car_crossing else (255, 0, 255)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), line_color, 5)
    car_crossing = False  # Reset for next frame
    
            
                                

   


    

    
    # Only the IDs visible in this frame
    current_frame_ids_unique = sorted(set(current_frame_ids))
    if current_frame_ids_unique:
        print(f"Visible IDs this frame: {current_frame_ids_unique}  Total visible: {len(current_frame_ids_unique)}")

    

    cv2.imshow("image", img)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()