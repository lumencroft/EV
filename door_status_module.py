import cv2
import numpy as np

ROI_RATIO = (0.00, 0.20, 0.35, 0.65)
BRIGHTNESS_THRESHOLD = 220
MIN_AREA_RATIO = 0.40

call_count = 0

def get_door_status(frame):
    global call_count

    height, width, _ = frame.shape
    
    top = int(height * ROI_RATIO[0])
    bottom = int(height * ROI_RATIO[1])
    left = int(width * ROI_RATIO[2])
    right = int(width * ROI_RATIO[3])
    
    roi = frame[top:bottom, left:right]

    if roi.size == 0:
        call_count = 0
        return 1

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray_roi, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)

    bright_area = cv2.countNonZero(bright_mask)
    total_roi_area = roi.shape[0] * roi.shape[1]
    area_ratio = bright_area / total_roi_area if total_roi_area > 0 else 0
    
    if area_ratio > MIN_AREA_RATIO:
        call_count += 1
    else:
        call_count = 0
    
    if call_count > 5:
        door_status = 2
        call_count = 0
    else:
        door_status = 1
        
    return door_status