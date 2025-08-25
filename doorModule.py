import cv2
import numpy as np

ROI_RATIO = (0.00, 0.20, 0.35, 0.65)
BRIGHTNESS_THRESHOLD = 210
MIN_AREA_RATIO = 0.4

def get_door_status(frame):
    h, w = frame.shape[:2]
    
    y1, y2 = int(h * ROI_RATIO[0]), int(h * ROI_RATIO[1])
    x1, x2 = int(w * ROI_RATIO[2]), int(w * ROI_RATIO[3])
    
    roi = frame[y1:y2, x1:x2]
    
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray_roi, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    if not bright_mask.size:
        return 0
        
    bright_ratio = cv2.countNonZero(bright_mask) / bright_mask.size
    
    return int(bright_ratio > MIN_AREA_RATIO)