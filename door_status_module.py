import cv2
import numpy as np

ROI_RATIO = (0.00, 0.20, 0.35, 0.65)
BRIGHTNESS_THRESHOLD = 200
MIN_AREA_RATIO = 0.30

def get_door_status(frame):
    global call_count
    call_count += 1

    if call_count <= 5:
        door_status = 2
    else:
        door_status = 1
        call_count = 0

    return door_status