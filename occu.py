# occu.py (수정)

import cv2
import numpy as np

DEPTH_LEVEL_START = 0.5
DEPTH_LEVEL_END = 2.1
DEPTH_LEVEL_STEP = 0.2
MIN_CONTOUR_AREA = 1000

def calculate_occupancy_score(depth_map, original_shape):
    h, w = original_shape
    depth_map_resized = cv2.resize(depth_map, (w, h))
    total_ratio = 0.0
    
    contours_to_draw = []
    hulls_to_draw = []
    
    levels = np.arange(DEPTH_LEVEL_START, DEPTH_LEVEL_END, DEPTH_LEVEL_STEP)
    for level in levels:
        mask = (depth_map_resized <= level).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue
        
        main_contour = max(contours, key=cv2.contourArea)
        area_contour = cv2.contourArea(main_contour)
        
        if area_contour < MIN_CONTOUR_AREA:
            continue

        hull = cv2.convexHull(main_contour)
        area_hull = cv2.contourArea(hull)
        if area_hull > 0:
            total_ratio += (area_hull - area_contour) / area_hull
            contours_to_draw.append(main_contour)
            hulls_to_draw.append(hull)
            
    return total_ratio, contours_to_draw, hulls_to_draw