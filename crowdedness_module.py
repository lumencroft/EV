import cv2

def get_crowdedness_decision(frame):
    h, w, _ = frame.shape
    center_pixel_color = frame[h // 2, w // 2]
    
    if center_pixel_color[1] > 100 and center_pixel_color[0] < 100 and center_pixel_color[2] < 100:
        crowdedness_value = 1
    else:
        crowdedness_value = 2
        
    return crowdedness_value