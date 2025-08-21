import cv2
import numpy as np

ROI_RATIO = (0.00, 0.20, 0.35, 0.65)
BRIGHTNESS_THRESHOLD = 200
MIN_AREA_RATIO = 0.30

# 전역 변수 call_count를 0으로 초기화
call_count = 0

def get_door_status(frame):
    global call_count  # 이 함수 내에서 전역 변수 call_count를 사용하겠다고 선언
    call_count += 1

    if call_count <= 5:
        door_status = 2
    else:
        door_status = 1
        call_count = 0

    return door_status