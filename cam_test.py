import cv2
import numpy as np

# 설정값 (상수)
ROI_RATIO = (0.00, 0.20, 0.35, 0.65)  # (top, bottom, left, right) 비율
BRIGHTNESS_THRESHOLD = 210           # 밝기 임계값
MIN_AREA_RATIO = 0.25                 # 밝은 영역이 ROI의 40% 이상이면 'OPEN'으로 판단

def get_door_status(frame):
    """주어진 프레임의 ROI를 분석하여 문 상태, 비율, 마스크를 반환하는 함수"""
    height, width, _ = frame.shape
    
    top = int(height * ROI_RATIO[0])
    bottom = int(height * ROI_RATIO[1])
    left = int(width * ROI_RATIO[2])
    right = int(width * ROI_RATIO[3])
    
    roi = frame[top:bottom, left:right]

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray_roi, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)

    bright_area = cv2.countNonZero(bright_mask)
    total_roi_area = roi.shape[0] * roi.shape[1]
    
    area_ratio = bright_area / total_roi_area if total_roi_area > 0 else 0
    
    # 상태(bool), 비율(float), 마스크 이미지(array)를 모두 반환
    return area_ratio > MIN_AREA_RATIO, area_ratio, bright_mask

# 웹캠 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라를 찾을 수 없습니다.")
        break

    # 1. 문 상태, 비율, 마스크를 한번에 받아오기
    is_open, area_ratio, bright_mask = get_door_status(frame)
    
    if is_open:
        status_text = "OPEN"
        box_color = (0, 0, 255)
    else:
        status_text = "CLOSED"
        box_color = (0, 255, 0)

    # 메인 영상에 Bounding Box와 텍스트 그리기
    height, width, _ = frame.shape
    top = int(height * ROI_RATIO[0])
    bottom = int(height * ROI_RATIO[1])
    left = int(width * ROI_RATIO[2])
    right = int(width * ROI_RATIO[3])
    cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
    cv2.putText(frame, status_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

    # 2. 새로운 창: 밝은 영역(마스크) 표시
    cv2.imshow("Bright Mask", bright_mask)

    # 3. 새로운 창: Area Ratio 값 표시용 이미지 생성
    # 검은색 배경 이미지 생성 (높이 100, 너비 400)
    ratio_display = np.zeros((100, 400), dtype=np.uint8)
    # 표시할 텍스트
    ratio_text = f"Area Ratio: {area_ratio:.3f}" # 소수점 3자리까지 표시
    # 텍스트를 이미지에 추가
    cv2.putText(ratio_display, ratio_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # Area Ratio 창 표시
    cv2.imshow("Area Ratio", ratio_display)

    # 메인 영상 창 표시
    cv2.imshow("Door Status Cam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()