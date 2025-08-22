import cv2
import numpy as np

# 설정값 (상수)
ROI_RATIO = (0.00, 0.20, 0.35, 0.65)  # (top, bottom, left, right) 비율
BRIGHTNESS_THRESHOLD = 210            # 밝기 임계값
MIN_AREA_RATIO = 0.4                 # 밝은 영역이 ROI의 40% 이상이면 'OPEN'으로 판단

def get_door_status(frame):
    """주어진 프레임의 ROI를 분석하여 문 상태를 반환하는 함수"""
    height, width, _ = frame.shape
    
    # ROI 좌표 계산
    top = int(height * ROI_RATIO[0])
    bottom = int(height * ROI_RATIO[1])
    left = int(width * ROI_RATIO[2])
    right = int(width * ROI_RATIO[3])
    
    roi = frame[top:bottom, left:right]

    # ROI를 흑백으로 변환 후, 임계값 이상인 부분만 흰색으로 마스킹
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray_roi, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)

    # 밝은 영역의 면적 계산
    bright_area = cv2.countNonZero(bright_mask)
    total_roi_area = roi.shape[0] * roi.shape[1]
    
    # 밝은 영역의 비율 계산
    area_ratio = bright_area / total_roi_area if total_roi_area > 0 else 0
    
    # 설정된 비율(MIN_AREA_RATIO)을 넘으면 True(OPEN) 반환
    return area_ratio > MIN_AREA_RATIO

# 1. 웹캠 열기
cap = cv2.VideoCapture(0)

# 2. 해상도 설정 (640x480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("카메라를 찾을 수 없습니다.")
        break

    # 3. 문 상태 분석
    is_open = get_door_status(frame)
    
    # 상태에 따라 텍스트와 색상 결정
    if is_open:
        status_text = "OPEN"
        box_color = (0, 0, 255)  # 빨간색
    else:
        status_text = "CLOSED"
        box_color = (0, 255, 0)  # 초록색

    #4. Bounding Box 그리기
    height, width, _ = frame.shape
    top = int(height * ROI_RATIO[0])
    bottom = int(height * ROI_RATIO[1])
    left = int(width * ROI_RATIO[2])
    right = int(width * ROI_RATIO[3])
    
    #화면에 사각형 그리기
    cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
    
    #5. 상태 텍스트 표시
    cv2.putText(frame, status_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

    #화면에 영상 출력
    cv2.imshow("Door Status Cam", frame)

    #'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()