import cv2
import numpy as np
import door

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    
    y1 = int(h * door.ROI_RATIO[0])
    y2 = int(h * door.ROI_RATIO[1])
    x1 = int(w * door.ROI_RATIO[2])
    x2 = int(w * door.ROI_RATIO[3])
    
    roi = frame[y1:y2, x1:x2]
    
    if roi.size == 0:
        continue

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray_roi, door.BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    bright_ratio = cv2.countNonZero(bright_mask) / bright_mask.size
    decision = int(bright_ratio > door.MIN_AREA_RATIO)

    status_text = "OPEN" if decision == 1 else "CLOSED"
    color = (0, 255, 0) if decision == 1 else (0, 0, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    text = f"Ratio: {bright_ratio:.2f} | Status: {status_text}"
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Live Feed", frame)
    cv2.imshow("Mask", bright_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()