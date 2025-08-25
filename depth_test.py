# depth_test.py (수정)

import cv2
import numpy as np
import occu
import depth

try:
    depth_model = depth.DepthEstimator(depth.OPENVINO_MODEL_XML_PATH)
except Exception as e:
    print(f"Model initialization failed: {e}")
    depth_model = None
    exit()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    depth_map = depth_model.run_inference(frame)
    total_ratio, contours, hulls = occu.calculate_occupancy_score(depth_map, frame.shape[:2])

    contour_display = np.zeros_like(frame)
    if contours:
        cv2.drawContours(contour_display, contours, -1, (0, 255, 0), 1)
        cv2.drawContours(contour_display, hulls, -1, (255, 0, 0), 1)

    text = f"Total Ratio: {total_ratio:.3f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Live Feed", frame)
    cv2.imshow("Contour Visualization", contour_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()