import cv2, os, numpy as np
from datetime import datetime
from commu import Communicator
from collections import deque

SAVE_DIR = "captures"
THRESH = 1500
MAX_FRAMES = 100

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    comm = Communicator()

    if comm.wait_for_signal():
        cap = cv2.VideoCapture(0)
        frames, files = deque(maxlen=MAX_FRAMES), deque(maxlen=MAX_FRAMES)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Webcam", frame)

            if not frames or all(np.mean((f - gray) ** 2) >= THRESH for f in frames):
                if len(files) == MAX_FRAMES and os.path.exists(files[0]):
                    os.remove(files[0])
                name = os.path.join(SAVE_DIR, f"capture_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg")
                cv2.imwrite(name, frame)
                frames.append(gray)
                files.append(name)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        comm.close()
