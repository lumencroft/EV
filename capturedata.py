import cv2
import os
from datetime import datetime
from commu import Communicator
import numpy as np
from collections import deque

SAVE_DIR = "captures"
CHANGE_THRESHOLD = 1500 
MAX_SAVE_COUNT = 100

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    communicator = Communicator()

    if communicator.wait_for_signal():
        print("Signal received! Starting webcam...")
        cap = cv2.VideoCapture(0)
        
        saved_frames_gray = deque(maxlen=MAX_SAVE_COUNT)
        saved_filenames = deque(maxlen=MAX_SAVE_COUNT)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('Webcam - Press "q" to stop', frame)
            
            if not saved_frames_gray:
                print("First unique frame captured.")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(SAVE_DIR, f"capture_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                saved_frames_gray.append(current_frame_gray)
                saved_filenames.append(filename)
                continue

            is_new_scene = True
            for past_frame_gray in saved_frames_gray:
                mse = np.mean((past_frame_gray.astype("float") - current_frame_gray.astype("float")) ** 2)
                if mse < CHANGE_THRESHOLD:
                    is_new_scene = False
                    break
            
            if is_new_scene:
                if len(saved_filenames) == MAX_SAVE_COUNT:
                    file_to_delete = saved_filenames[0]
                    if os.path.exists(file_to_delete):
                        os.remove(file_to_delete)
                        print(f"Limit reached. Deleting oldest file: {os.path.basename(file_to_delete)}")

                print(f"New unique scene detected. Saving frame.")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(SAVE_DIR, f"capture_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                
                saved_frames_gray.append(current_frame_gray)
                saved_filenames.append(filename)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print("Stopping...")
        cap.release()
        cv2.destroyAllWindows()
        communicator.close()