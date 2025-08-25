import cv2
import time
import door
import occu
import depth
from commu import Communicator

DOOR_OPEN_STREAK_THRESHOLD = 5
GO_STREAK_THRESHOLD = 10
CROWD_CHECK_FRAMES = 20
DECISION_THRESHOLD = 0.5

try:
    depth_model = depth.DepthEstimator(depth.TENSORRT_ENGINE_PATH)
except Exception as e:
    print(f"Model initialization failed: {e}")
    depth_model = None

def get_crowdedness_decision(frame):
    if depth_model is None:
        return -1 
    if frame is None or frame.size == 0:
        return 1
    
    depth_map = depth_model.run_inference(frame)
    score = occu.calculate_occupancy_score(depth_map, frame.shape[:2])
    
    return 2 if score > DECISION_THRESHOLD else 1

def main():
    comm = Communicator()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        comm.close()
        return

    while True:
        comm.wait_for_signal()

        door_open_streak = 0
        door_opened = False
        print("\nPhase 1: Starting door check...")
        
        for _ in range(99999):
            ret, frame = cap.read()
            if not ret: continue
            
            if door.get_door_status(frame) == 1:
                door_open_streak += 1
            else:
                door_open_streak = 0
            
            print(f"  Door status: {'Open' if door_open_streak > 0 else 'Closed'}. Streak: {door_open_streak}", end='\r')

            if door_open_streak >= DOOR_OPEN_STREAK_THRESHOLD:
                door_opened = True
                print(f"\nDoor confirmed open. Proceeding to crowd check after 0.5s...")
                time.sleep(0.5)
                break
        
        if not door_opened:
            print("\nFailed to detect open door within the time limit. Resetting.")
            continue

        go_frame_streak = 0
        command_sent = False
        print("\nPhase 2: Starting crowd check...")
        
        for frame_count in range(1, CROWD_CHECK_FRAMES + 1):
            ret, frame = cap.read()
            if not ret: continue

            decision = get_crowdedness_decision(frame)
            
            if decision == 1:
                go_frame_streak += 1
            else:
                go_frame_streak = 0
            
            print(f"  Frame {frame_count:2d}/{CROWD_CHECK_FRAMES}: Decision={'GO' if decision==1 else 'STOP'}. Go Streak: {go_frame_streak}")

            if go_frame_streak >= GO_STREAK_THRESHOLD:
                comm.send_command(crowdedness_status=1)
                command_sent = True
                break
        
        if not command_sent:
            comm.send_command(crowdedness_status=2)

    cap.release()
    comm.close()

if __name__ == '__main__':
    main()