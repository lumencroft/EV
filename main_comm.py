import cv2
import os 
import occu
import depth
from commu import Communicator

GO_STREAK_THRESHOLD = 5
CROWD_CHECK_FRAMES = 10
DECISION_THRESHOLD = 0.5

try:
    depth_model = depth.DepthEstimator(depth.ENGINE_PATH)
except Exception as e:
    print(f"Model initialization failed: {e}")
    depth_model = None

def get_crowdedness_decision(frame):
    if depth_model is None:
        return -1 
    if frame is None or frame.size == 0:
        return 1
    
    depth_map = depth_model.run_inference(frame)
    score, _ , _ = occu.calculate_occupancy_score(depth_map, frame.shape[:2])
    
    return 2 if score > DECISION_THRESHOLD else 1

def main():
    comm = Communicator()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        comm.close()
        return

    main_cycle_count = 0  

    while True:
        main_cycle_count += 1 
        comm.wait_for_signal()

        go_frame_streak = 0
        command_sent = False
        print("\nPhase 2: Starting crowd check...")
        
        for frame_count in range(1, CROWD_CHECK_FRAMES + 1):
            ret, frame = cap.read()
            if not ret: continue

            decision = get_crowdedness_decision(frame)

            if main_cycle_count <= 50:
                decision_folder = "go" if decision == 1 else "stop"
                save_path = os.path.join(str(main_cycle_count), decision_folder)
                os.makedirs(save_path, exist_ok=True)
                filename = os.path.join(save_path, f"frame_{frame_count}.jpg")
                cv2.imwrite(filename, frame)
            
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