import socket
import struct
import cv2
import time  # time 모듈 추가
import door_status_module as door_checker
import crowdedness_module as crowd_checker

ROBOT_HMI_IP = '192.168.1.30'
JETSON_AI_IP = '0.0.0.0'
PORT = 5001

START_BYTE = b'POLA'
ID_EV_RECOG_INFO = 109
HEADER_FORMAT = '<4sHH'
EV_RECOG_INFO_PAYLOAD_FORMAT = '<BBBB4x'

def create_full_spec_packet(payload_state):
    header = struct.pack(HEADER_FORMAT, START_BYTE, ID_EV_RECOG_INFO, 16)
    payload = struct.pack(
        EV_RECOG_INFO_PAYLOAD_FORMAT,
        payload_state['activate_status'],
        payload_state['door_status'],
        payload_state['boarding_status'],
        payload_state['crowdedness']
    )
    return header + payload

def main():
    base_payload_state = {
        'activate_status': 5,
        'door_status': 1,
        'boarding_status': 1,
    }
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((JETSON_AI_IP, PORT))
    
    cap = cv2.VideoCapture(0)

    print(f"AI 시뮬레이터 시작 (수신 대기: {JETSON_AI_IP}:{PORT})")

    while True:
        print("\n" + "="*50)
        print(f"HMI({ROBOT_HMI_IP})로부터 시작 신호를 기다립니다...")
        
        data, addr = sock.recvfrom(1024)
        
        if addr[0] == ROBOT_HMI_IP:
            print(f"\n📬 HMI 신호 수신! [1단계] 문 열림 감지를 시작합니다.")
            
            door_open_streak = 0  # 문 열림 연속 프레임 카운터
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                door_status = door_checker.get_door_status(frame)
                
                if door_status == 1: # 문이 열렸으면
                    door_open_streak += 1
                else: # 문이 닫혔으면
                    door_open_streak = 0 # 카운터 초기화
                
                status_text = "Open" if door_status == 1 else "Closed"
                print(f"  Checking Door Status... {status_text} (Streak: {door_open_streak})")
                
                # cv2.imshow("Door Check", frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'): break

                if door_open_streak >= 5: # 5프레임 연속으로 열림이 감지되면
                    print(f"\n✅ 5프레임 연속 문 열림 감지! 0.5초 후 혼잡도 측정을 시작합니다.")
                    time.sleep(0.5) # 0.5초 대기
                    # cv2.destroyWindow("Door Check")
                    break
            
            go_frame_streak = 0
            command_sent = False

            for frame_count in range(1, 21):
                ret, frame = cap.read()
                if not ret: break

                decision = crowd_checker.get_crowdedness_decision(frame)
                
                if decision == 1:
                    go_frame_streak += 1
                else:
                    go_frame_streak = 0
                
                print(f"  Frame {frame_count:2d}: Decision={decision} (Go Streak: {go_frame_streak})")
                
                # cv2.imshow("Crowdedness Check", frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'): break

                if go_frame_streak >= 10:
                    print(f"\n✅ 15프레임 연속 'Go' 감지! HMI에게 Go 명령을 전송합니다.")
                    final_payload = base_payload_state.copy()
                    final_payload['door_status'] = 1
                    final_payload['crowdedness'] = 1
                    packet_to_send = create_full_spec_packet(final_payload)
                    sock.sendto(packet_to_send, (ROBOT_HMI_IP, PORT))
                    command_sent = True
                    break
            
            # cv2.destroyWindow("Crowdedness Check")

            if not command_sent:
                print(f"\n❌ 타임아웃! HMI에게 Stop 명령을 전송합니다.")
                final_payload = base_payload_state.copy()
                final_payload['door_status'] = 1
                final_payload['crowdedness'] = 2
                packet_to_send = create_full_spec_packet(final_payload)
                sock.sendto(packet_to_send, (ROBOT_HMI_IP, PORT))

    cap.release()
    # cv2.destroyAllWindows()
    sock.close()

if __name__ == '__main__':
    main()