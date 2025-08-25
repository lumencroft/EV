import socket
import struct

ROBOT_HMI_IP = '192.168.1.30'
JETSON_AI_IP = '192.168.1.21'
PORT = 5001

START_BYTE = b'POLA'
ID_EV_RECOG_INFO = 109
HEADER_FORMAT = '<4sHH'
EV_RECOG_INFO_PAYLOAD_FORMAT = '<BBBB4x'

def create_full_spec_packet(payload_state):
    header = struct.pack(HEADER_FORMAT, START_BYTE, ID_EV_RECOG_INFO, 8)
    
    payload = struct.pack(
        EV_RECOG_INFO_PAYLOAD_FORMAT,
        payload_state['activate_status'],
        payload_state['door_status'],
        payload_state['boarding_status'],
        payload_state['crowdedness']
    )
    return header + payload

def get_user_crowdedness_input():
        return int(input(" crowdedness 값을 입력하세요 (1: Go, 2: Stop): "))
            

def main():
    base_payload_state = {
        'activate_status': 5,
        'door_status': 1,
        'boarding_status': 5,
    }
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((JETSON_AI_IP, PORT))
    
    print(f"AI 시뮬레이터 시작 (수신 대기: {JETSON_AI_IP}:{PORT})")

    try:
        while True:
            print("\n" + "="*50)
            print(f"HMI({ROBOT_HMI_IP})로부터 시작 신호를 기다립니다...")
            
            data, addr = sock.recvfrom(1024)
            
            if addr[0] == ROBOT_HMI_IP:
                print(f"\n📬 HMI 신호 수신! 사용자 입력을 받습니다.")
                
                crowdedness_input = get_user_crowdedness_input()

                final_payload = base_payload_state.copy()
                final_payload['crowdedness'] = crowdedness_input
                
                packet_to_send = create_full_spec_packet(final_payload)
                
                sock.sendto(packet_to_send, (ROBOT_HMI_IP, PORT))
                
                status_text = "Go" if crowdedness_input == 1 else "Stop"
                print(f"✅ HMI에게 '{status_text}'({crowdedness_input}) 명령을 전송했습니다.")

    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    finally:
        sock.close()

if __name__ == '__main__':
    main()