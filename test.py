import socket
import struct
# cv2, door_status_module, crowdedness_module 임포트는 더 이상 필요 없으므로 삭제합니다.

# --- 상수 정의 (기존과 동일) ---
ROBOT_HMI_IP = '192.168.1.30'
JETSON_AI_IP = '192.168.1.21'
PORT = 5001

START_BYTE = b'POLA'
ID_EV_RECOG_INFO = 109
HEADER_FORMAT = '<4sHH'
EV_RECOG_INFO_PAYLOAD_FORMAT = '<BBBB4x'

def create_full_spec_packet(payload_state):
    """주어진 payload 상태로 전체 UDP 패킷을 생성합니다."""
    # 헤더 생성: 시작 바이트, ID, 페이로드 길이(8)
    header = struct.pack(HEADER_FORMAT, START_BYTE, ID_EV_RECOG_INFO, 8)
    
    # 페이로드 생성
    payload = struct.pack(
        EV_RECOG_INFO_PAYLOAD_FORMAT,
        payload_state['activate_status'],
        payload_state['door_status'],
        payload_state['boarding_status'],
        payload_state['crowdedness']
    )
    return header + payload

def get_user_crowdedness_input():
    """사용자로부터 유효한 혼잡도 값(1 또는 2)을 입력받습니다."""
    while True:
        try:
            # 사용자에게 입력 안내 메시지 출력
            user_input = input(" crowdedness 값을 입력하세요 (1: Go, 2: Stop): ")
            decision = int(user_input)
            
            # 입력값이 1 또는 2인지 확인
            if decision in [1, 2]:
                return decision
            else:
                print("❌ 잘못된 입력입니다. 반드시 1 또는 2를 입력해주세요.")
        except ValueError:
            # 숫자가 아닌 값이 입력된 경우 예외 처리
            print("❌ 잘못된 입력입니다. 숫자 1 또는 2를 입력해주세요.")

def main():
    """메인 실행 함수"""
    # 전송할 패킷의 기본 상태 값
    base_payload_state = {
        'activate_status': 5,
        'door_status': 0,
        'boarding_status': 5,
    }
    
    # UDP 소켓 설정
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((JETSON_AI_IP, PORT))
    
    print(f"AI 시뮬레이터 시작 (수신 대기: {JETSON_AI_IP}:{PORT})")

    try:
        while True:
            print("\n" + "="*50)
            print(f"HMI({ROBOT_HMI_IP})로부터 시작 신호를 기다립니다...")
            
            # HMI로부터 데이터 수신 대기
            data, addr = sock.recvfrom(1024)
            
            # 수신된 데이터의 IP 주소가 HMI IP와 일치하는지 확인
            if addr[0] == ROBOT_HMI_IP:
                print(f"\n📬 HMI 신호 수신! 사용자 입력을 받습니다.")
                
                # 사용자로부터 혼잡도 값(1 또는 2) 입력받기
                crowdedness_input = get_user_crowdedness_input()

                # 전송할 최종 페이로드 생성
                final_payload = base_payload_state.copy()
                final_payload['door_status'] = 1  # 문은 열렸다고 가정
                final_payload['crowdedness'] = crowdedness_input
                
                # 최종 패킷 생성
                packet_to_send = create_full_spec_packet(final_payload)
                
                # HMI로 패킷 전송
                sock.sendto(packet_to_send, (ROBOT_HMI_IP, PORT))
                
                status_text = "Go" if crowdedness_input == 1 else "Stop"
                print(f"✅ HMI에게 '{status_text}'({crowdedness_input}) 명령을 전송했습니다.")

    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    finally:
        # 소켓 리소스 정리
        sock.close()

if __name__ == '__main__':
    main()