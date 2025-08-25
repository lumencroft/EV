import socket
import struct

ROBOT_HMI_IP = '192.168.1.30'
JETSON_AI_IP = '0.0.0.0'
PORT = 5001
START_BYTE = b'POLA'
ID_EV_RECOG_INFO = 109
HEADER_FORMAT = '<4sHH'
PAYLOAD_FORMAT = '<BBBB4x'

class Communicator:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1.0)
        self.sock.bind((JETSON_AI_IP, PORT))
        self.hmi_address = (ROBOT_HMI_IP, PORT)
        print(f"Socket initialized. Listening on {JETSON_AI_IP}:{PORT}")

    def wait_for_signal(self):
        print(f"\nWaiting for start signal (ID 109, Payload 5,0,0,0) from HMI...")
        while True:
            try:
                data, addr = self.sock.recvfrom(1024)
                
                if addr[0] != ROBOT_HMI_IP:
                    continue

                if len(data) < 16:
                    continue
                
                header_data = data[:8]
                payload_data = data[8:16]
                
                _, packet_id, _ = struct.unpack(HEADER_FORMAT, header_data)
                
                if packet_id == ID_EV_RECOG_INFO:
                    actv, door, board, crowd = struct.unpack(PAYLOAD_FORMAT, payload_data)
                    
                    if actv == 5 and door == 0 and board == 0 and crowd == 0:
                        print("Correct start signal received from HMI.")
                        return True

            except socket.timeout:
                continue

    def send_command(self, crowdedness_status):
        payload_state = {
            'activate_status': 5,
            'door_status': 1,
            'boarding_status': 1,
            'crowdedness': crowdedness_status
        }
        header = struct.pack(HEADER_FORMAT, START_BYTE, ID_EV_RECOG_INFO, 16)
        payload = struct.pack(
            PAYLOAD_FORMAT,
            payload_state['activate_status'],
            payload_state['door_status'],
            payload_state['boarding_status'],
            payload_state['crowdedness']
        )
        packet = header + payload
        self.sock.sendto(packet, self.hmi_address)
        status_text = "GO" if crowdedness_status == 1 else "STOP"
        print(f"\nSent {status_text} command to HMI.")

    def close(self):
        self.sock.close()