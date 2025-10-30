# EV
main_comm.py - 모든 작업을 하는 메인코드. 시작프로그램에 등록되어있음.

door.py - 문열림을 엘리베이터 천장의 조명을 이용해서 감지.

door_test.py - 웹캠을 실행하고 door.py 의 작동을 검증하는 디버깅용.

depth.py - DepthEstimator 설정.

occu.py - depth_map을 이용해서 내부의 혼잡도 산출.

depth_test.py - 위 2개를 이용해서 웹캠을 켜고 혼잡도 산출에 쓰이는 등고선과 수치가 나오는 디버깅용.

commu.py - UDP 통신 관리. 



---
create_service.sh - 프로젝트를 서비스로 등록


setnetwork.sh - 이더넷 아이피를 192.168.1.21/24 로 고정.


depth_engine - tensorrt로 빌드된 depth_anything의 inference 파일 (하드웨어 종속이라서 jetson orin nano에서만 작동 보장됨)


---
door 관련은 일부 엘리베이터에 한해서 작동하므로 (일반적이지 않음) depth로부터 열림을 감지하는 로직을 짜야한다.


아니면 이부분을 bypass해버리고, 로봇마다 도착신호->문이 열림 만큼의 간격을 측정하고 sleep 해서 진행하는것이 가장 간단한 해결책

