# EV
main_comm.py - 모든 작업을 하는 메인코드. 시작프로그램에 등록되어있음.

door.py - 문열림을 엘리베이터 천장의 조명을 이용해서 감지.

door_test.py - 웹캠을 실행하고 door.py 의 작동을 검증하는 디버깅용.

depth.py - DepthEstimator 설정.

occu.py - depth_map을 이용해서 내부의 혼잡도 산출.

depth_test.py - 위 2개를 이용해서 웹캠을 켜고 혼잡도 산출에 쓰이는 등고선과 수치가 나오는 디버깅용.

commu.py - UDP 통신 관리. 
