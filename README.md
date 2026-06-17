# EV (Elevator) Recognition System for Robot Navigation

## 프로젝트 개요
이 프로젝트는 로봇이 엘리베이터에 탑승할 때, 내부의 혼잡도를 파악하고 문 열림 상태를 파악하여 안전하게 탑승할 수 있도록 지원하는 비전 기반 AI 시스템입니다. NVIDIA Jetson Orin Nano와 같은 엣지 환경에서 TensorRT를 활용해 실시간 추론을 수행하며, 로봇 제어부(HMI)와 UDP 통신을 통해 탑승 가능 여부(GO/STOP)를 제어합니다.

## 주요 기능 및 특징
* **엘리베이터 내부 혼잡도 산출**: Depth-Anything 모델을 기반으로 뎁스 맵(Depth Map)을 생성한 후, 등고선(Contour)과 볼록 선체(Convex Hull)의 면적 비율을 계산하여 내부의 사람 및 장애물 밀집도를 수치화합니다.
* **조명 기반 문 열림 감지**: 엘리베이터 특정 관심 영역(ROI) 내 천장 조명의 밝기 픽셀 비율을 분석하여 엘리베이터 문의 개폐 상태를 시각적으로 감지합니다.
* **실시간 UDP 통신 제어**: 로봇 HMI(192.168.1.30)로부터 시작 신호를 수신받고, 혼잡도를 분석한 뒤 탑승 명령(GO: 1, STOP: 2) 패킷을 반환합니다. 10프레임 동안 연속해서 탑승 가능 판정이 5회 이상 유지될 때 최종 'GO' 명령을 하달하여 안정성을 높였습니다.

## 시스템 아키텍처 및 동작 흐름
1. **대기 (Standby)**: HMI 시스템으로부터 엘리베이터 인식 시작 신호 대기.
2. **캡처 및 추론 (Capture & Inference)**: 웹캠으로부터 프레임을 획득하고 TensorRT로 구동되는 Depth 모델을 통과시켜 뎁스 맵 도출.
3. **판단 (Decision)**: 도출된 뎁스 맵을 기반으로 혼잡도 점수(Occupancy Score)를 계산하여 탑승 가능 여부 판단.
4. **결과 피드백 (Feedback)**: 산출된 상태를 UDP 패킷으로 HMI에 전송.

## 환경 설정 (Environment)
* **Target Device**: NVIDIA Jetson Orin Nano
* **네트워크 구성**:
  * Jetson AI (Host): `21` (Port: 5001)
  * Robot HMI: `30`

## 파일 구성 및 역할 (File Description)
### 핵심 로직 모듈
* **`main_comm.py`**: 프로젝트의 메인 실행 파일입니다. 비전 데이터 수집, 혼잡도 판단, 네트워크 통신 등 전체 사이클을 제어합니다. 부팅 시 자동 실행되도록 서비스에 등록됩니다.
* **`depth.py`**: TensorRT(`depth.engine`)와 PyCUDA를 활용하여 입력 프레임에 대한 Depth Estimation 추론을 고속으로 수행하는 모듈입니다.
* **`occu.py`**: `depth.py`에서 얻은 뎁스 맵을 기반으로 등고선을 추출하고, Convex Hull 면적비 알고리즘을 통해 혼잡도 수치를 계산하는 모듈입니다.
* **`door.py`**: 엘리베이터 천장 조명의 밝기를 이용해 문 열림 상태를 감지하는 비전 처리 모듈입니다.
* **`commu.py`**: HMI와의 UDP 통신을 전담하며, 정해진 프로토콜 규격(Header/Payload 포맷)에 따라 패킷을 파싱하고 송수신합니다.

### 시스템 및 네트워크 스크립트
* **`create_service.sh`**: `main_comm.py`를 리눅스 systemd 데몬 서비스(`EV_recognition.service`)로 등록하여 재부팅 시 자동으로 백그라운드에서 실행되도록 구성하는 배시 스크립트입니다.
* **`setnetwork.sh`**: Jetson 보드의 이더넷 IP 주소를 `192.168.1.21/24`로 고정하는 네트워크 설정 스크립트입니다.
* **`depth.engine`**: TensorRT로 빌드된 추론 가중치 파일입니다 (Jetson Orin Nano 전용으로 하드웨어에 종속적입니다).

### 디버깅 모듈 (Debugging)
* **`depth_test.py`**, **`door_test.py`**, **`commu_test.py`**: 혼잡도 시각화, 문 열림 상태 및 통신 상태를 독립적으로 검증하고 테스트하기 위한 디버깅용 스크립트 모음입니다.

## 한계점 및 향후 개선 과제 (Known Issues & Future Works)
현재 구현된 `door.py`의 조명 기반 문 열림 감지 로직은 엘리베이터의 특정 조명 환경에 종속적으로 작동하므로, 범용적으로 사용하기 어려운 한계가 있습니다. 
* **해결 방안 1**: Depth 데이터를 활용해 물리적인 문의 개폐를 직접 추적하는 범용적 로직으로 마이그레이션이 필요합니다.
* **해결 방안 2 (Bypass)**: 해당 과정을 생략하고, HMI에서 엘리베이터 도착 신호를 받은 후 '통상적인 문 열림 소요 시간'만큼 `sleep` 대기를 거친 후 진입하도록 로직을 단순화하는 우회 방법 적용이 가능합니다.
