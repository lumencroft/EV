#!/bin/bash

# -e: 스크립트 실행 중 오류가 발생하면 즉시 중단
set -e

# --- 설정 변수 (여기만 수정하면 됩니다) ---
SERVICE_NAME="my_script.service"
DESCRIPTION="My Python Script Service"
PYTHON_SCRIPT_PATH="/home/$USER/EV/main_comm.py" # $USER 변수로 현재 사용자 홈 디렉토리 자동 설정
WORKING_DIRECTORY="/home/$USER/EV"

# -----------------------------------------

SERVICE_FILE_PATH="/etc/systemd/system/$SERVICE_NAME"

echo ">> 현재 사용자($USER)를 위한 systemd 서비스 파일을 생성합니다..."

# 1. /etc/systemd/system/ 경로에 서비스 파일 생성
# $USER 변수가 현재 사용자의 이름으로 자동 치환됩니다.
sudo tee $SERVICE_FILE_PATH > /dev/null <<EOF
[Unit]
Description=$DESCRIPTION
After=network.target

[Service]
ExecStart=/usr/bin/python3 $PYTHON_SCRIPT_PATH
WorkingDirectory=$WORKING_DIRECTORY
StandardOutput=inherit
StandardError=inherit
Restart=always
User=$USER

[Install]
WantedBy=multi-user.target
EOF

echo ">> 서비스 파일 생성 완료: $SERVICE_FILE_PATH"

# 2. systemd 데몬 리로드
echo ">> systemd 데몬을 리로드합니다..."
sudo systemctl daemon-reload

# 3. 서비스 활성화 (부팅 시 자동으로 시작되도록 설정)
echo ">> 서비스를 활성화합니다..."
sudo systemctl enable $SERVICE_NAME

echo "✅ 성공! '$SERVICE_NAME' 서비스가 시스템에 등록되고 활성화되었습니다."
echo "sudo systemctl start $SERVICE_NAME 명령어로 지금 바로 서비스를 시작할 수 있습니다."