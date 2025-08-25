#!/bin/bash

# -e: 스크립트 실행 중 오류가 발생하면 즉시 중단
# -x: 실행되는 명령어를 터미널에 출력
set -ex

# 1. 패키지 목록 업데이트 및 netplan.io 설치
sudo apt update
sudo apt install -y netplan.io

# 2. /etc/netplan/01-netcfg.yaml 파일 생성
# cat과 EOF를 사용하여 파일 내용을 직접 작성합니다.
# sudo tee를 사용하여 root 권한으로 파일을 씁니다.
sudo tee /etc/netplan/01-netcfg.yaml > /dev/null <<EOF
network:
  version: 2
  renderer: networkd
  ethernets:
    eno1:
      dhcp4: no
      addresses:
        - 192.168.1.21/24
      routes:
        - to: default
          via: 192.168.1.254
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
EOF

# 3. 새로운 네트워크 설정 적용
sudo netplan apply

echo "네트워크 설정이 성공적으로 적용되었습니다."