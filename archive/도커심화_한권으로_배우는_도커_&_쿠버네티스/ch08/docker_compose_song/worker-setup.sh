#!/bin/bash

# 시스템 업데이트 및 필수 패키지 설치
apt-get update
apt-get install -y apt-transport-https ca-certificates curl software-properties-common

# Docker 설치
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
apt-get update
apt-get install -y docker-ce

# 1. 기존 Kubernetes 관련 파일 및 설정 제거
sudo rm -f /etc/apt/sources.list.d/kubernetes.list
sudo rm -f /etc/apt/sources.list.d/docker.list
sudo rm -f /usr/share/keyrings/kubernetes-archive-keyring.gpg
sudo rm -f /etc/apt/keyrings/kubernetes-apt-keyring.gpg

# 2. apt 캐시 정리
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*

# 3. 새로운 Kubernetes apt 저장소 설정
KUBE_VERSION="v1.28"
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://pkgs.k8s.io/core:/stable:/${KUBE_VERSION}/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/${KUBE_VERSION}/deb/ /" | sudo tee /etc/apt/sources.list.d/kubernetes.list
# 4. apt 업데이트 및 오류 확인
sudo apt-get update 2>&1 | tee /tmp/apt-update-output.txt
if grep -q "NO_PUBKEY" /tmp/apt-update-output.txt; then
  MISSING_KEY=$(grep "NO_PUBKEY" /tmp/apt-update-output.txt | awk '{print $NF}')
  sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys $MISSING_KEY
  sudo apt-get update
fi

# 5. Kubernetes 패키지 설치 (선택사항)
# sudo apt-get install -y kubelet kubeadm kubectl
# sudo apt-mark hold kubelet kubeadm kubectl

# 스왑 비활성화
swapoff -a
sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab

# 필요한 커널 모듈 활성화
modprobe overlay
modprobe br_netfilter

# 시스템 설정
cat <<EOF | tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF

sysctl --system