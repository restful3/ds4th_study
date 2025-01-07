#!/bin/bash
set -e

# 기본 시스템 업데이트 및 필수 패키지 설치
apt-get update
apt-get install -y \
    curl \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    iproute2 \
    iptables \
    procps \
    mount

# 기존 프로세스 및 파일 정리
pkill containerd || true
pkill kubelet || true
rm -f /run/containerd/containerd.sock
rm -f /var/lib/kubelet/config.yaml
rm -rf /etc/kubernetes/manifests/*
rm -f /etc/kubernetes/kubelet.conf

# 필요한 디렉토리 생성
mkdir -p \
    /var/lib/kubelet \
    /var/lib/kubernetes \
    /var/run/kubernetes \
    /var/lib/cni \
    /etc/cni/net.d \
    /opt/cni/bin \
    /etc/kubernetes/manifests

# containerd 설치
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
echo \
  "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y containerd.io

# swap 비활성화
swapoff -a

# CNI 플러그인 설치
curl -L "https://github.com/containernetworking/plugins/releases/download/v1.2.0/cni-plugins-linux-amd64-v1.2.0.tgz" | tar -C /opt/cni/bin -xz

# crictl 설치
VERSION="v1.24.0"
curl -L "https://github.com/kubernetes-sigs/cri-tools/releases/download/$VERSION/crictl-$VERSION-linux-amd64.tar.gz" | tar -C /usr/local/bin -xz

# containerd 설정
mkdir -p /etc/containerd
cat > /etc/containerd/config.toml << EOF
version = 2
root = "/var/lib/containerd"
state = "/run/containerd"

[grpc]
  address = "/run/containerd/containerd.sock"
  uid = 0
  gid = 0

[plugins]
  [plugins."io.containerd.grpc.v1.cri"]
    sandbox_image = "registry.k8s.io/pause:3.9"
    [plugins."io.containerd.grpc.v1.cri".containerd]
      default_runtime_name = "runc"
      [plugins."io.containerd.grpc.v1.cri".containerd.runtimes]
        [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc]
          runtime_type = "io.containerd.runc.v2"
          [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc.options]
            SystemdCgroup = false
EOF

# containerd 실행
mkdir -p /run/containerd
containerd > /var/log/containerd.log 2>&1 &

# containerd가 시작될 때까지 대기
sleep 5

# crictl 설정
cat > /etc/crictl.yaml <<EOF
runtime-endpoint: unix:///run/containerd/containerd.sock
image-endpoint: unix:///run/containerd/containerd.sock
timeout: 10
debug: false
EOF

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

# kubelet 설정
cat > /var/lib/kubelet/config.yaml <<EOF
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
cgroupDriver: cgroupfs
authentication:
  anonymous:
    enabled: false
staticPodPath: /etc/kubernetes/manifests
address: 0.0.0.0
healthzBindAddress: 127.0.0.1
healthzPort: 10248
failSwapOn: false
EOF

# kubeadm 설정
cat > /root/kubeadm-config.yaml <<EOF
apiVersion: kubeadm.k8s.io/v1beta3
kind: InitConfiguration
nodeRegistration:
  criSocket: "unix:///run/containerd/containerd.sock"
---
apiVersion: kubeadm.k8s.io/v1beta3
kind: ClusterConfiguration
networking:
  podSubnet: "10.244.0.0/16"
  serviceSubnet: "10.96.0.0/12"
kubernetesVersion: "v1.24.0"
---
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
cgroupDriver: cgroupfs
failSwapOn: false
EOF

# kubelet 설정
cat > /var/lib/kubelet/config.yaml <<EOF
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
cgroupDriver: cgroupfs
authentication:
  anonymous:
    enabled: false
  webhook:
    enabled: false
authorization:
  mode: AlwaysAllow
staticPodPath: /etc/kubernetes/manifests
address: 0.0.0.0
healthzBindAddress: 0.0.0.0
healthzPort: 10248
failSwapOn: false
evictionHard:
  memory.available: "100Mi"
  nodefs.available: "10%"
  nodefs.inodesFree: "5%"
maxPods: 110
EOF

# kubelet 실행
kubelet --config=/var/lib/kubelet/config.yaml \
  --container-runtime=remote \
  --container-runtime-endpoint=unix:///run/containerd/containerd.sock \
  --pod-infra-container-image=registry.k8s.io/pause:3.9 \
  --hostname-override=$(hostname) \
  --root-dir=/var/lib/kubelet \
  --v=5 \
  --fail-swap-on=false \
  --register-node=true \
  > /var/log/kubelet.log 2>&1 &

# kubelet가 시작될 때까지 대기
sleep 10

# healthz 확인
curl -v http://127.0.0.1:10248/healthz || true

# kubeadm 초기화
kubeadm init \
  --config=/root/kubeadm-config.yaml \
  --ignore-preflight-errors=all \
  --v=5

# kubeconfig 설정
mkdir -p $HOME/.kube
cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
chown $(id -u):$(id -g) $HOME/.kube/config

# Flannel CNI 설정
kubectl apply -f https://github.com/flannel-io/flannel/releases/download/v0.20.0/kube-flannel.yml

# Join 노드 설정
kubeadm token create --print-join-command > /join-command

# 디버깅을 위한 정보 출력
echo "===================== DEBUG INFO ====================="
ps aux | grep kubelet
ps aux | grep containerd
ls -la /etc/kubernetes/manifests/
cat /var/log/kubelet.log
echo "==================================================="