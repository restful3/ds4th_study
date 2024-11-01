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

# 필요한 디렉토리 생성
mkdir -p \
    /var/lib/kubelet \
    /var/lib/kubernetes \
    /var/run/kubernetes \
    /var/lib/cni \
    /etc/cni/net.d \
    /opt/cni/bin

# 필요한 마운트 설정
mount --make-shared /
mount --make-shared /sys
mount --make-shared /sys/fs/cgroup
mount --make-shared /var/lib/kubelet

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

# CNI 설정
cat > /etc/cni/net.d/10-containerd-net.conflist <<EOF
{
  "cniVersion": "1.0.0",
  "name": "containerd-net",
  "plugins": [
    {
      "type": "bridge",
      "bridge": "cni0",
      "isGateway": true,
      "ipMasq": true,
      "hairpinMode": true,
      "ipam": {
        "type": "host-local",
        "ranges": [
          [
            {
              "subnet": "10.85.0.0/16"
            }
          ]
        ],
        "routes": [
          { "dst": "0.0.0.0/0" }
        ]
      }
    },
    {
      "type": "portmap",
      "capabilities": {"portMappings": true}
    }
  ]
}
EOF

# crictl 설정
cat > /etc/crictl.yaml <<EOF
runtime-endpoint: unix:///run/containerd/containerd.sock
image-endpoint: unix:///run/containerd/containerd.sock
timeout: 10
debug: false
EOF

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

# Kubernetes apt 저장소 설정
mkdir -p /etc/apt/keyrings
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.24/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
chmod 644 /etc/apt/keyrings/kubernetes-apt-keyring.gpg

echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.24/deb/ /" | tee /etc/apt/sources.list.d/kubernetes.list

# 패키지 설치
apt-get update
apt-get install -y kubelet kubeadm kubectl
apt-mark hold kubelet kubeadm kubectl

# containerd 중지 및 재시작
pkill containerd || true
sleep 2

# containerd 직접 실행
mkdir -p /run/containerd
containerd > /var/log/containerd.log 2>&1 &

# containerd가 시작될 때까지 대기
sleep 10

# kubelet 설정
cat > /var/lib/kubelet/config.yaml <<EOF
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
cgroupDriver: cgroupfs
EOF

# kubelet 직접 실행
kubelet --config=/var/lib/kubelet/config.yaml --container-runtime=remote --container-runtime-endpoint=unix:///run/containerd/containerd.sock --pod-infra-container-image=registry.k8s.io/pause:3.9 > /var/log/kubelet.log 2>&1 &

sleep 10

# 마스터 노드의 join 커맨드를 기다림
echo "Worker node setup completed. Please run the join command from the master node."
echo "The join command can be found in the master node's /join-command file."
echo "Copy the contents of that file and run it on this worker node."

# 무한 대기 (컨테이너가 종료되지 않도록)
tail -f /dev/null