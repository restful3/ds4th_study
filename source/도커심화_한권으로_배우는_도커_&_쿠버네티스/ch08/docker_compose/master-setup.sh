#!/bin/bash
set -e

# �⺻ �ý��� ������Ʈ �� �ʼ� ��Ű�� ��ġ
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

# ���� ���μ��� �� ���� ����
pkill containerd || true
pkill kubelet || true
rm -f /run/containerd/containerd.sock
rm -f /var/lib/kubelet/config.yaml
rm -rf /etc/kubernetes/manifests/*
rm -f /etc/kubernetes/kubelet.conf

# �ʿ��� ���丮 ����
mkdir -p \
    /var/lib/kubelet \
    /var/lib/kubernetes \
    /var/run/kubernetes \
    /var/lib/cni \
    /etc/cni/net.d \
    /opt/cni/bin \
    /etc/kubernetes/manifests

# containerd ��ġ
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
echo \
  "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y containerd.io

# swap ��Ȱ��ȭ
swapoff -a

# CNI �÷����� ��ġ
curl -L "https://github.com/containernetworking/plugins/releases/download/v1.2.0/cni-plugins-linux-amd64-v1.2.0.tgz" | tar -C /opt/cni/bin -xz

# crictl ��ġ
VERSION="v1.24.0"
curl -L "https://github.com/kubernetes-sigs/cri-tools/releases/download/$VERSION/crictl-$VERSION-linux-amd64.tar.gz" | tar -C /usr/local/bin -xz

# containerd ����
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

# containerd ����
mkdir -p /run/containerd
containerd > /var/log/containerd.log 2>&1 &

# containerd�� ���۵� ������ ���
sleep 5

# crictl ����
cat > /etc/crictl.yaml <<EOF
runtime-endpoint: unix:///run/containerd/containerd.sock
image-endpoint: unix:///run/containerd/containerd.sock
timeout: 10
debug: false
EOF

# Kubernetes apt ����� ����
mkdir -p /etc/apt/keyrings
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.24/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
chmod 644 /etc/apt/keyrings/kubernetes-apt-keyring.gpg

echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.24/deb/ /" | tee /etc/apt/sources.list.d/kubernetes.list

# ��Ű�� ��ġ
apt-get update
apt-get install -y kubelet kubeadm kubectl
apt-mark hold kubelet kubeadm kubectl

# kubelet ����
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

# kubeadm ����
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

# kubelet ����
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

# kubelet ����
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

# kubelet�� ���۵� ������ ���
sleep 10

# healthz ��������Ʈ Ȯ��
curl -v http://127.0.0.1:10248/healthz || true

# kubeadm �ʱ�ȭ
kubeadm init \
  --config=/root/kubeadm-config.yaml \
  --ignore-preflight-errors=all \
  --v=5

# kubeconfig ����
mkdir -p $HOME/.kube
cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
chown $(id -u):$(id -g) $HOME/.kube/config

# Flannel CNI ��ġ
kubectl apply -f https://github.com/flannel-io/flannel/releases/download/v0.20.0/kube-flannel.yml

# Join ���ɾ� ����
kubeadm token create --print-join-command > /join-command

# ������� ���� ���� ���
echo "===================== DEBUG INFO ====================="
ps aux | grep kubelet
ps aux | grep containerd
ls -la /etc/kubernetes/manifests/
cat /var/log/kubelet.log
echo "==================================================="