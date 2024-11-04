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

# �ʿ��� ���丮 ����
mkdir -p \
    /var/lib/kubelet \
    /var/lib/kubernetes \
    /var/run/kubernetes \
    /var/lib/cni \
    /etc/cni/net.d \
    /opt/cni/bin

# �ʿ��� ����Ʈ ����
mount --make-shared /
mount --make-shared /sys
mount --make-shared /sys/fs/cgroup
mount --make-shared /var/lib/kubelet

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

# CNI ����
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

# crictl ����
cat > /etc/crictl.yaml <<EOF
runtime-endpoint: unix:///run/containerd/containerd.sock
image-endpoint: unix:///run/containerd/containerd.sock
timeout: 10
debug: false
EOF

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

# Kubernetes apt ����� ����
mkdir -p /etc/apt/keyrings
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.24/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
chmod 644 /etc/apt/keyrings/kubernetes-apt-keyring.gpg

echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.24/deb/ /" | tee /etc/apt/sources.list.d/kubernetes.list

# ��Ű�� ��ġ
apt-get update
apt-get install -y kubelet kubeadm kubectl
apt-mark hold kubelet kubeadm kubectl

# containerd ���� �� �����
pkill containerd || true
sleep 2

# containerd ���� ����
mkdir -p /run/containerd
containerd > /var/log/containerd.log 2>&1 &

# containerd�� ���۵� ������ ���
sleep 10

# kubelet ����
cat > /var/lib/kubelet/config.yaml <<EOF
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
cgroupDriver: cgroupfs
EOF

# kubelet ���� ����
kubelet --config=/var/lib/kubelet/config.yaml --container-runtime=remote --container-runtime-endpoint=unix:///run/containerd/containerd.sock --pod-infra-container-image=registry.k8s.io/pause:3.9 > /var/log/kubelet.log 2>&1 &

sleep 10

# ������ ����� join Ŀ�ǵ带 ��ٸ�
echo "Worker node setup completed. Please run the join command from the master node."
echo "The join command can be found in the master node's /join-command file."
echo "Copy the contents of that file and run it on this worker node."

# ���� ��� (�����̳ʰ� ������� �ʵ���)
tail -f /dev/null