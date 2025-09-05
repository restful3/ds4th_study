# 15  
로봇을 위한 MLOps 1부  
에지 디바이스와 K8s, 에어플로우  


문종식  
2024.07.17

최근 AI를 비롯한 다양한 기술의 발전에 힘입어 자율주행이 급속도로 발전하고 있습니다. 우아한형제들 역시 자율주행 배달 로봇 기술에 적극적으로 투자해 실외 배달 로봇인 ‘딜리’를 자체 개발했고, 2023년부터 테헤란로 등 여러 장소에서 로봇 배달 시범 서비스를 시행하고 있습니다.*

자율주행 로봇에는 다양한 기술이 필요한데, 예를 들어 머신러닝을 이용해 주변 환경을 인지하고 대응하는 기술을 들 수 있습니다. 딜리의 인지 기술에 대한 자세한 내용은 WOOWACON 2023에서 발표한 (배달 로봇의 주변 환경 인지)**에서 확인할 수 있습니다.

이처럼 머신러닝 모델을 학습시키고 배포하려면 많은 코드와 인프라스트럭처가 필요합니다. 그래서 이런 것들을 체계적으로 관리하는 MLOps라는 분야가 따로 다뤄졌습니다. MLOps란 머신러닝 모델의 개발, 배포, 유지보수 과정을 체계적으로 관리하는 방법론을 뜻합니다. 구글에서 발표한 논문 (Hidden Technical Debt in Machine Learning Systems)**이 MLOps를 잘 소개하고 있는데, 이 논문에서 머신러닝 시스템에서 모델 코드는 중요하지만 전체 시스템 중 작은 부분일 뿐이며, 효율적인 머신러닝 시스템 개발을 위해서는 모델 코드 외에도 다양한 기술이 필요하다는 것을 역설합니다.

머신러닝 시스템의 구성요소***
설정 → 데이터 수집 → 데이터 검증 → 특성 추출 → 머신러닝 모델 생성 → 기계 자원 관리 → 프로세스 관리 도구 → 서빙·모니터링

MLOps 시스템은 조직이 마주한 병목을 체계적으로 관리하고 해결하는 데 주목적인 만큼, 하나의 황금 레시피를 찾을 수는 없습니다. 각자 상황에 맞게 개발 과정을 체계화하고, 생산성을 저하시키는 문제점을 찾아 이를 해결하는 시스템을 구축해야 합니다.

이 글에서는 자율주행 로봇을 위한 머신러닝 모델을 개발하는 과정에서 마주칠 수 있는 문제점들을 알아보고, 이를 해결하기 위해 우아한형제들이 도입한 MLOps 시스템을 소개하겠습니다.

https://www.youtube.com/watch?v=xUGGmGNPYPg  
https://www.youtube.com/watch?v=7QFMnUdZuTo  
https://bit.ly/3zhxHba  
출처 : (Hidden Technical Debt in Machine Learning System) Figure 1

## 머신러닝 모델을 개발하는 과정과 문제들
먼저 머신러닝 모델을 개발하는 과정에서 발생할 수 있는 문제들을 확인하겠습니다. 과정은 크게 세 단계로 나눌 수 있습니다.

1. 데이터 준비
2. 모델 생성
3. 서비스에 모델 적용

‘1. 데이터 준비’, ‘2. 모델 생성’ 단계에서는 일반적으로 재현성(reproducibility)과 추적성(traceability) 문제가 핵심이 됩니다. 이들의 의미는 다음과 같습니다.

재현성: 머신러닝 워크플로의 각 단계에 필요한 데이터 및 환경을 저장해두어, 추후 누군든 같은 모델과 결과를 재현해낼 수 있는 성질

추적성: 모델이 생성된 후에도 모델의 모든 버전과 그에 대한 메타 데이터를 추적할 수 있는 능력

자율주행 로봇을 개발할 때에는 ‘3. 서비스에 모델 적용’ 단계에서 독특한 문제가 발생합니다. 로봇에서는 보통 에지 디바이스라 불리는 소형 저전력 컴퓨터에 의해 코드가 실행되기 때문입니다. 이로 인해 발생할 문제를 생각해보기 전에, 먼저 에지 디바이스가 무엇인지, 이를 사용하기 위해 어떤 작업들이 추가되어야 하는지 알아보겠습니다.

에지 디바이스
에지 디바이스는 서버로 쓰는 컴퓨터와는 여러 면에서 다릅니다. 지연 시간을 줄이고 실시간에 가까운 응답을 얻기 위해 데이터 처리 및 연산을 서버가 아닌 로컬 장치에서 수행할 때, 로컬 장치를 에지 디바이스라고 합니다. 에지 디바이스의 가장 큰 특징은 전력 효율이 좋지만 컴퓨팅 성능이 비교적 떨어진다는 겁니다. 따라서 에지 디바이스에서 머신러닝 모델을 구동하려면 모델의 성능을 어느 정도 포기하더라도 최대한 경량화해야 합니다(우리 팀에서는 이를 ‘모델을 구추닝한다’고 표현합니다).

모델을 ‘구추닝’하기 위해 여러 가지 작업이 필요합니다. 가장 잘 알려진 작업은 모델 추론이 에지 디바이스에서 동작하도록 모델을 변환하는 것으로, 이런 변환에 쓰이는 대표적인 도구로 TensorRT가 있습니다. 변환된 모델의 전체 추론 속도는 물론, 데이터별 병목도 확인해야 합니다. 또한 여러 모델의 추론 동작이 자동스럽지만 스케줄링이 잘 동작하는지도 확인해야 합니다.

다음으로 모델을 더 경량화하려 양자화를 수행하기도 합니다. TensorRT를 이용한 양자화와 INT8, QAT 과정도 이미 ‘13장’ 로봇 머신러닝 모델의 경량화 1부’에서 알아보았습니다.


변환된 모델은 또한 테스트를 거쳐야 합니다. 로봇에 배포하려면 모델의 성능 테스트 이외에도 하드웨어를 포함한 다양한 테스트를 추가적으로 수행해야 하며, 실제 및 엣지에서의 연결부터 동작을 원격에서 확인해야 하며, 시스템 레벨에서 다양한 로그를 확인할 필요도 있습니다. 이런 테스트는 클라우드에서 수행하기 어렵기 때문에 온프레미스on-premise로 관리할 필요성이 생깁니다.

결국 로봇을 위한 머신러닝 모델을 개발하려면 파이썬 언어 등을 사용해 모델을 학습시킬 뿐 아니라 프로그램밍 언어와 도구를 이용해 모델을 변환하고 최적화해야 합니다. 또한 이 결과를 테스트할 수 있는 환경을 구추하고, 테스트 파이프라인을 만들 필요가 있습니다.

이제 로봇을 위한 머신러닝 모델 개발이 어떠한 방식으로 이루어지는지 구체적으로 알아보고, 그 과정에서 MLOps가 해결해야 할 문제들을 살펴보겠습니다.

## 로봇을 위한 머신러닝 개발 과정과 MLOps 시스템이 해결해야 할 문제들

에지 디바이스가 쓰임을 고려해 앞서 언급한 머신러닝 모델 개발의 3단계를 구체화하면 다음과 같습니다.

Training with NVIDIA TensorRT에서 자세히 확인할 수 있습니다.

먼저 학습용 컴퓨터에서 데이터를 준비하고 모델을 학습시킵니다. 때 모델을 학습시킬 뿐 아니라 양자화가 필요한 PTQ 혹은 QAT 작업을 미리 수행해둡니다. 다음으로 학습된 모델을 에지 디바이스로 옮겨 모델을 변환하고, 다양한 테스트 과정을 거쳐야 합니다. 여기서 생겨나는 병목들은 모니터링하고, 그 결과를 모델 학습 과정에 반영시켜 모델을 업그레이드합니다. 최종적으로 성능이 개선되면 테스트를 통과한 모델을 로봇에 배포합니다. 통과하지 못한 경우엔 다시 앞의 과정을 거쳐 더 개선된 모델을 생성합니다. 이런 작업을 하기 위해 해결해야 하는 문제는 크게 두 가지로 요약할 수 있습니다.

1. 재현성, 추적성을 확보할 수 있도록 워크플로를 관리하는 문제
2. 온프레미스 시스템의 자원을 할당하는 문제

먼저 재현성과 추적성을 확보하는 문제를 생각해보겠습니다. 각 모델에 대해 확인할 수 있도록 모델 개발에는 여러 차례의 다양한 학습 및 테스트 과정이 필요하며, 각 단계는 여러 컴퓨터에 걸쳐서 수행되어야 합니다. 또한 파이썬 기반 학습 도구(파이토치, 텐서플로우Tensorflow 등) 외에도 CUDA 코드 및 엔비디아 플랫폼이 제공하는 다양한 도구를 이용해야 하고,

과정에서는 매번 통제해야 할 변수가 많습니다. 때문에 만약 워크플로를 잘 구축해두지 않으면 개발자가 각 컴퓨터에 접속해 파일을 복사하거나 각 단계를 실행해야 하는 불편함이 생길 수 있고, 이에 따라 실수가 발생해 여러 데이터가 많고, 아니면 변수를 명확하게 통제하지 못하게 되어 재현성 및 추적성 확보에도 실패하게 됩니다. 따라서 모든 과정을 파이프라인으로 구축할 필요가 있습니다.

다음으로 자원 할당 문제를 살펴보겠습니다. 개발 인력이 많아지고, 프로젝트의 규모가 확대되어 코드베이스가 커지면, 이에 맞게 모델 학습 및 평가에 더 많은 컴퓨팅 자원이 필요하게 됩니다. 문제는 에지 디바이스를 효율적으로 이용하려면 클라우드 자원을 사용하거나 온프레미스 인프라스트럭처를 구축해 사용해야 한다는 점입니다. 클라우드 등에서 제공되는 다양한 옵션을 사용할 수 없는 상황에서 MLOps 시스템이 다양한 컴퓨팅 자원, 특히 GPU 자원을 잘 관리할 수 있어야 합니다.

K3s와 에어플로 : 자원 관리 솔루션과 워크플로 관리 솔루션
위에서 언급한두 가지 문제를 해결하기 위한 도구인 K3s와 에어플로Airflow를 소개합니다.
먼저 K3s부터 알아보겠습니다. 자원 관리 솔루션으로 알려진 도구로 쿠버네티스(K8s)가 있습니다. 쿠버네티스는 대표적인 컨테이너 오케스트레이션 도구로, 여러 노드(한 대의 물리적 기계 또는 가상 기계들)를 묶어 하나의 클러스터를 형성하고, 클러스터 내의 자원 관리를 자동으로 수행합니다. 대부분 컴퓨팅 작업은 YAML 파일에 기술해두면 필요 시 자원을 할당받을 수 있고, 하나의 엔트리 포인트를 통해 여러 노드에 걸친 자원을 요청하고 할당받을 수 있습니다. 쿠버네티스 클러스터를 구축할 수 있는 방법은 다양합니다. K3s는 그중 가장 간편한 축에 속하는 도구로, CLI 명령어 몇 줄로 클러스터를 구축할 수 있고 ARM 아키텍처 기반이 보다 쉽습니다. 실제 클러스터에서 추가할 수 있는 장점이 있습니다(ARM 아키텍처 기반의 에지 디바이스가 많기 때문에 이 점이 유효한 장점이 됩니다). 물론 전체 기능을 설치하는 것에 비하면 기능의 한계가 있지만 규모가 비교적 작은 시스템에는 효율적인 도구입니다.

다음으로는 에어플로입니다. 에어플로는 대표적인 워크플로 관리 도구로, 방향성 비순환 그래프 DAG*** 파일로 구성하여 다양한 워크플로를 만들고 실행할 수 있습니다. 헬름Helm**을 이용해 쿠버네티스 클러스터에 설치할 수 있으며, KubernetesPodOperator를 이용해 손쉽게 여러 노드에서 프로세스를 실행할 수 있습니다. 이 도구들을 이용한 시스템을 다음 그림과 같이 표현할 수 있습니다.

https://k3s.io  
https://airflow.apache.org  
Directed Acyclic Graph(방향성 비순환 그래프) : 노드(공정의 방향에 있는 간선)으로 이루어진 그래프. 손쉽게 생성할 수 있고, DAG를 각 작업 단위로 구분해 자동화된 워크플로 작성 시 사용됨. 워크플로 전체의 작업 처리, 배분 관리 시스템 등에서 자주 쓰임.
Helm : 쿠버네티스 애플리케이션을 배포, 관리하는 패키지 매니저. 에어플로처럼 클러스터에 쉽게 설치하고 업그레이드 할 수 있도록 도와줌.

* K3s와 에어플로를 이용해 구성한 MLOps 시스템 *

데이터 준비  
모델 학습 (양자화/ PTQ/QAT)  
KPI 추출  
모델 변환  
전처리, 추론/ 모니터링, 통합 KPI 추출  
에어플로  
GPU 워크 스테이션  
GPU 워크 스테이션  
GPU 워크 스테이션  
에지 디바이스  
에지 디바이스  
K3s 클러스터  

이런 구조의 MLOps는 다음과 같이 이용할 수 있습니다. 먼저, 학습용 노드들과 에지 디바이스 노드들이 클러스터 하에 포함되어 있을 때, 엔트리 포인트 하나로 어떤 작업이든 한말만 수 있습니다. 이렇게 하면 온프레미스 시스템 자원 할당 문제를 해결할 수 있습니다.

다음으로 에어플로를 이용해 여러 워크플로를 구축하고, 이를 자동으로 실행할 수 있습니다. 특히 학습 노드와 에지 디바이스 노드가 따로 있는 상태에서 이 두 가지 노드를 모두 사용하는 다양한 테스트를 실행해야 하는데, 이를 DAG로 구성해뒀던 전체 프로세스를 한 번의 클릭만으로 간단하게 실행할 수 있습니다. 에어플로는 Bash 명령어를 실행할 수 있기 때문에 다양한 언어나 플랫폼 등으로 만든 프로그램을 손쉽게 실행할 수 있습니다. 에어플로를 도입하는 것만으로 재현성과 추적성을 완벽하게 확보할 수는 없지만, 이들을 위한 좋은 기반이 될 수 있습니다.

## 설치하기
이제 각 도구별 구축 방법을 알아보겠습니다.

### K3s 설치하기
K3s는 하나의 실행 파일로 간편하게 쿠버네티스 클러스터를 구축할 수 있습니다.

1단계 먼저 마스터(master) 노드에서 다음 명령어로 K3s를 설치 및 실행합니다. 도커를 이용하도록 --docker 옵션을 넣어줍니다.

curl -sFL https://get.k3s.io | sh -s - --docker

2단계 다음으로 워커(worker) 노드를 설치하겠습니다. 먼저 마스터 노드의 /var/lib/rancher/k3s/server/node-token에서 K3s 토큰 값을 확인해야 합니다. 그러면 워커 노드에서 다음 명령어로 K3s를 설치하고 실행할 수 있습니다. 아래에서 master_node_token은 K3s 토큰 값으로 치환해주세요.

curl -sFL https://get.k3s.io | K3S_URL=https://master_ip:6443 K3S_TOKEN=master_node_token sh -s - --docker

클러스터 설치 및 구축이 완료되었습니다. 다음으로 필요한 CLI 도구인 kubectl을 설치합니다.

### kubectl 설치하기  
이제부터는 특별히 언급이 없으면 마스터와 워커 노드 같은 방법으로 설치해줍니다.

1단계 먼저 필요한 패키지를 설치합니다.
```bash
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl
```

2단계 다음으로 구글 클라우드 공개 저장소 사이 키를 다운로드하고, 쿠버네티스 apt 리포지터리를 추가합니다.
```bash
sudo curl -fsSLo /etc/apt/keyrings/kubernetes-archive-keyring.gpg https://packages.cloud.google.com/apt/doc/apt-key.gpg
echo "deb [signed-by=/etc/apt/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list
```

3단계 apt를 이용해 kubectl을 설치합니다.
```bash
sudo apt-get update
sudo apt-get install -y kubectl
```

4단계 이제 kubectl이 사용할 config 파일을 세팅해야 합니다. k3s 설치 시 config 파일이 마스터 노드에 /etc/rancher/k3s/k3s.yaml로 생성되므로 이를 이용합니다. 먼저 마스터 노드에는 다음과 같이 config 파일을 세팅합니다.
```bash
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chmod 600 ~/.kube/config
sudo chown -R $USER ~/.kube
```

5단계 다음으로 워커 노드에서 같은 config 파일을 마스터 노드로부터 복사해 같은 위치에 세팅합니다. 이때 IP 주소를 변경해야 하는데, config 파일을 열고 IP 주소를 `server: https://master_ip:6443`으로 변경합니다.  
이후 다음 명령어를 실행해 kubectl이 정상 동작하는 것을 확인합니다.
```bash
kubectl get pods -A
```

### 엔비디아 GPU 세팅하기  
엔비디아 GPU를 이용하기 위해 엔비디아가 제공하는 DaemonSet을 생성해야 합니다.

1단계 그 전에 도커의 default-runtime을 엔비디아로 변경해야 하므로 /etc/docker/daemon.json을 다음과 같이 변경합니다.
```json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "args": [],
      "path": "nvidia-container-runtime"
    }
  }
}
```

2단계 다음으로 도커를 재실행합니다.
```bash
sudo systemctl restart docker
```

3단계 다음으로 DaemonSet을 생성합니다.
```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/deployments/static/nvidia-device-plugin.yml
```

4단계 배포가 완료되면 다음 명령어로 노드에서 이용 가능한 GPU 개수를 확인합니다.
```bash
kubectl get nodes --o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia.com/gpu
```

여기까지 GPU 지원 관리가 가능한 클러스터를 구축해보았습니다. 다음으로 에어플로를 설치합니다.

### 에어플로 설치하기

쿠버네티스에 패키지를 설치할 때 헬름이 권장됩니다. 에어플로를 헬름으로 설치해보겠습니다.

1단계 먼저 헬름을 설치합니다.
```bash
curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
sudo apt-get install apt-transport-https --yes
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt-get update
sudo apt-get install helm
```

2단계 다음으로 에어플로를 설치합니다. 다음 그림처럼 리포지터리에서 INSTALL을 클릭 후 에어플로 헬름 차트를 다운로드할 수 있습니다. 옵션을 쉽게 변경하기 위해, 또한 이후 항상 관리의 편의성을 위해, helm 명령어를 사용하는 대신 직접 tar 파일을 다운로드할 것을 권장합니다.

*헬름을 이용한 에어플로 설치 방법*

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/30400652/bcd3a2cc-69e8-404c-8323-ae1f4a3cd763/7.jpg)

3단계 에어플로는 DAG와 로그 파일을 읽고 저장할 스토리지가 필요합니다. 온프레미스에서 작업한다면 NAS를 이 스토리지로 이용할 수 있습니다. 적절한 위치에 NFS 타입으로 PV, PVC를 생성하고, 에어플로 헬름 차트의 values.yaml에서 dags와 logs를 다음과 같이 수정합니다.

```yaml
dags:
  persistence:
    enabled: true
    existingClaim: DAG-PVC-NAME
  ...

logs:
  persistence:
    enabled: true
    existingClaim: LOG-PVC-NAME
```

4단계 이후 다음 명령어를 실행하면 에어플로 배포가 완료됩니다.
```bash
kubectl create namespace airflow
helm install airflow -n airflow
```

DAG 예제  
여기까지 자원 관리, 워크플로 관리에 필요한 도구들을 모두 설치했습니다. 이제 이들이 설치된 환경에서 에어플로 DAG를 실행해보며 MLOps 시스템이 목표한 바로 이를 수 있는지 확인하겠습니다. 학습용 컴퓨터와 에지 디바이스를 오가며 명령을 실행하는 상황에 간단한 DAG를 구성하겠습니다.

다음은 학습용 컴퓨터에서 파이토치 버전을 출력한 후 에지 디바이스로 넘어가 TensorRT 버전을 출력하는 간단한 코드입니다.

```python
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.utils.dates import days_ago
from kubernetes.client import V1ResourceRequirements

default_args = {
    "owner": "woowa",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
}

resources = V1ResourceRequirements(requests={"nvidia.com/gpu": "1"}, limits={"nvidia.com/gpu": "1"})

train_affinity = {
    "nodeAffinity": {
        "requiredDuringSchedulingIgnoredDuringExecution": {
            "nodeSelectorTerms": [{
                "matchExpressions": [{
                    "key": "kubernetes.io/hostname",
                    "operator": "In",
                    "values": [GPU_WORKSTATION_HOST_NAME]
                }]
            }]
        }
    }
}
```
```python
edge_affinity = {
    "nodeAffinity": {
        "requiredDuringSchedulingIgnoredDuringExecution": {
            "nodeSelectorTerms": [{
                "matchExpressions": [{
                    "key": "kubernetes.io/hostname",
                    "operator": "In",
                    "values": [EDGE_HOST_NAME]
                }]
            }]
        }
    }
}

dag = DAG("woowa_sample",
    default_args=default_args,
    description="woowa sample dag",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False
)

train_task = KubernetesPodOperator(
    namespace="airflow",
    image="pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel",
    cmds=["bash", "-c"],
    arguments=["python3 -c 'import torch; print(torch.__version__)'"],
    name="train_task",
    task_id="run_train_task",
    in_cluster=True,
    is_delete_operator_pod=True,
    get_logs=True,
    container_resources=resources,
    dag=dag,
    affinity=train_affinity
)

edge_task = KubernetesPodOperator(
    namespace="airflow",
    image="nvcr.io/nvidia/l4t-jetpack:r35.1.0",
    cmds=["bash", "-c"],
    arguments=["dpkg -l | grep nvinfer"],
    name="edge_task",
    task_id="run_edge_task",
    in_cluster=True,
    is_delete_operator_pod=True,
    get_logs=True,
    container_resources=resources,
    dag=dag,
    affinity=edge_affinity
)

train_task >> edge_task
```

train_affinity, edge_affinity를 이용해 실행하는 장소를 명시했으며, resources를 이용해 GPU 자원을 요청한 것을 확인할 수 있습니다. 이 프로세스는 에어플로의 Web UI에서 실행하고 그 결과를 확인할 수 있습니다.

먼저 DAG가 등록된 것을 확인했습니다.

* DAG가 등록된 에어플로 Web UI *

이 DAG를 실행한 결과는 아래 두 이미지와 같습니다. 위 그림에서 파이토치 버전 1.11.0, 아래 그림에서 TensorRT의 C++ API인 nvinfer의 버전은 8.4.1.1-cuda11.4임입니다. 이렇게 KubernetesPodOperator를 이용해 여러 노드를 오가며 프로세스를 실행할 수 있음을 확인했습니다.

* GPU workstation에서 실행된 프로세스의 로그 *

한편 GPU 자원이 잘 할당되고 관리될지 확인할 필요가 있습니다. 먼저 다음 명령어로 노드에서 이용 가능한 GPU 개수를 확인할 수 있습니다.

```
kubectl describe node $GPU_WORKSTATION_HOST_NAME
```

결과는 다음과 같습니다.

```
Capacity:
  cpu:                28
  ephemeral-storage:  1921207520Ki
  hugepages-1Gi:      0
  hugepages-2Mi:      0
  memory:             131568
  pods:               110
```

한 개의 GPU 자원을 요청한 첫 번째 프로세스가 실행되면 GPU를 점유 중인 프로세스가 다음과 같이 변경되는 것을 확인할 수 있습니다.
Allocated resources:
(Total limits may be over 100 percent, i.e., overcommitted.)
Resource         Requests     Limits
---------        --------     ------
cpu              0 (0%)       0 (0%)
memory           0 (0%)       0 (0%)
ephemeral-storage 0 (0%)      0 (0%)
hugepages-1Gi    0 (0%)       0 (0%)
hugepages-2Mi    0 (0%)       0 (0%)
nvidia.com/gpu   1            1

위에서 nvidia.com/gpu가 1를 넘으면 쿠버네티스는 더 이상 프로세스를 할당하지 않고 다른 컴퓨터에 해당 프로세스를 넘기거나 펜딩Pending 상태에서 실행하지 않게 됩니다.

마치며  
여기까지 로봇을 위한 MLOps 시스템이 갖춰야 할 요소들과 이를 구현하는 방법을 알아보았습니다. 에지 파이프라인의 구성은 곧바로 2부에서 알아보겠습니다.
