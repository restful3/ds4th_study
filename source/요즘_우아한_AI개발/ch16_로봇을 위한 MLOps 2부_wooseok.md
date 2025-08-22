에지 파이프라인의 구성

송동일  
2024.07.19

➊ 15장 ‘로봇을 위한 MLOps 1부 : 에지 디바이스와 K8s, 에어플로우’에서는 로봇 기반 머신러닝 모델 개발 과정과 이를 뒷받침하는 MLOps 인프라스트럭처 구축 방법을 설명했습니다. GPU 워크스테이션 및 에지 디바이스들에 워커(worker) 노드를 설치하고 쿠버네티스 및 에어플로우를 사용해 이들 노드에서 전체 파이프라인을 실행하는 방법을 살펴보았습니다.

여기에서는 에지 디바이스에서 작동하는 에지 파이프라인(edge pipeline) 구축하는 방법, 그리고 이 파이프라인에서 사용한 도구와 특징을 소개합니다. 특히 다중 모델을 동시에 추론할 때 성능을 평가하는 자체 개발 도구 Triton Inferencesight도 소개하오니 기대해주세요.

참고로 이 글과 같은 프로젝트를 다른 관점에서 살펴본 연재 장들은 다음과 같습니다.

➊ 13장 로봇 머신러닝 모델의 경쟁력 1부 : 훈련 효과
➊ 15장 로봇을 위한 MLOps 1부 : 에지 디바이스와 K8s, 에어플로우

## 에지 파이프라인의 필요성  
로봇, 자동차, 드론 등에 쓰이는 자율주행 기술은 일상에서 흔히 접할 수 있게 되었습니다. 이런 장치들은 우리가 사용하는 컴퓨터와 유사한 기능을 하는 하드웨어가 탑재되어 있습니다. 이 하드웨어는 일반적으로 ARM 아키텍처를 기반으로 하는 CPU를 사용합니다. ARM 기반 CPU는 낮은 전력 소모 특이로 발열도 적고 베터리도 적게 사용한다는 장점이 있기 때문입니다. 또한 하드웨어에는 보통 예약 GPU를 포함하고 있어, 그래서 디스플레이 처리뿐만 아니라 딥러닝 모델의 추론, 이미지 및 비디오 처리 등 고성능의 병렬 연산을 요구하는 작업을 수행할 수 있습니다.

우아한형제들 자율주행 로봇 ‘딜리’에도 CPU, GPU뿐만 아니라 딥러닝 가속기*, 영상 이미지 합성기**, HW 인코더/디코더*** 등과 함께 여러 센서 인터페이스가 패키징된 엔비디아 젯슨 플랫폼을 기반으로 하다는 임베디드 보드가 내장되어 있습니다. 임베디드 보드의 GPU, LiDAR, 라, 카메라 등 다양한 센서들을 로봇에 연결해, 자율주행과 관련된 복잡한 연산을 로봇에서 효율적으로 수행합니다. 이와 같은 임베디드 보드가 우리의 에지 디바이스가 됩니다.

* Deep Learning Accelerator, DLA : 딥러닝 작업의 빠르고 효율적으로 수행하기 위해 설계된 하드웨어 장치. GPU 또는 SoC에 포함. 주요 GPU, TPU 같은 장동 하드웨어가 사용되며, 영상 처리, 모델의 온컬한 연산을 빠르게 수행하기 위해 사용된다.
** Video Image Compositor, VIC : 여러 개의 비디오나 이미지로 결합하여 하나의 최종 비디오 또는 이미지를 만드는 하드웨어에 탑재된 도구. 이 과정에서 다양한 시각 효과, 라이팅, 트랜지션 등을 적용할 수 있다.
**Inertial Measurement Unit(관성 센서)** : 가속도계와 자이로스코프를 이용해 실험체의 가속도, 회전 속도, 기울기 등의 정보를 측정하는 장치. 로봇, 드론, 자율주행 자동차 등에서 위치 추정에 사용된다.
**LiDAR(Light Detection and Ranging)**  : 레이저 빛을 사용해 물체까지의 거리를 측정하는 장치. 주로 자율주행 차, 드론, 지도 생성 등에 사용된다. 레이저 빛을 발사한 반사된 신호를 기반으로 주변 환경의 3D 지도를 만드는 데 활용된다.

### 에지 디바이스에서의 AI 연산이 필요한 이유  
최근 온디바이스on-device AI라는 용어를 많이 들어봤을 텐데요. 이는 클라우드나 기타 서버에 의존하지 않고 에지 디바이스 자체에서 AI 연산을 수행하는 것을 말합니다. 이를 위해서는 디바이스 내부의 연산 작업만을 할 수 있도록 연산을 하는 기술이 필요합니다. 이런 온디바이스 AI 기술은 자율주행 로봇과 자율주행 자동차에 필수입니다. 첫째, 자율주행 기계들은 실제 환경에서 일어나는 일들에 실시간으로 기민하게 반응해야 하기 때문입니다. 연산할 때마다 서버와의 무선 통신이 필요하다면 시간 지연이 생길 겁니다. 둘째, 주행 중 인터넷 연결이나 기타 무선 통신 연결이 시 단절되는 일이 종종 발생할 텐데, 그런 상황에서도 자율주행 기계는 멈추지 않고 작동해야 하기 때문입니다. 셋째, 민감한 개인 정보를 보호하는 데도 온디바이스 AI가 유리하기 때문입니다.

### 에지 파이프라인의 목적  
에지 디바이스에는 비용, 전력, 무게, 부피 등에 대한 제약이 있기 때문에, 시스템 자원과 성능이 한정된 하드웨어를 사용합니다. 사용하는 하드웨어를 결정하고 전체 시스템을 설계하고 난 뒤, 하드웨어를 확장하거나 변경

### 에지 파이프라인의 필요성  
로봇, 자동차, 드론 등에 쓰이는 자율주행 기술은 일상에서 흔히 접할 수 있게 되었습니다. 이런 장치들은 우리가 사용하는 컴퓨터와 유사한 기능을 하는 하드웨어가 탑재되어 있습니다. 이 하드웨어는 일반적으로 ARM 아키텍처를 기반으로 하는 CPU를 사용합니다. ARM 기반 CPU는 낮은 전력 소모 특이로 발열도 적고 베터리도 적게 사용한다는 장점이 있기 때문입니다. 또한 하드웨어에는 보통 예약 GPU를 포함하고 있어, 그래서 디스플레이 처리뿐만 아니라 딥러닝 모델의 추론, 이미지 및 비디오 처리 등 고성능의 병렬 연산을 요구하는 작업을 수행할 수 있습니다.

우아한형제들 자율주행 로봇 ‘딜리’에도 CPU, GPU뿐만 아니라 딥러닝 가속기*, 영상 이미지 합성기**, HW 인코더/디코더*** 등과 함께 여러 센서 인터페이스가 패키징된 엔비디아 젯슨 플랫폼을 기반으로 하다는 임베디드 보드가 내장되어 있습니다. 임베디드 보드의 GPU, LiDAR****, 라**, 카메라 등 다양한 센서들을 로봇에 연결해, 자율주행과 관련된 복잡한 연산을 로봇에서 효율적으로 수행합니다. 이와 같은 임베디드 보드가 우리의 에지 디바이스가 됩니다.

엔비디아 TensorRT는 C++과 파이썬 API를 제공합니다. 엔진으로 변환된 코드를 API를 사용해 직접 구현할 수 있어 AI를 사용하는 여러 오픈소스 도구가 존재합니다. 이런 이미 공개된 도구들을 이용하면, 에지 디바이스에서 동작하는 추론 모델을 비교적 손쉽게 실행할 수 있습니다.

## 엔비디아 도구들 소개  
엔비디아에서 제공하는 개발 도구인 타일러그래픽(trtexec), 트텍스트TtREX, 엑스라이트 시스템tBright Systems를 간단히 살펴보겠습니다.

### 타일러그래픽
타일러그래픽은 TensorRT를 가장 쉽고 편하게 사용할 수 있는 CLI(command line interface)입니다. 이 도구를 사용하면, 별도의 애플리케이션을 개발하지 않고도 모델의 변환과 추론 성능을 손쉽게 평가할 수 있습니다. 핵심 기능으로는 ‘디바이스에 최적화된 플랜 파일 생성’, ‘생성된 플랜의 추론 성능을 간단히 벤치마크’가 있습니다.

‘디바이스에 최적화된 플랜 파일 생성’은 트레이닝된 모델을 에지 디바이스에서 최적화한 엔진 파일로 생성한다는 의미입니다. 이 엔진을 플랜(PLAN)이라고 부릅니다. 이렇게 생성된 엔진 파일은 추론 작업을 수행하거나 또는 애플리케이션에 통합해 사용할 수 있습니다.

‘생성된 플랜의 추론 성능을 간단히 벤치마크’라는 빌드는 생성된 플랜, 또는 입력값으로 지정된 플랜을 사용해 네트워크의 추론 성능을 테스트하는 기능입니다. 사용자에게 여러 추론 관련 옵션을 제공해 입력과 출력, 성능 측정을 위한 반복 횟수, 정밀도 등을 설정할 수 있습니다. 더불어 네트워크가 실제 환경에서 어떻게 동작할지 미리 확인할 수 있습니다.

이처럼 타일러그래픽을 이용하면 복잡한 개발 과정을 거치지 않고도 신속하게 추론 성능을 테스트하고 최적화할 수 있습니다. TensorRT를 처음 접하거나 변환된 단일 모델의 성능을 빠르게 확인할 때 매우 유용합니다. 타일러그래픽에 지원하는 다양한 옵션*을 사용하는 것만으로도, C++이나 파이썬을 통해 구현할 수 있는 대부분의 모델 변환과 추론을 수행할 수 있습니다.

```
trtexec <model option> <build option> <inference option> <reporting option> <system option>
```

#### 모델 옵션  
--uff, --onnx, --model, --mdkl, --deploy를 사용해 변환 전 입력 모델 파일의 경로를 전달합니다.

https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec-flags

#### 빌드 옵션  
TensorRT C++ 빌더를 사용해 플랜을 생성할 때 사용자가 설정값을 전달할 수 있습니다. 설정값들을 전달하지 않으면 기본값을 사용하거나 모델을 탐색해 입력, 출력, 레이어와 포맷 등을 자동으로 추출해 사용합니다.

이때 --fp16, --int8 등의 옵션을 사용해 레이어의 정밀도를 낮춤으로써, 모델 추론이 소요되는 시간을 줄일 수 있습니다. --layerPrecisions를 사용하면 특정 레이어의 정밀도를 지정할 수도 있습니다. 예를 들어 특정 레이어에 대해서만 --fp16 또는 --int8 옵션을 사용해 정밀도를 지정할 수 있습니다. 이런 방법으로 모델의 일부 레이어에서는 높은 정밀도를 유지하면서도 다른 레이어는 낮은 정밀도로 처리해 성능을 최적화할 수 있습니다.

--maxBatch, --minShapes, --optShapes, --maxShapes와 같은 옵션들을 사용하면 모델의 입력 형태 및 배치 크기를 지정할 수 있습니다. 이런 옵션들은 input shape가 바뀔 수 있는 때에도 같은 모델을 테스트하거나 사용할 수 있는 유연성을 제공합니다.

--saveEngine을 사용하면 변환된 플랜을 파일로 저장할 수 있으며, --loadEngine 옵션으로는 빌드 과정에서 소요되는 시간을 줄이고 추론 동작을 즉시 테스트해볼 수 있습니다. 한편 --buildOnly 옵션을 사용해 추론 단계를 건너뛰고 빌드만 수행할 수도 있습니다.

옵션 --profilingVerbosity는 엔진을 빌드하는 과정에서 엔진 자체, 각 레이어, 그리고 바이닝에 대한 정보를 얼마나 자세히 표시할지를 정합니다. 이 옵션을 detaill로 설정해 verbosity 단계를 높이면, 리포팅 과정에서 결과물이 더 상세하게 보이게 되어, 플랜의 구조와 특징을 더 자세히 파악할 수 있습니다.

#### 추론 옵션  
--iterations, --duration, --warmUp 등으로 추론 수행 방식과 반복 횟수를 조정할 수 있습니다. 예를 들어 --iterations=100 옵션을 사용해 100번의 추론을 반복 수행하거나, --warmUp=200 옵션을 사용해 200ms 동안에 모델을 워밍업할 수 있습니다. 이는 모델의 성능을 초기화가 진행되는 워밍업 이후에 얼마나 안정적으로 측정할 수 있는지 알 수 있습니다.

#### 리포팅 옵션  
추론 결과와 성능 데이터를 기록하고 분석할 때 사용하는 옵션들도 있습니다. --exportTimes 옵션을 사용하면 추론 시간 데이터들을 JSON 파일로 내보낼 수 있습니다. --exportOutput, --exportProfile 옵션으로 추론 및 프로파일링(profiling) 결과를 JSON 파일로 저장할 수 있습니다. 또한 --dumpOutput, --dumpProfile, --dumpLayerInfo 등의 옵션을 사용해 추론 결과와 각 레이어의 프로파일링 정보를 출력할 수 있습니다.

#### 시스템 옵션  
--device=N 옵션을 사용해 특정 GPU 디바이스를 선택하거나, --useDLACore=N 옵션으로 DLA(Deep Learning Accelerator) 코어를 활용할 수 있습니다. 또한 --allowGPUFallback 옵션을 사용하면 DLA가 지원하지

없는 레이어를 GPU에서 처리하도록 설정할 수 있습니다.

#### 예제  
다음은 이런 옵션들을 활용한 타일러그래픽 명령어의 간단한 예시입니다.

```
trtexec \
  --onnx=/root/ml/onnx/object_detector/model.onnx \
  --noDataTransfers \
  --buildOnly \
  --separateProfileRun \
  --saveEngine=/root/ml/tensorrt/object_detector/model.plan \
  --exportTimes=/root/ml/tensorrt/object_detector/model.timing.json \
  --exportProfile=/root/ml/tensorrt/object_detector/model.profile.json \
  --exportLayerInfo=/root/ml/tensorrt/object_detector/model.graph.json \
  --timingCacheFile=/root/ml/tensorrt/object_detector/model.timing.cache \
  --plugins=/root/ml/plugins/libcustom_tensorrt_ops.so \
  --profilingVerbosity=detailed \
  --int8 \
  --dumpProfile
```

이처럼 타일러그래픽을 사용하면 모델을 엔비디아 플랫폼에서 TensorRT를 이용해 동작하는 플랜으로 변환할 수 있고 추론을 간편하게 수행할 수도 있습니다.
![[6.Others/Excalidraw/16장 로봇을 위한 MLOps 2부.md#^hRb4y53LylgN3Gk3GsnEF]]
### TREX(trt-engine-explorer)  
트렉스는 파이썬 패키지 모듈이며 변환된 플랜과 추론 과정에서 추출된 생성물을 분석합니다. 이 도구는 주피터 노트북과 함께 사용할 수 있습니다. 모델의 변환과 추론 과정에서 생성된 결과물을 입력값으로 해 초기 성능과 플랜의 계층 구조를 시각화할 때에 유용합니다. 이 도구는 “TensorRT git 리포지터리”에 포함되어 있습니다.

빠르게 트렉스의 동작 환경을 구성하고 트렉스를 설치하는 방법은 다음과 같습니다.

```
git clone https://github.com/NVIDIA/TensorRT.git
cd TensorRT/tools/experimental/trt-engine-explorer
python3 -m pip install virtualenv
python3 -m virtualenv env_trex
source env_trex/bin/activate
python3 -m pip install -e .
```

앞서 말씀드린 것처럼 트렉스는 파이썬 기반의 여러 스크립트툴로 구성되는데, 그중 TensorRT를 통해 변환된 플랜을 그래프로 시각화하는 스크립트를 유용하게 사용할 수 있습니다. 보통 .tflite, .caffemodel, .pth, .onnx 포맷의 모델들은 네트론*과 같은 도구를 이용해서 입력, 출력, 내부 레이어를 시각화할 수 있습니다.

https://bit.ly/4eg4BDD  
Netron : 주요 딥러닝 모델의 시각화를 위한 도구 이름으로, 방금 그렸던 방안의 네트론이 적절한 툴입니다. https://netron.app

이들의 구조를 파악할 수 있지만, TensorRT에 의해 변환된 플랜에 대해서는 그러한 도구들이 시각화를 지원하지 않습니다.

따라서 아래 스크립트*를 이용하면, 엔진을 간단히 시각화해볼 수 있습니다.

```python
import argparse
import shutil
import sys
import trex

def draw_engine(engine_json_fname: str):
    plan = trex.EnginePlan(engine_json_fname)
    formatter = trex.layer_type_formatter
    display_regions = True
    expand_layer_details = False
    graph = trex.to_dot(plan,
                        formatter,
                        display_regions=display_regions,
                        expand_layer_details=expand_layer_details)
    trex.render_dot(graph, engine_json_fname, "png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw engine graph from JSON file")
    parser.add_argument("--graph", type=str, help="Path to the graph file")
    args = parser.parse_args()
    if args.graph is None:
        parser.print_help()
        sys.exit(1)
    draw_engine(args.graph)
    sys.exit(0)
```

* 참고: 엔비디아 기술 블로그에서 ‘Exploring NVIDIA TensorRT Engines with TREX’

### 엔비디아 Nsight Systems  
엔비디아 엔사이트 시스템은 엔비디아에서 제공하는 강력한 성능 분석 도구입니다. CPU와 GPU뿐만 아니라 다양한 가속기를 분석해 병목 bottleneck을 파악하고 최적화하는 데 쓰입니다. 특히 멀티 코어 CPU와 멀티 GPU, DLA 등에서 런타임 스케줄의 병렬 연산이 동작하게 되는 복잡한 애플리케이션을 이해하고 최적화할 때에 유용한 도구입니다.

엔사이트 시스템은 GUI와 CLI의 두 가지 방식의 유저 인터페이스 방식을 제공합니다. GUI에서도 프로파일링을 시작하는 것이 가능하지만 GUI를 동작시킬 때 약간의 오버헤드가 있기 때문에 프로파일링을 nsys-CLI에서 수행하고 나서 생성된 결과물을 nsys-ui에서 읽어서 시각화 분석을 하는 방식이 간편합니다.

#### 설치  
엔사이트 시스템은 다양한 OS 기반의 호스트 및 타깃 디바이스에서 프로파일링을 실행하거나 결과를 분석할 수 있습니다. 가장 간편하게 사용하는 방법은 엔비디아 젯슨 플랫폼의 테그라 시스템에서 엔비디아 젯팩jetpack의 일부로 제공되는 엔사이트 시스템 임베디드 플랫폼 메타서 Embedded Platforms Edition을 실행하는 겁니다. 여기에는 타깃 디바이스에 사용할 수 있는 검증된 버전이 설치되어 있으므로 로컬 머신에서 이를 바로 실행할 수 있습니다.

만약 젯팩 시스템의 이미지를 최초로 설치할 때에 엔사이트 시스템 패키지 설치가 누락되었다면, 젯슨 젯팩 리포지터리**에서 타깃에 적합한 버전의 엔사이트를 설치할 수 있습니다. 우분투 리눅스 시스템에서는 엔사이트를 다음과 같은 방법으로 설치할 수 있습니다.

```
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/${dkpg --print-architecture}/ /"
sudo apt install nsight-systems
```

* Tegra : 엔비디아의 모바일 및 임베디드 장치용 SoC(System on Chip) 브랜드  
  https://repo.download.nvidia.com/jetson

#### 엔사이트 시스템 CLI  
엔사이트 시스템 CLI는 다양한 옵션을 제공합니다. 자주 사용하는 전체 프로파일링 명령어 옵션은 profile이며 이름을 통해 CPU와 GPU의 실행 타임, 메모리 사용량, API 호출 등의 정보를 수집해 성능상의 병목을 파악할 수 있습니다. 다음은 엔사이트 시스템 CLI /root에 사용하는 프로파일링을 수행하는 스크립트입니다. 이 스크립트는 run_inference.sh라는 실행 파일을 프로파일링하고, 결과를 /root/profile.nsys-rep라는 파일로 저장합니다.

```
sudo nsys profile \
  --trace=cuda,osrt,ntv,cudnn,cudla,tegra-accelerators \
  --cuda-memory-usage=true \
  --output=/root/profile \
  --show-output=true \
  --stop-on-exit=true \
  --stats=false \
  --force-overwrite=true \
  --cpuctxsw=system-wide \
  --gpuctxsw=true \
  ./run_inference.sh
```

프로파일링할 때에는 꼭 관리자 권한으로 실행해야 합니다. 그래야 시스템 전체 자원에 접근하며 프로파일링할 수 있습니다.

https://docs.nvidia.com/nsight-systems/UserGuide

- 프로파일링 옵션 -

| 옵션                                                    | 설명                                                                                                   |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| --trace=cuda,osrt,ntvx,cudnn,cudla,tegra-accelerators | CUDA, OS 인터럽트, NTVX, cuDNN, cuDLA, 테그라 가속기 등의 트레이스를 절차합니다. 다양한 라이브러리와 HW 가속기가 어떻게 쓰이는지 분석하는 데 사용합니다. |
| --cuda-memory-usage=true                              | CUDA 메모리 사용량을 절차합니다. GPU 메모리 사용을 분석하는 데 유용합니다.                                                       |
| --output=/root/profile                                | 프로파일링 결과를 /root/ 폴더에 profile 파일명으로 저장합니다. 이 파일은 나중에 분석에 쓰입니다.                                        |
| --show-output=true                                    | 프로파일링 실행 중 출력 내용을 화면에 표시합니다.                                                                         |
| --stop-on-exit=true                                   | 프로파일링앱의 프로세스가 exit되거나 지정된 시간이 지나면, 데이터 수집을 자동으로 종료합니다.                                               |
| --stats=false                                         | 기본 통계를 출력하지 않습니다(필요 시 true로 변경 가능).                                                                  |
| --force-overwrite                                     | 기존 프로파일링 결과 파일을 덮어씁니다.                                                                               |
| --cpuctxsw=system-wide                                | 시스템 전체의 CPU 컨텍스트 스위칭을 절차합니다.                                                                         |
| --gpuctxsw                                            | GPU 컨텍스트 스위칭을 절차합니다.                                                                                 |

이 스크립트를 실행하면 /root/profile.nsys-rep 파일이 생성되고 이름 엔사이트 시스템 GUI에서 분석할 수 있습니다.

#### 엔사이트 시스템 GUI  
GUI 도구에서는 샘플링 주기에 맞추어 측정된 시스템 자원의 타임라인 뷰Timeline View가 제공되며, 성능을 추적하는 가속기나 라이브러리의 API 호출 스택 및 통계를 분석할 수 있습니다. 추적하는 자원이 많을수록 프로파일링에 따른 시스템 부하가 있으므로, 프로파일링할 때 절차하는 자원 수를 최소로 하고, 필요하다면 일부 자원을 별도로 프로파일링하는 것이 좋습니다.

### Trt-Infersight 개발  
타일러그래픽은 훌륭한 도구이지만, 다중 모델 추론을 동시에 하는 시스템의 성능을 검증할 때 몇 가지 한계가 있습니다. 타일러그래픽의 성능 벤치마크는 변환된 단일 모델의 성능 프로파일링에 초점을 맞추고 있으며, 실제 센서 입력 주기에 따른 지원 사나 설정, 또는 다수 모델의 동기 또는 비동기 프로파일 방식을 고려하는 옵션을 제공하지 않습니다.

일반적으로 자율주행 분야에서는 카메라, 라이다, 레이더 등 여러 센서를 동시에 사용해 주변 환경을 인지하는데, 매 딥러닝을 사용합니다. 따라서 다양한 종류의 센서 데이터를 활용하기 위해 다수의 딥러닝 모델이 로봇에 탑재될 수 있습니다. 각 센서는 주기적으로 주변 환경을 스캔하고 에지 디바이스 컴퓨터에 자료를 전달하면, 로봇은 센서 데이터가 전달될 때마다 적절한 딥러닝 모델의 추론을 수행해야 합니다. 결국, 여러 머신러닝 모델의 추론을 동시에 수행해야 합니다. 에지 디바이스에서 이 모델들 이 센서 주기들에 맞추어 모두 잘 동작하는지 점검할 필요가 있습니다.

또한 프로파일링을 커스터마이징할 수 있고, 기능을 유연하게 확장할 수 있고, 실제 자율주행과 관련된 로직들도 검증할 수 있는 도구가 필요했습니다.

이런 요구사항을 충족시키는 도구로 엔비디아 TensorRT C++ API*를 이용해 자체 개발하기로 했습니다. 이렇게 해서 탄생한 것이 트리트-인퍼사이트Trt-infersight입니다. 자체 제작한 툴킷 트리트-인퍼사이트의 주요 기능으로는 모델 초기화, 모델 역직렬화 및 탐색, 성능 프로파일러 생성, 추론 실행 및 스텝 관리, 출력 생성이 있습니다. 해당 도구가 오픈 소스는 아니지만, 도구가 제공하는 기능들을 압축되 비슷한 업무를 처리할 때 무조건 필요합h니다.

* https://docs.nvidia.com/deeplearning/tensorrt/api/c_api

#### 모델 초기화  
도구가 시작될 때 입력 파일에 따라 초기화를 수행합니다. 전체 프로세스의 시작으로 사용자가 설정한 값에 따라 동작시켜야 하는 모든 모델의 인스턴스를 생성하고 초기화합니다.

#### 모델 역직렬화 및 탐색  
각 모델을 순환하며 TensorRT로 변환된 엔진을 역직렬화합니다. 그리고 역직렬화된 컨텐츠를 탐색해, 모델이 요구하는 입력 및 출력을 포함한 모델의 특성을 추출합니다. 입력 생성기Input generator는 탐색 후에 추출된 입력에 따라 텐서input shape과 포맷을 활용해, 입력 텐서Input tensor와 출력 텐서output tensor를 생성합니다. 또한 적절한 범위를 가지는 임의의 값은 입력 텐서Input tensor에 저장합니다. 사용자가 입력 파일을 제공한 경우, 파일을 통해 입력 텐서의 값을 생성합니다.

#### 성능 프로파일러 생성  
트리트-인퍼사이트에서 구현된 프로파일러는 설정 파일에 포함된 평가

옵션에 따라 생성됩니다. 프로파일러는 성능 프로파일링에 필요한 기능을 수행하고, 추론 시간을 측정하거나 시스템 성능을 수집하는 등의 작업을 수행한 후 그 결과를 rpt(인사이트)에 저장합니다.

#### 추론 실행 및 스레드 관리  
트리트-인퍼사이트에서 구현된 실행기executor는 각 모델에 설정된 가속기의 타입(GPU, DLA)에 따라 추론을 하는 스레드를 다르게 생성합니다. 그러고 추론을 수행해 모든 이터레이션iteration이 완료될 때까지 대기합니다. 
모델을 추론할 때는 비교적 간단합니다. 데이터의 입력 주기에 따라 추론을 하는 경우, 현재 이터레이션에서 대기하다가 다음 입력 데이터가 들어올 때 다음 이터레이션을 시작합니다.

모델이 2개 이상으로 늘어나면, 모델 동작이 동기적인지 비동기적인지, 데이터의 입력 주기는 어떠한지, 이터레이션마다 대기해야 하는지 등의 설정에 따라 다양한 방식으로 CUDA 스트림stream들이 동작하게 됩니다.

다음 그림은 ‘멀티 인퍼런스Multi inference, 동기’ 설정에서 실행을 보여줍니다. CUDA 스트림이 생성되며, CUDA 스트림 두 개에서 동시에 추론을 수행합니다. 여러 추론 작업이 단일 GPU에서 병렬로 실행될 경우, GPU의 시간당 처리량이 제한되어 있으므로 각 추론 시간이 길어질 수 있습니다. 따라서 GPU, DLA가 다수 존재한다면 각 모델을 다른 가속기에 배치해 시간당 처리량을 높이는 방법이 효율적입니다.

다음 그림은 ‘멀티 인퍼런스, 비동기’ 설정에서 실행을 보여줍니다. CUDA 스트림 두 개로 동작하지만, 각 이터레이션마다 모든 추론이 완료될 때까지 대기한 후 다음 이터레이션으로 넘어갑니다.

멀티 인퍼런스, 비동기, 이터레이션 대기

다음 그림은 ‘멀티 인퍼런스, 비동기, 입력 데이터 주기에 따른 대기, 이터레이션 대기’ 설정에서 실행을 보여줍니다. 각 모델별로 센서의 입력 주기를 설정하고, 매 이터레이션마다 모든 모델의 센서 입력 주기가 끝날 때까지 대기합니다.

멀티 인퍼런스, 비동기, 입력 데이터 주기에 따른 대기, 이터레이션 대기

| 모델 0 입력 간격 | 모델 0 |
| 모델 1 입력 간격 | 모델 1 |

성능 측정 결과로 다음과 같은 파일을 생성합니다.

- profile_report.yaml : 여러 모델의 추론이 동시에 실행될 때, 각 모델에 대해 측정된 시간, 시스템 지표(GPU 점유, CPU 점유, 온도 등)의 결과를 종합해 YAML 형식으로 저장합니다.
- system_profile.txt : tegrasstats를 사용해 일정 주기마다 취득한 시스템 성능 데이터를 텍스트 형식으로 저장합니다.
- system_profile_graph.png : system_profile.txt를 기반으로, 시간에 따른 시스템 성능 변화 추이를 그래프로 저장합니다.

### 에지 파이프라인의 구성  
지금까지 소개한 타일러그래픽, 트렉스, 엔사이트 시스템, 트리트-인퍼사이트를 활용해 다음과 같은 작업을 수행하는 파이프라인을 구성할 수 있습니다.

1. 학습된 머신러닝 모델이 에지 디바이스에서 추론 가능한 플랫폼으로 변환됐는지를 확인합니다. 지원되지 않는 레이어가 있을 때는 사용하는 해당 레이어를 제거하거나 커스텀 연산 라이브러리를 만들어야 합니다(타일러그래픽).
2. 여러 모델이 동작할 때 시스템 성능 요구조건을 충족하는지를 판단할 수 있는 지표들을 추출합니다(트리트-인퍼사이트).
3. 프로파일링 후에는, 변환된 엔진을 분석하는 데에 쓸 수 있는 자료를 제공합니다(타렉스, 엔사이트 시스템).

#### 에지 파이프라인

- 입력 확인
- 파일/모델 변환 실행 및 성능 측정
- 측정을 저장할 위한 디렉터리 생성
- 트리트-인퍼사이트, 엔사이트 시스템
- 모델 최적화 및 탐색
- 모델 추론 및 성능 평가
- TensorRT 분석
- 모델별 추론 결과 시각화

이처럼, 주어진 역할을 수행하는 데 필요한 모든 태스크task들을 정의하고, 각 태스크를 수행할 수 있는 도구들을 적절히 사용해, 에어플로 DAG 형태로 구성한 에지 파이프라인을 살펴보겠습니다.

### 에지 파이프라인 DAG *

1단계 KubernetesPodOperator를 사용해 각 단계에 필요한 태스크들을 정의하고 실행하는 첫 단계로, 에어플로 DAG의 기본 설정값 및 변수를 선언합니다.

```python
from datetime import datetime, timedelta
import json

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
import KubernetesPodOperator
from utils.common import read_json
from kubernetes.client import models as k8s

CONFIG_FILE_NAME = "model_config.json"
EDGE_NODE_NAME = "edge-node"
OUTPUT_DIR = "/root/dags/edge/output"
COPY_ONNX_FROM = "/root/dags/edge/onnx"

default_args = {
    "owner": "airflow_user",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
}

dag = DAG(
    "edge_pipeline",
    default_args=default_args,
    description="Edge pipeline DAG running on Kubernetes",
    schedule_interval=timedelta(days=1),
)

config = read_json("/opt/airflow/dags/config/" + CONFIG_FILE_NAME)
config_json = json.dumps(config)
```

2단계 에어플로 DAG에서 쓰인 루버네티스 볼륨과 볼륨 마운트를 설정합니다.

```python
volume = k8s.V1Volume(
    name="airflow-dags-pvc",
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
        claim_name="airflow-dags-pvc"
    ),
)
volume_mount = k8s.V1VolumeMount(
```python
    name="airflow-dags-pvc",
    mount_path="/root/dags",
    sub_path=None,
    read_only=False,
)

nsight_volume = k8s.V1Volume(
    name="nsight-volume",
    host_path={"path": "/opt/nvidia/nsight-systems"},
)
nsight_volume_mount = k8s.V1VolumeMount(
    name="nsight-volume",
    mount_path="/root/nsight-systems",
    read_only=False,
)

tegra_volume = k8s.V1Volume(
    name="tegra-volume",
    host_path={"path": "/usr/bin"},
)
tegra_volume_mount = k8s.V1VolumeMount(
    name="tegra-volume",
    mount_path="/root/bin",
    read_only=False,
)
```

3단계 작업이 에지 디바이스에서 실행되도록 노드 어피니티를 설정합니다.

```python
edge_affinity = {
    "nodeAffinity": {
        "requiredDuringSchedulingIgnoredDuringExecution": {
            "nodeSelectorTerms": [
                {
                    "matchExpressions": [
                        {
                            "key": "kubernetes.io/hostname",
                            "operator": "In",
                            "values": [EDGE_NODE_NAME],
                        }
                    ]
                }
            ]
        }
    }
}
```

4단계 각 태스크를 KubernetesPodOperator를 사용해 정의합니다.

```python
make_output_dir_task = KubernetesPodOperator(
    namespace="airflow",
    image="myregistry.com/myimage/ml_pipeline:0.0.0",
    cmds=["python3"],
    arguments=[
        "env_setup/make_output_dir.py",
        "--output_dir", OUTPUT_DIR,
    ],
)
```

```python
    "--ts", "{{ ts_nodash }}",
    "--use_env_vars"
],
env_vars=[{"name": "CONFIG_JSON", "value": config_json}],
name="make_output_dir_task",
task_id="make_output_dir",
volumes=[volume],
volume_mounts=[volume_mount],
get_logs=True,
dag=dag,
affinity=edge_affinity,
is_delete_operator_pod=True,
image_pull_policy="Never",
startup_timeout_seconds=300,
)

make_onnx_dir = KubernetesPodOperator(
    namespace="airflow",
    image="myregistry.com/myimage/ml_pipeline:0.0.0",
    cmds=["python3"],
    arguments=[
        "env_setup/make_onnx_dir.py",
        "--output_dir", OUTPUT_DIR,
        "--ts", "{{ ts_nodash }}",
        "--copy_onnx_from", COPY_ONNX_FROM,
    ],
    name="make_onnx_dir",
    task_id="make_onnx_dir",
    volumes=[volume],
    volume_mounts=[volume_mount],
    get_logs=True,
    dag=dag,
    affinity=edge_affinity,
    is_delete_operator_pod=True,
)

build_trt_engine_task = KubernetesPodOperator(
    namespace="airflow",
    image="myregistry.com/myimage/ml_pipeline:0.0.0",
    cmds=["python3"],
    arguments=[
        "tensorrt_builder/build_trt_engine.py",
        "--output_dir", OUTPUT_DIR,
        "--ts", "{{ ts_nodash }}",
    ],
    name="build_trt_engine_task",
    task_id="build_trt_engine",
    volumes=[volume],
    volume_mounts=[volume_mount],
    get_logs=True,
    dag=dag,
    affinity=edge_affinity,
    is_delete_operator_pod=True,
)

run_trt_infersight_task = KubernetesPodOperator(
    namespace="airflow",
    image="myregistry.com/myimage/ml_pipeline:0.0.0",
    cmds=["python3"],
    arguments=[
        "tensorrt_profiler/run_trt_infersight.py",
        "--output_dir", OUTPUT_DIR,
        "--ts", "{{ ts_nodash }}",
    ],
)
```

```python
name="run_trt_infersight_task",
task_id="run_trt_infersight",
volumes=[volume, nsight_volume, tegra_volume],
volume_mounts=[volume_mount, nsight_volume_mount, tegra_volume_mount],
get_logs=True,
dag=dag,
affinity=edge_affinity,
is_delete_operator_pod=True,
)

run_trex_task = KubernetesPodOperator(
    namespace="airflow",
    image="myregistry.com/myimage/ml_pipeline:0.0.0",
    cmds=["python3"],
    arguments=[
        "tensorrt_profiler/run_trex.py",
        "--output_dir", OUTPUT_DIR,
        "--ts", "{{ ts_nodash }}",
    ],
    name="run_trex_task",
    task_id="run_trex",
    volumes=[volume],
    volume_mounts=[volume_mount],
    get_logs=True,
    dag=dag,
    affinity=edge_affinity,
    is_delete_operator_pod=True,
)
```

5단계 마지막으로 정의한 태스크들이 실행될 순서를 설정합니다.

```python
make_output_dir_task >> make_onnx_dir >> build_trt_engine_task >> run_trt_infersight_task >> run_trex_task
```

에어플로 DAG를 구성해, 에지 디바이스에서 다중 모델의 추론과 프로파일링을 수행하는 자동화된 파이프라인을 만드는 방법을 알아보았습니다. 각 작업은 KubernetesPodOperator를 사용해 정의되며, 쿠버네티스 클러스터에서 실행됩니다.

## 마치며  
GPU 서버에서 모델을 학습시키는 학습 파이프라인과 에지 파이프라인을 연결하면 모델의 학습, 배포, 검증 등이 모두 자동으로 수행되는 MLOps 시스템을 구성할 수 있습니다. 이 글에서는 자율주행 로봇을 위한 머신러닝 모델 개발을 자동화하고 개발된 모델들을 검증하는 방법을 소개했습니다. 앞으로도 다양한 머신러닝 모델들이 ‘딜리’ 안에서 잘 동작하도록 기술 개발에 힘쓰겠습니다.
