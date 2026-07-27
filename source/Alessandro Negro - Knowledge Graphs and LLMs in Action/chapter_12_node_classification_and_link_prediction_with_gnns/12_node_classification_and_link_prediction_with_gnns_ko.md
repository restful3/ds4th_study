---
lang: ko
format:
  html:
    toc: true
    embed-resources: true
    theme: cosmo
---

# GNN을 이용한 노드 분류와 링크 예측

### 이 장에서 다루는 내용


실제 시나리오에서 그래프 신경망 (graph neural networks) 사용하기

노드 분류 시스템 구축하기

링크 예측 시스템 구축하기

이 장에서는 노드 분류와 링크 예측에 그래프 신경망(GNN)을 사용하는 방법을 살펴봅니다. 이러한 과제는 그래프 기반 기계학습(ML)의 근본적인 도전 과제를 나타내며, 많은 실제 응용의 핵심입니다.

먼저 자금세탁방지 (anti-money laundering, AML) 응용에 초점을 맞추어 노드 분류를 위한 GNN의 적용을 논의합니다. 금융 거래를 그래프로 표현함으로써, GNN은 의심스러운 패턴을 식별하고, 노드를 합법 또는 불법으로 분류하며, 금융 사기 방지에 도움을 주는 데 사용될 수 있습니다. 그런 다음 추천 시스템에서의 링크 예측에 초점을 맞춥니다. 우리는 평점을 기반으로 잠재적인 사용자–영화 상호작용을 예측하기 위해 GNN 기반 접근법을 사용할 것입니다. 사용자와 영화에 대한 임베딩을 학습하고 그들 사이의 링크를 사용함으로써, 사용자의 선호도에 따라 영화를 추천할 수 있습니다. 과제와 응용 도메인이 서로 다름에도 불구하고, 두 시나리오는 그림 12.1에 제시된 동일한 엔드투엔드 프레임워크로 다룰 수 있습니다.

![](images/ko/figure-12-1-ko.png)  
그림 12.1 여러 과제와 응용 도메인에 채택될 수 있는 GNN 기반 시스템을 개발하기 위한 엔드투엔드 프레임워크. 입력 데이터는 일반적으로 서로 다른 객체와 그들 사이의 상호작용을 설명하는 반정형 (semistructured) 소스들의 모음입니다. 목표는 데이터를 처리하여 인코더–디코더 아키텍처의 입력이 되는 그래프를 생성하는 것입니다. 우리의 시나리오에서 인코더는 GNN 모델을 사용하고, 디코더는 특정 다운스트림 과제 (downstream task)에 맞게 모델을 최적화할 수 있게 해 주는 함수를 기반으로 합니다. 이 프레임워크의 출력은 추론 단계에 채택될 수 있는 학습된 모델입니다.

이 장 전반에 걸쳐 우리는 서로 다른 GNN 아키텍처의 성능을 비교합니다: 그래프 합성곱 네트워크 (Graph convolutional networks, GCNs) [1], GraphSAGE (SAGE) [2], 그래프 어텐션 네트워크 (graph attention networks, GATs) [3]. 이러한 모델은 PyTorch Geometric (PyG) 라이브러리를 사용하여 개발되며, 정밀도 (precision), 재현율 (recall), F1 점수와 같은 지표를 사용하여 평가됩니다. 혼동 행렬 (confusion matrices)에서 얻은 통찰과 지표 추세의 시각화는 실제 시나리오에서 이러한 접근법의 절충점과 역량을 더욱 잘 보여 줍니다. 이 장을 마치면 노드 분류와 링크 예측 과제를 해결하기 위해 GNN을 적용하는 방법을 포괄적으로 이해하게 되며, 다양한 응용을 위해 그래프 기반 ML을 사용할 수 있는 도구를 갖추게 될 것입니다.

### 12.1 자금세탁방지 애플리케이션을 위한 노드 분류


노드 분류는 그래프 기반 ML에서 핵심적인 과제이며, 금융 범죄 대응을 포함한 다양한 응용 도메인에 적합합니다. 자금세탁방지 (anti-money laundering, AML)의 맥락에서 금융 거래는 그래프로 모델링될 수 있으며, 여기서 노드는 계좌 또는 엔터티를 나타내고 엣지는 거래 관계를 나타냅니다.

이 절에서는 Elliptic 데이터셋이 제공하는 금융 거래 네트워크에서 합법 노드와 불법 노드를 탐지하기 위해 GNN을 사용하는 데 초점을 맞춥니다. 그림 12.2는 노드 분류에 GNN을 적용하기 위한 종단 간 모델을 보여 주며, 각 블록은 우리 과제의 요구사항을 충족하기 위한 특정 구현으로 특징지어집니다.

![](images/ko/figure-12-2-ko.png)  
그림 12.2 AML 맥락에서 노드 분류에 맞추어진 종단 간 프레임워크. 거래 데이터와 노드 레이블은 동질 그래프 구조로 변환됩니다. 동질 GNN인 인코더는 국소 그래프 구조를 포착하여 노드 임베딩을 학습합니다. 디코더는 로그 소프트맥스 함수와 교차 엔트로피 손실을 사용하여 노드를 합법 또는 불법으로 분류하고, 의심스러운 활동을 탐지하기 위한 학습된 모델을 생성합니다.

#### 12.1.1 입력 데이터


우리는 금융 거래 네트워크에서 합법 및 불법 노드를 탐지하는 데 GNN이 지닌 잠재력을 탐구하기 위해 엘립틱 데이터셋 (Elliptic dataset)을 사용할 것입니다. 엘립틱 데이터셋은 200,000건이 넘는 Bitcoin 거래(노드), 234,000개의 방향성 결제 흐름(엣지), 그리고 익명화된 데이터에서 도출된 166개의 노드 특성으로 구성된 시계열 그래프입니다.

이 데이터셋은 세 개의 파일로 제공됩니다. 2025년 1월 기준으로, 이 파일들은 다음 정보를 제공합니다.

elliptic\_txs\_features.csv—203,769개 행을 가진 167열 파일입니다. 첫 번째 열에는 모든 노드 ID가 나열되어 있으며, 나머지 열은 각 노드와 관련된 익명화된 특성을 나타냅니다. 이 파일의 데이터는 features 변수에 저장됩니다.

elliptic\_txs\_edgelist.csv —네트워크의 엣지 수에 해당하는 234,355개 행을 포함하는 2열 파일입니다. 첫 번째 열에는 거래 출처를 나타내는 노드 ID가 나열되어 있으며, 두 번째 열에는 거래 대상을 나타내는 노드 ID가 나열되어 있습니다. 이 파일의 데이터는 edges 변수에 저장됩니다.

elliptic\_txs\_classes.csv—네트워크의 노드 수에 해당하는 203,769개 행을 포함하는 2열 파일입니다. 첫 번째 열에는 모든 노드 ID가 나열되어 있으며, 두 번째 열은 각 노드와 관련된 레이블인 1, 2 또는 unknown을 지정합니다. 이 파일의 데이터는 classes 변수에 저장됩니다.

이 데이터셋에서 합법, 불법, 미상 노드를 정의하는 클래스는 표 12.1에 표시된 것처럼 분포합니다.

표 12.1 노드와 관련된 합법, 불법, 미상 클래스의 분포
<table><tr><td>클래스</td><td>레이블</td><td>개수</td><td>비율</td></tr><tr><td>미상</td><td>Unknown</td><td>157205</td><td>77.15%</td></tr><tr><td>합법</td><td>2</td><td>42019</td><td>20.62%</td></tr><tr><td>불법</td><td>1</td><td>4545</td><td>2.23%</td></tr></table>

다음 단계에서는 CSV 파일을 전처리하여 우리 데이터에 더 간결한 수치 표현을 할당합니다. 그런 다음 금융 거래 그래프가 GNN 모델에 적합하도록 구조를 생성할 것입니다.

#### 12.1.2 그래프 프로세서: 데이터 준비


전처리 단계는 원본 데이터의 간결한 수치 표현을 달성하고 GNN 학습 단계를 위해 우리의 금융 거래 그래프 구조를 준비하는 데 매우 중요합니다. 먼저 edges 변수의 정보를 처리하는 것부터 시작하겠습니다. 표 12.2는 원본 데이터의 샘플을 보여 줍니다. txId1 열은 소스 노드의 ID를 나열하고, txId2는 대상 노드의 ID를 나열합니다.

표 12.2 소스 노드(txId1 열)와 대상 노드(txId2 열)를 보여 주는 edges 변수에 저장된 데이터 샘플
<table><tr><td>txld1</td><td>txld2</td></tr><tr><td>230425980</td><td>5530458</td></tr><tr><td>232022460</td><td>232438397</td></tr><tr><td>230460314</td><td>230459870</td></tr><tr><td>230333930</td><td>230595899</td></tr><tr><td>232013274</td><td>232029206</td></tr></table>

다음 리스팅의 코드는 각 노드에 새로운 증가 ID를 할당하고, 이러한 ID를 사용하여 엣지 데이터의 갱신된 버전을 생성합니다.

리스팅 12.1 노드에 새로운 증가 ID를 할당하고 엣지 데이터 갱신하기   
tx\_id\_mapping = {tx\_id: idx for idx, tx\_id in enumerate(features['txId'])}   
edges\_with\_features = edges.assign(   
원래 ID와 새로운 증가 ID 사이의   
Id1=edges['txId1'].map(tx\_id\_mapping),   
딕셔너리를 생성하며,   
Id2=edges['txId2'].map(tx\_id\_mapping),   
노드 특징 파일에서 시작합니다   
>   
edges\_with\_features = edges\_with\_features.dropna(subset=['Id1', 'Id2'])   
edges\_with\_features = edges\_with\_features.astype({   
'Id1': 'int64', 소스와 대상이 모두   
'Id2': 'int64', 매핑에 존재하는 엣지만   
}) < 새로운 ID가 정수임을 보장합니다   
(map은 NaN이 존재한 경우   
소스 및 대상 노드를   
증가 ID에 매핑할 때 부동소수를 산출할 수 있습니다)

표 12.3은 이 과정의 출력 샘플을 보여 주며, 이는 edges\_with\_ features 변수에 저장됩니다.

표 12.3 거래에 관여하는 각 노드와 연결된 새로운 증가 ID를 포함한 엣지 데이터
<table><tr><td></td><td></td><td></td><td></td></tr><tr><td>230425980</td><td>5530458</td><td>o</td><td>2</td></tr><tr><td>232022460</td><td>232438397</td><td>2</td><td>3</td></tr><tr><td>230460314</td><td>230459870</td><td>4</td><td>5</td></tr><tr><td>230333930</td><td>230595899</td><td>6</td><td>7</td></tr><tr><td>232013274</td><td>232029206</td><td>8</td><td>9</td></tr></table>

새로운 ID를 출발점으로 하여, 그래프의 엣지를 설명하는 텐서인 edge\_index를 생성할 수 있습니다. 이 구조를 생성하는 방법은 다음과 같습니다.

리스팅 12.2 증가 노드 ID를 사용하여 edge\_index 텐서 생성하기

```python
edge_index = torch.tensor(
edges_with_features[['Id1', 'Id2']].values.T,
dtype=torch.long
)
```

edge\_index 텐서의 샘플은 다음과 같습니다.

리스팅 12.3 edge\_index 텐서 샘플   
tensor([[ 0, 2, 4, ..., 201921, 201480, 201954], [ 1, 3, 5, ..., 202042,   
201368, 201756]])

다음으로, 이전 단계에서 생성된 증분 ID에 각 행이 대응하는 새로운 텐서를 생성하여 원래의 노드 특성 데이터 구조를 처리하겠습니다. 이 단계에서는 원래 노드 ID를 제거함으로써 학습 단계에 사용할 노드 특성을 준비할 수 있습니다.

리스팅 12.4 원래 노드 ID를 제거하여 노드 특성 텐서 생성하기

```python
node_features = torch.tensor(
features.drop(columns=['txId']).values,
dtype=torch.float
)
```

출력은 크기가 [203769, 166]인 node\_features 텐서에 저장됩니다. 이 텐서의 첫 번째 차원은 노드 수에 대응하고, 두 번째 차원은 특성 수에 대응합니다.

앞서 언급했듯이, 이 행렬의 각 행 인덱스는 이전 단계에서 생성된 증분 ID에 대응합니다. 다시 말해, node\_features 텐서의 0번째 행은 원래 ID가 230425980인 노드 0과 관련된 특성 집합에 대응합니다.

마지막 단계는 데이터셋의 원래 레이블을 수치 표현으로 변환하는 것입니다. 이 결과를 얻기 위해 scikit-learn 라이브러리의 LabelEncoder 클래스를 사용합니다.

```python
Listing 12.5 Transforming original node classes into a numerical representation
from sklearn.preprocessing import LabelEncoder Creates an instance
le = LabelEncoder() < of LabelEncoder
class_labels = le.fit_transform(classes['class']) < Assigns numerical
original_labels = le.inverse_transform(class_labels) labels to each class
node_labels = torch.tensor(class_labels, dtype=torch.long) and converts the
categorical labels
Converts the numerical labels (class_labels) back Converts the class_labels list into corresponding
into their original categorical labels, based on into a PyTorch tensor numerical labels
the mapping created in fit_transform
```

출력은 크기가 203769인 새로운 node\_labels 텐서이며, 여기서 레이블 0은 합법 노드(원래 데이터셋의 “1”)에 대응하고, 레이블 1은 불법 노드(원래 데이터셋의 “2”)에 대응하며, 레이블 2는 알 수 없는 노드(원래 데이터셋의 “unknown”)에 대응합니다. 다음으로 PyG를 사용하여 GNN 모델의 입력으로 사용할 그래프 데이터를 구축하겠습니다.

#### 12.1.3 그래프 프로세서: 동질 PyG 그래프


이제 PyG 기능을 사용하여 그래프 데이터 구조를 구축할 수 있습니다. 그림 12.3은 그래프 프로세서가 적용하는 단계, 즉 준비 단계, PyG Data 객체의 구성, 그리고 학습, 검증, 테스트 데이터셋의 생성에 대한 개요를 제공합니다.

![](images/ko/figure-12-3-ko.png)  
그림 12.3 Elliptic 데이터셋에 적용되는 데이터 처리 개요. 원천 데이터는 노드 특성, 엣지 인덱스, 인코딩된 레이블을 생성하기 위해 전처리됩니다. 이러한 텐서는 PyG Data 객체에 통합되며, 여기서 노드 마스킹 접근법은 노드를 학습(80%), 검증(10%), 테스트(10%) 세트로 분할합니다.

첫 번째 단계는 전처리 단계의 결과(node\_features, edge\_index, node\_labels)를 사용하여 PyG Data 객체를 생성하는 것입니다.

```python
Listing 12.6 Creating a PyG Data object from the data preparation results
from torch_geometric.data import Data
data = Data(x=node_features,
edge_index=edge_index,
y=node_labels)
```

PyG Data 객체는 노드의 특성, 엣지에 대한 정보, 그리고 각 노드와 연관된 레이블을 포함합니다. 다음 단계는 이 객체에 마스크를 적용하는 것으로, 이는 학습 및 평가 단계에서 사용할 노드를 지정합니다.

우리의 GNN 모델은 어떤 노드가 합법이고 어떤 노드가 불법인지 학습해야 하므로, 알려진 레이블을 가진 노드만 모델에 “보이도록” 해야 합니다. 이를 위해 다음 코드를 실행할 수 있습니다.

#### 목록 12.7 알려진 노드 레이블만 검색하기 위한 마스크 필터 준비


known\_mask = (data.y == 0) | (data.y == 1) < 요소의 텐서   
unknown\_mask = data.y == 2 < 레이블이 0   
(합법) 또는 1(불법)인 노드에 해당하며, True로 설정됨   
레이블이 2(알 수 없음)인 노드에 해당하는   
요소의 텐서이며, False로 설정됨

유효한 노드의 텐서(known\_mask)는 학습, 검증, 테스트 세트의 차원을 정의하는 데 사용됩니다.

#### 목록 12.8 학습, 검증, 테스트 데이터셋의 차원


known\_mask에서 True 값을 세고 결과 스칼라 텐서를 Python 정수로 변환합니다. 인덱스의 순열을 생성합니다.   
학습 데이터셋의 크기를 계산합니다.   
(노드의 80%)   
import numpy as np   
num\_known\_nodes = known\_mask.sum().item() 검증 데이터셋의 크기를 계산합니다.   
permutations = torch.randperm(num\_known\_nodes) < (10%)   
train\_size = int(0.8 \* num\_known\_nodes) <   
val\_size = int(0.1 \* num\_known\_nodes) < 테스트 데이터셋의 크기를 계산합니다.   
test\_size = num\_known\_nodes - train\_size - val\_size < (10%)   
total = train\_size + val\_size + test\_size < 전체 크기의 합이 원래 알려진 노드 수와 일치하는지 확인합니다.

total 변수에 저장된 결과의 차원을 사용하여 관측치 수, 즉 학습, 검증, 테스트 데이터셋에 포함될 레이블이 지정된 노드 수를 결정합니다. 결과는 표 12.4에 요약되어 있습니다.

표 12.4 레이블이 지정된 노드 수를 기준으로 한 학습, 검증, 테스트 데이터셋의 차원
<table><tr><td>데이터셋</td><td>노드 수</td><td>비율</td></tr><tr><td>학습</td><td>37,251</td><td>80%</td></tr><tr><td>검증</td><td>4,656</td><td>10%</td></tr><tr><td>테스트</td><td>4,657</td><td>10%</td></tr></table>

다음 코드를 사용하여 이전 단계에서 설정한 차원을 기반으로 PyG Data 객체에 학습, 검증, 테스트 인덱스 마스크를 생성합니다.

#### 목록 12.9 PyG Data 객체의 학습, 검증, 테스트 마스크


```python
Initializes the training set mask (all node Initializes the Initializes the
indices are marked as False by default) validation set mask testing set mask
train_mask = torch.zeros(data.num_nodes, dtype=torch.bool) Fills training set
val_mask = torch.zeros_like(train_mask) < indices with a
test_mask = torch.zeros_like(train_mask) < uniquely shuffled
nonzero_indices = known_mask.nonzero(as_tuple=True)[0] batch of True values
train_indices = nonzero_indices[permutations[:train_size]] from known_mask
```

```ini
val_indices = nonzero_indices[ Fills validation set indices with a
permutations[train_size:train_size + val_size] non-overlapping batch of True
] < values from known_mask
test_indices = nonzero_indices[permutations[train_size + val_size:]] ≤
train_mask[train_indices] = True < Sets the training
val_mask[val_indices] = True < set indices to Fills testing set indices
test_mask[test_indices] = True < True in the with another distinct,
data.train_mask = train_mask training set mask non-overlapping
data.val_mask = val_mask batch of True values
data.test_mask = test_mask Sets the validation set from known_mask
indices to True in the
Sets the testing set indices to validation set mask
True in the testing set mask
```
표 12.5는 학습과 평가를 위해 데이터가 어떻게 분할되는지 명확히 하기 위해 데이터셋의 통계를 보여 줍니다. 이 시나리오에서는 각 데이터셋에서 합법 및 불법 정보의 균형을 직접 확인할 수 있습니다.

표 12.5 각 데이터셋의 학습, 검증, 테스트 데이터셋 크기와 클래스 분포
<table><tr><td>데이터셋</td><td>총 개수</td><td>합법</td><td>합법 (%)</td><td>불법</td><td>불법 (%)</td></tr><tr><td>학습</td><td>37,251</td><td>33,645</td><td>90.32</td><td>3,606</td><td>9.78</td></tr><tr><td>검증</td><td>4,656</td><td>4,193</td><td>90.06</td><td>463</td><td>9.88</td></tr><tr><td>테스트</td><td>4,657</td><td>4,181</td><td>89.78</td><td>476</td><td>9.45</td></tr></table>

우리는 여러 스크립트를 정의하여 분류 작업을 위한 분할 과정을 수동으로 수행했습니다. 12.2.3절에서는 PyG 라이브러리가 데이터셋 분할을 돕는 기능을 제공한다는 것을 살펴볼 것입니다. 이제 노드 분류 시스템을 구축하기 위한 종단 간 아키텍처를 논의해 보겠습니다.

#### 12.1.4 인코더–디코더 아키텍처


11장에서는 인코더–디코더 아키텍처를 소개했습니다. 그림 12.4는 다운스트림 작업 (downstream task)에서 GNN 모델을 학습하는 데 포함되는 단계의 개요를 제공합니다.

다음 목록은 노드 분류를 위한 인코더–디코더를 정의하는 Python 클래스의 구현을 보여줍니다.

```python
Listing 12.10 Encoder–decoder implementation for node classification
import torch
import torch.nn.functional as F
class NodeClassifier(torch.nn.Module):
def __init__(self, gnn_model):
super().__init__()
self.gnn = gnn_model
def forward(self, x, edge_index):
x = self.gnn(x, edge_index)
return F.log_softmax(x, dim=1)
```

![](images/ko/figure-12-4-ko.png)  
그림 12.4 인코더–디코더 아키텍처의 주요 단계 개요. 거래(노드)와 비트코인 흐름(엣지)으로 구성된 동질 그래프 데이터 (homogeneous graph data)는 GCN, GAT, SAGE라는 세 가지 GNN 인코더를 사용하여 처리됩니다. 각 인코더는 이웃 특징을 집계하여 고유한 노드 표현을 생성합니다. 그런 다음 이러한 표현은 확률값으로 변환되며, 이는 거래를 합법 또는 불법으로 분류하는 데 사용됩니다.

NodeClassifier 클래스의 forward 메서드는 GNN이 이웃의 정보로 노드 표현을 갱신하기 위해 수행하는 인코딩 단계와, PyTorch가 제공하는 log\_softmax 함수가 실행하여 각 노드와 관련된 확률값을 생성하는 디코딩 단계를 포함합니다. 이러한 확률값은 노드가 합법인지 불법인지를 판정하는 데 사용됩니다. 이제 이러한 구성 요소의 구현을 자세히 살펴보겠습니다.

#### 인코더


인코더 구성 요소는 동종 GNN (homogeneous GNN)을 사용하여 메시지 전달 과정 (message-passing process)을 수행합니다(11장 참조). 이는 인코더가 금융 거래 네트워크에서 제공되는 것과 같이 단일 유형의 노드와 에지를 포함하는 동종 그래프를 처리하도록 설계되었음을 의미합니다. 서로 다른 GNN 인코더의 동작을 비교하기 위해, 다음 목록에서는 이러한 인코더의 공통 특성을 지정하는 기본 그래프 모델을 정의합니다.

#### 목록 12.11 일반적인 GNN 인코더를 정의하기 위한 기본 그래프 모델


```python
import torch
import torch.nn.functional as F
class BaseGraphModel(torch.nn.Module):
def __init__(self, input_dim,
hidden_dim, Defines the model's initialization, including
out_dim, the input, hidden, and output dimensions;
conv_layer, type of convolution layer; and arguments that
**conv_kwargs): < may be passed to the convolutional layer
```

```python
Calls the superclass initializer Initializes the first graph convolution
to set up the model layer with input and hidden dimensions
D super(BaseGraphModel, self).__init__()
self.conv1 = conv_layer(input_dim, hidden_dim, **conv_kwargs) <
self.conv2 = conv_layer(hidden_dim, out_dim, **conv_kwargs) <
def forward(self, x, edge_index): < Initializes the second graph
convolution layer with hidden
x = self.conv1(x, edge_index) ≤
and output dimensions
x = F.relu(x) <
x = self.conv2(x, edge_index) Defines the forward pass
Returns the return x function for the model
Applies the second
final node convolution layer to the Applies the first convolution layer to
features after transformed features the input features and edge indices
processing
Applies the ReLU activation function
to introduce nonlinearity
```

이 기본 클래스는 PyG에서 제공하는 어떤 GNN 구현이든 이웃 집계와 노드 갱신 연산(컨볼루션)에 사용할 수 있는 2계층 아키텍처를 보여 줍니다. \*\*conv\_kwargs 인수를 사용하면 특정 GNN 구현에 필요할 수 있는 추가 매개변수를 전달할 수 있습니다. 다음 목록에서 우리 SAGE 모델의 구현을 살펴봅니다.

#### 목록 12.12 기본 클래스에서 구현한 SAGE 모델


```python
from torch_geometric.nn import SAGEConv
class SAGE(BaseGraphModel):
def __init__(self, input_dim, hidden_dim, out_dim):
super(SAGE, self).__init__(
input_dim=input_dim,
hidden_dim=hidden_dim,
out_dim=out_dim,
conv_layer=SAGEConv
)
```

PyG의 SAGEConv 계층은 추가 매개변수를 필요로 하지 않으므로, 우리 SAGE 모델을 초기화하는 것은 간단합니다. 이 코드는 두 개의 SAGEConv 계층을 가진 GNN 모델을 생성합니다.

다른 GNN 모델은 우리 기본 클래스를 확장해야 합니다. 다음 목록에 표시된 GAT 모델의 구현을 살펴보겠습니다.

#### 목록 12.13 기본 클래스를 확장하여 구현한 GAT 모델


```python
from torch_geometric.nn import GATConv
class GAT(BaseGraphModel):
def _init__(self, input_dim,
hidden_dim,
out_dim,
num_heads=8, Initializes the
add_self_loops=True): < GAT model
```

![](images/ko/listing-12-13-continuation-ko.png)

이 경우 PyG에서 구현된 GATConv 계층이 요구하는 추가 매개변수를 포함하기 위해 합성곱 계층을 다시 구현해야 합니다.

#### 디코더

디코더 구성 요소는 PyTorch의 log\_softmax 함수를 사용하여 신경망의 출력, 구체적으로는 GNN 인코더의 출력을 확률값으로 변환합니다. 표준 softmax 함수와 달리, log\_softmax는 확률 계산 중 매우 작거나 큰 수의 영향을 완화함으로써 학습 과정의 안정성을 향상합니다.

이 맥락에서 log\_softmax 함수는 PyTorch의 Cross-EntropyLoss 함수와 함께 작동합니다. 이 함수는 softmax 확률의 로그와 음의 로그우도 손실 (negative log-likelihood loss)을 결합하여 분류 작업에 효율적인 접근 방식을 제공합니다. 예를 들어 AML 시스템에서 GNN 인코더는 의심스러운 행동을 탐지하기 위해 거래 데이터를 분석할 수 있으며, 각 노드에는 합법 또는 불법과 같은 서로 다른 범주에 대한 확률이 할당될 수 있습니다.

log\_softmax 함수는 이러한 확률이 수치적으로 안정적이고 해석 가능하도록 보장하며, CrossEntropyLoss 함수는 모델의 예측이 실제 레이블과 얼마나 잘 일치하는지를 측정합니다. 모델이 불법에 대해 높은 확률을 예측했지만 해당 노드가 합법으로 레이블링되어 있다면, 손실 함수는 오류를 식별하고 정확도를 향상하도록 모델을 최적화합니다.

#### 12.1.5 평가 및 분석


우리의 분석에서는 노드 분류를 위한 세 가지 엔드투엔드 모델을 비교했습니다. 각 모델은 서로 다른 GNN 인코더, 즉 GCN, GAT, 또는 SAGE를 사용합니다. 표 12.6은 각 모델의 파라미터 수와 총 학습 시간을 초 단위로 보여주며, 우리는 T4 Colab 머신에서 400 에포크 이후 이러한 결과를 얻었습니다.

참고 코드 예제를 실행할 때, 결과가 본문에 제시된 것과 약간 다를 수 있습니다. 이는 머신러닝에서 정상적인 동작이며, 알고리즘은 종종 무작위성(예를 들어 초기화 또는 샘플링)을 포함합니다. 이러한 변동성은 코드에 오류가 있음을 의미하지 않습니다.

표 12.6 각 인코더의 파라미터 수와 총 학습 시간
<table><tr><td>인코더</td><td>파라미터</td><td>학습 시간(초)</td></tr><tr><td>GCN</td><td>2,723</td><td>19.02</td></tr><tr><td>GAT</td><td>22,025</td><td>43.45</td></tr><tr><td>SAGE</td><td>5,427</td><td>36.71</td></tr></table>

결과는 GCN 모델이 가장 효율적이고, GAT 모델이 가장 비효율적이며, SAGE 모델은 다른 두 모델 사이에 위치함을 나타냅니다. 짐작할 수 있듯이, 학습 시간은 파라미터 수와 직접적으로 연결되어 있습니다. 12.1.4절에서 우리는 GAT 모델이 이웃 에지에 대한 학습 가능한 계수를 도입하는 주의 메커니즘(attention mechanism)을 포함하며, 이것이 학습 파라미터 수를 증가시킨다고 설명했습니다.

#### 정밀도, 재현율, F1-점수


이 모델들의 일반화 능력을 탐색하기 위해 정밀도 (precision), 재현율 (recall), F1-점수 (F1-score) 지표에서 성능을 평가할 수 있습니다. AML을 위한 노드 분류의 맥락에서, 이러한 지표는 다음 질문들에 답할 수 있게 해 줍니다.

정밀도—모델이 어떤 노드가 합법 또는 불법이라고 말할 때, 얼마나 자주 맞습니까?

재현율—존재하는 모든 합법 및 불법 노드 중에서, 모델이 올바르게 찾아낸 것은 얼마나 됩니까?

F1-점수—모델은 예측을 할 때 정확한 것과 가능한 한 많은 올바른 노드를 찾아내는 것 사이에서 얼마나 잘 균형을 맞춥니까?

그림 12.5는 학습 단계 동안 검증 데이터셋에서 정밀도, 재현율, F1-점수 지표 전반에 걸친 GCN, GAT, SAGE의 성능 플롯을 보여 줍니다. 세로축은 지표 값을 나타내고, 가로축은 에포크 수를 나타냅니다.

우리는 정밀도, 재현율, F1-점수에 대한 sklearn 라이브러리의 내장 채점 함수를 사용하여 이러한 결과를 얻었습니다. 이러한 채점 함수는 합법 노드가 불법 노드보다 훨씬 더 많은 우리 데이터셋과 같은 불균형 데이터셋을 처리하기 위해 average 파라미터를 제공합니다. 우리는 데이터 분포를 반영하는 성능 지표를 제공하고 일반적인 평가에 적합하기 때문에 가중 평균을 사용하기로 결정했습니다.

![](images/ko/figure-12-5a-ko.png)

![](images/ko/figure-12-5b-ko.png)

![](images/ko/figure-12-5c-ko.png)  
그림 12.5 학습 단계 동안 검증 데이터셋에서 정밀도, 재현율, F1-점수 전반에 걸친 GCN, GAT, SAGE의 성능

합법 및 불법 노드를 탐지하는 AML 시나리오의 맥락에서 앞서 제기한 질문들에 답해 보겠습니다.

 정밀도—모델들이 어떤 노드가 합법인지 불법인지 예측할 때, SAGE 모델이 가장 정확하며 학습 단계 전반에 걸쳐 일관되게 높은 정밀도를 유지합니다. 이는 SAGE가 거짓 양성 (false positive)을 최소화하고 합법 및 불법 노드에 대해 더 신뢰할 수 있는 예측을 수행함을 나타냅니다. GAT 모델도 이 측면에서 좋은 성능을 보이며 SAGE를 근접하게 따라갑니다. GCN 모델은 가장 낮은 정밀도를 보이며, 특히 초기 에포크에서 합법 및 불법 노드를 자주 잘못 분류합니다.

재현율—모델들은 실제 합법 및 불법 노드를 모두 식별하는 데 있어 유사한 성능을 보이며, 학습이 진행됨에 따라 세 모델 모두 비슷한 재현율 값을 달성합니다. 이는 GCN, GAT, SAGE가 거의 모든 합법 및 불법 노드를 효과적으로 탐지할 수 있음을 의미합니다. 그러나 재현율만으로는 거짓 양성을 설명하지 못하므로, 전체 모델 성능을 이해하려면 정밀도와 결합해야 합니다.

 F1-score—이 점수는 합법 및 불법 노드를 탐지할 때 모델이 정밀도와 재현율의 균형을 얼마나 잘 맞추는지를 반영합니다. SAGE 모델은 가장 높은 값을 달성하여 적절한 균형을 찾는다는 것을 입증합니다. 즉, 대부분의 합법 및 불법 노드를 식별하고 잘못된 예측을 최소화합니다. GAT 모델은 SAGE와 거의 비슷하게 강력하며, 차이는 미미합니다. GCN 모델은 높은 재현율에도 불구하고 낮은 정밀도로 인해 높은 점수를 달성하는 데 어려움을 겪으며, 이는 이러한 지표들의 균형을 맞추는 전반적 성능이 더 약함을 나타냅니다.

#### 혼동 행렬


전반적인 동작을 넘어, 합법 노드에서의 이들 모델 성능과 불법 노드에서의 성능을 구분하기 위해 더 구체적인 평가를 수행할 수 있습니다. 이를 위해 앞서 정의한 지표들과 비교하여 각 클래스(합법 또는 불법)에 대한 올바른 예측과 잘못된 예측의 개수를 보여 줌으로써 분류 성능을 세분화하는 혼동 행렬 (confusion matrix)을 사용할 수 있습니다. 그림 12.6은 테스트 데이터셋에서 GCN, GAT, SAGE 모델의 혼동 행렬을 보여 줍니다.

![](images/ko/figure-12-6a-ko.png)

![](images/ko/figure-12-6b-ko.png)

![](images/ko/figure-12-6c-ko.png)  
그림 12.6 각 클래스(합법 또는 불법)에서 GCN, GAT, SAGE 모델의 성능을 이해하기 위한 혼동 행렬

전반적으로 SAGE가 가장 우수한 성능을 보이며, 대부분의 합법 및 불법 노드를 올바르게 분류하고 오분류를 최소화합니다. GAT 모델도 강력한 성능을 보이며 정확도에서 SAGE에 근접하지만, 불법 클래스에서 오분류가 약간 더 많습니다. GCN 모델은 가장 약한 성능을 보이며, 불법 노드에 대한 정밀도와 재현율이 더 낮아 다른 모델들과 비교했을 때 이러한 노드를 구분하는 데 어려움이 있음을 나타냅니다. 각 모델의 동작을 자세히 분석해 보겠습니다.

 GCN—이 모델은 합법 및 불법 노드를 분류하는 데 중간 정도의 성능을 보입니다. 불법 노드의 약 68%와 합법 노드의 약 99%를 올바르게 분류합니다. 그러나 불법 노드의 약 3분의 1이 합법으로 잘못 분류되어, 불법 노드를 올바르게 식별하는 데 어려움이 있음을 보여 줍니다. 합법 노드의 경우, 합법 노드의 약 1%가 불법으로 잘못 예측되어 합법 노드 예측에서는 더 나은 정확도를 보입니다. 이러한 불균형은 GCN이 합법 노드를 효과적으로 식별하지만, 불법 노드에 대한 정밀도는 개선이 필요함을 나타냅니다.

GAT—이 모델은 불법 노드의 약 81%와 합법 노드의 약 99.5%를 올바르게 분류하여 GCN보다 우수한 성능을 보입니다. GCN과 비교했을 때 불법 클래스에 대한 오분류를 줄입니다. 즉, 불법 노드 약 5개 중 1개가 합법으로 잘못 분류되는 반면, 합법 노드 중 불법으로 잘못 분류되는 것은 40개에 불과합니다. 이러한 개선은 GAT가 두 클래스 전반에 걸쳐 더 신뢰할 수 있는 균형을 제공한다는 것을 입증합니다.

SAGE—이 모델은 세 모델 중 가장 높은 성능을 달성합니다. 불법 노드의 약 83%와 합법 노드의 약 99%를 올바르게 분류합니다. 오분류율은 가장 낮습니다. 불법 노드 5개 중 1개 미만이 합법으로 잘못 분류되며(88개), 합법 노드 중 불법으로 잘못 분류되는 것은 약 50개에 불과합니다.

전반적인 분석에 따르면, SAGE 모델은 우수한 균형, 합법 및 불법 노드 모두에 대한 더 높은 정확도, 최소한의 오분류로 인해 가장 효과적이고 신뢰할 수 있는 선택지입니다. 특히 합법 노드에 대해 높은 정확도를 유지하면서 불법 노드 탐지가 중요한 자금세탁방지 (AML)와 같은 응용 분야에 매우 적합합니다.

### 12.2 영화 추천을 위한 링크 예측


링크 예측 (Link prediction)은 그래프 기반 ML에서 핵심적이며, 추천 시스템과 같은 응용 분야와 특히 관련이 깊습니다. 그래프 구조를 사용하면 상호작용과 선호를 엔터티 간의 링크로 모델링할 수 있으며, 이 경우에는 사용자와 영화 간의 관계를 나타냅니다.

이 절에서는 MovieLens 데이터셋을 데이터 소스로 사용하여 사용자–영화 링크를 예측하기 위한 그래프 신경망 (GNN)의 활용을 살펴봅니다. 그림 12.7은 종단 간 프레임워크를 보여줍니다. 목표는 관련 없는 추천을 피하면서 관련성 높은 영화를 제안하는 GNN의 능력을 평가하여 추천 과정을 향상하는 것입니다.

![](images/ko/figure-12-7-ko.png)  
그림 12.7 추천 시스템 맥락에서의 링크 예측을 위한 종단 간 프레임워크. 상호작용/선호 데이터는 사용자와 영화라는 두 가지 유형의 노드를 포함하는 이질 그래프 (heterogeneous graph) 구조로 변환됩니다. 인코더인 이질 GNN은 국소 그래프 구조를 포착하여 노드 임베딩을 학습합니다. 디코더는 내적 (dot-product) 연산을 이진 교차 엔트로피 (binary cross-entropy) 손실과 결합하여 사용자와 영화 간 링크의 존재를 예측함으로써, 사용자에게 관련성 높은 영화를 제안하기 위한 학습된 모델을 생성합니다.

#### 12.2.1 입력 데이터


추천 목적의 링크 예측을 수행하는 GNN의 역량을 탐구하기 위해 MovieLens 데이터셋의 소규모 버전을 사용합니다. 이 MovieLens 데이터셋 버전에는 600명의 사용자가 9,000개의 영화에 적용한 100,000개의 평점과 3,600개의 태그 적용 사례가 포함되어 있습니다. 2025년 1월 현재, 원시 데이터는 https:// files.grouplens.org/datasets/movielens/ml-latest-small.zip 에서 사용할 수 있으며, 다음 파일들을 포함합니다.

movies.csv—9,742개의 행과 3개의 열로 구성된 CSV 파일입니다. 각 열은 순서대로 영화의 ID, 영화의 제목, 장르를 정의합니다.

ratings.csv—100,836개의 행과 4개의 열로 구성된 CSV 파일입니다. 각 열은 사용자 ID, 영화 ID, 평점, 타임스탬프를 정의합니다.

링크 예측 작업을 위해 사용 가능한 열들 중 일부에 집중합니다. movies.csv의 경우 movieId와 genres 열을 사용합니다. 표 12.7은 이러한 열 부분집합을 포함한 이 파일의 샘플을 보여 줍니다. 영화는 1부터 시작하는 증가형 ID로 식별되며, 영화의 장르는 파이프 문자(|)로 구분된 범주형 문자열의 모음으로 제공됩니다.

표 12.7 movies.csv 파일에서 movieId 및 genres 열의 샘플
<table><tr><td>movieId</td><td>genres</td></tr><tr><td>1</td><td>Adventure|Animation|Children|Comedy|Fantasy</td></tr><tr><td>2</td><td>Adventure|Children|Fantasy</td></tr><tr><td>3</td><td>Comedy|Romance</td></tr><tr><td>4</td><td>Comedy|Drama|Romance</td></tr><tr><td>5</td><td>Comedy</td></tr></table>

ratings.csv 파일에서는 userId와 movieId 열만 고려합니다. 표 12.8은 샘플을 제공합니다. 각 행은 증가형 ID로 식별되는 특정 사용자가 특정 영화(movies.csv 파일에서 사용된 동일한 ID로 식별됨)에 부여한 평점을 정의합니다.

표 12.8 ratings.csv 파일에서 userId 및 movieId 열의 샘플
<table><tr><td>userId</td><td>movieId</td></tr><tr><td>1</td><td>1</td></tr><tr><td>2</td><td>3</td></tr><tr><td>2</td><td>6</td></tr><tr><td>1</td><td>47</td></tr><tr><td>1</td><td>50</td></tr></table>

#### 12.2.2 그래프 프로세서: 데이터 준비


노드 분류 작업에서와 마찬가지로, 원본 데이터의 간결한 수치 표현과 GNN 학습 단계를 위한 평점 그래프 구조를 얻기 위해 데이터를 준비해야 합니다. 먼저 movies.csv 파일을 처리하는 것부터 시작하겠습니다. 주요 목표는 다음 목록과 같이 장르 정보를 수치적으로 처리 가능한 것, 즉 특징 벡터로 변환하는 것입니다.

목록 12.14 장르 정보를 특징 벡터로 변환하기   
영화 데이터셋을 읽습니다   
movies\_df = pd.read\_csv(movies\_path, index\_col='movieId')   
그리고 이를 메모리에 로드합니다   
genres = movies\_df['genres'].str.get\_dummies('|')   
movie\_feat = torch.from\_numpy(genres.values).to(torch.float) <   
assert movie\_feat.size() == (9742, 20) <   
새 텐서를 생성합니다. 여기서   
이진 표현의 일관성을 확인합니다   
장르를 분리하고 변환합니다   
장르 수와 함께 장르는 특징 집합입니다   
이를 이진 지표로 변환합니다   
영화에 연결된 원래 표현입니다

movie\_feat 변수에 저장된 출력의 예는 표 12.9에 나와 있습니다. 이 표현에서 각 영화는 0과 1을 포함하는 특징 벡터와 연결되며, 특정 열의 값이 1이면 해당 영화가 대응하는 장르에 속함을 나타냅니다.

표 12.9 영화 장르에 대한 특징 벡터
<table><tr><td rowspan=1 colspan=1>movieId</td><td rowspan=1 colspan=1>액션</td><td rowspan=1 colspan=1>어드벤처</td><td rowspan=1 colspan=1>드라마</td><td rowspan=1 colspan=1>호러</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=3 colspan=1>234</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=2 colspan=1>0o</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>o</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr></table>

우리 예에서 ID가 1인 영화는 어드벤처 영화로 분류될 수 있으며, ID가 4인 영화는 드라마 영화로 분류됩니다. 여러 장르 값이 동일한 영화와 연결될 수 있습니다.

사용자와 영화 사이의 연결을 설명하는 edge\_index 텐서를 생성해야 합니다(목록 12.15). 첫 번째 단계는 영화와 사용자의 원래 ID를 0부터 시작하는 증가 ID에 매핑하는 것입니다. 그런 다음 새 ID를 사용하여 edge\_index를 생성할 수 있습니다.

목록 12.15 사용자 및 영화 ID로부터 edge\_index 텐서 생성하기   
ratings\_df = pd.read\_csv(ratings\_path) < 평점 파일을 읽고   
메모리에 로드합니다   
unique\_user\_id = ratings\_df['userId'].unique()   
unique\_user\_id = pd.DataFrame(data={ 원래 사용자 ID에서   
'userId': unique\_user\_id, 새로운 데이터 프레임을 생성하며   
'mappedID': pd.RangeIndex(len(unique\_user\_id)), [0, num\_user\_nodes) 범위로   
}) < 매핑합니다   
unique\_movie\_id = pd.DataFrame(data={ 원래 영화 ID에서   
'movieId': movies\_df.index, 새로운 데이터 프레임을 생성하며   
'mappedID': pd.RangeIndex(len(movies\_df)), [0, num\_movie\_nodes) 범위로   
}) < 매핑합니다   
ratings\_user\_id = pd.merge(   
ratings\_df['userId'],   
unique\_user\_id,   
left\_on='userId',   
평점의 새로운 원천 사용자   
right\_on='userId',   
ID를 새로운 데이터 프레임에   
how='left'   
저장합니다   
)   
ratings\_user\_id = torch.from\_numpy(ratings\_user\_id['mappedID'].values) <

사용자 ID를 연속된 값으로 매핑한 결과:   
userId mappedID   
0 1 0   
1 2 1   
2 3 2   
3 4 3   
4 5 4   
영화 ID를 연속된 값으로 매핑한 결과:   
movieId mappedID   
0 1 0   
1 2 1   
2 3 2   
3 4 3   
4 5 4   
사용자에서 영화로 향하는 최종 엣지 인덱스:   
==   
tensor([[ 0, 0, 0, ... 609, 609, 609],   
[ 0, 2, 5, ..., 9462, 9463, 9503]])

```python
ratings_movie_id = pd.merge(
ratings_df['movieId'],
unique_movie_id,
left_on='movieId',
Stores the new target
right_on='movieId', movie IDs of the ratings
how='left' into a new data frame
ratings_movie_id = torch.from_numpy(ratings_movie_id['mappedID'].values)
edge_index_user_to_movie = torch.stack([ratings_user_id, ratings_movie_id],
dim=0) < Creates the edge_index by stacking
the source and target data frames
```

사용자와 영화에 대한 ID 매핑의 출력과 edge\_index 텐서는 다음 목록에 표시되어 있습니다.

#### 목록 12.16 ID 매핑의 출력 및 edge_index의 샘플


엣지 인덱스 텐서의 크기는 [2, 100836]이며, 두 번째 원소는 데이터셋의 평점 수에 해당합니다. 전처리 단계 이후, 마침내 사용자와 영화 간의 상호작용으로 평점을 표현하기 위한 PyG 그래프 구조를 생성할 수 있습니다.

#### 12.2.3 그래프 프로세서: 이질적 PyG 그래프


그림 12.8은 그래프 프로세서가 적용하는 단계의 개요를 제공합니다. 여기에는 준비 단계, PyG HeteroData 객체의 구성, 학습, 검증 및 테스트 데이터셋의 생성, 미니배치의 준비가 포함됩니다.

![](images/ko/figure-12-8-ko.png)  
그림 12.8 MovieLens 데이터셋을 위한 데이터 처리 파이프라인으로, 증분 ID 매핑, 장르 인코딩, 엣지 생성을 포함합니다. 이질적 그래프 데이터는 두 가지 노드 유형(사용자와 영화)과 하나의 엣지 유형(사용자-평가-영화)으로 구성됩니다. 엣지는 학습(80%), 검증(10%), 테스트(10%) 세트로 분할되며, 검증 및 테스트를 위해 부정 예제가 생성됩니다. 그런 다음 미니배치 로더가 GNN 입력을 위한 부분 그래프를 준비하여, 메모리 용량을 초과하는 대규모 그래프에 대해 확장성을 보장합니다.

우리 시나리오에는 두 가지 유형의 노드, 즉 사용자와 영화가 있습니다. 우리는 이러한 노드에 대한 정보를 표현하기 위해 PyG의 HeteroData를 사용하여 그래프를 구성합니다. 단일 노드 유형과 단일 엣지 유형을 가정하는 동질적 그래프에서 사용되는 Data 클래스와 달리, HeteroData는 각 노드 유형의 특징을 구분하고 edge\_index를 특정 관계와 연결할 수 있게 해 줍니다. 이 경우에는 평가 관계입니다. 다음 목록은 관련 장르로 생성된 movie\_feat와 edge\_index로 구성된 전처리 단계의 결과를 사용하여 PyG HeteroData 객체를 구성합니다.

#### 목록 12.17 PyG HeteroData 객체 구축


```python
from torch_geometric.data import HeteroData
data = HeteroData()
data["user"].node_id = torch.arange(len(unique_user_id))
data["movie"].node_id = torch.arange(len(movies_df))
```

data["movie"].x = movie\_feat   
data["user", "rates", "movie"].edge\_index = edge\_index\_user\_to\_movie   
data = T.ToUndirected()(data)

이 경우, 사용자에서 영화로 그리고 그 반대로 GNN 메시지 전달 (message passing)이 명시적으로 이루어지도록 역방향 엣지 (reverse edges)를 추가해야 합니다.

그래프를 생성한 후에는 다음 목록에 보인 것처럼 평가 엣지를 학습, 검증, 테스트 데이터셋으로 분할할 수 있습니다. 우리의 주요 목표는 링크 측면에서 이러한 데이터셋 간 중복을 피하는 것입니다. 노드 분류의 맥락에서는 노드를 분리하기 위해 이 분할 작업을 수동으로 수행했지만, 이 경우에는 PyG에 내장된 transforms.RandomLinkSplit 함수를 사용합니다.

목록 12.18 학습, 검증, 테스트 데이터셋 생성   
import torch\_geometric.transforms as T 검증 엣지의 비율(10%)을 정의합니다   
검증 엣지의   
transform = T.RandomLinkSplit( 감독을 위한 엣지 비율을   
num\_val=0.1, < 정의합니다   
num\_test=0.1, < 테스트 엣지의 비율(10%)과 메시지   
disjoint\_train\_ratio=0.3, < 전달(70%)을 정의합니다   
neg\_sampling\_ratio=2, < 검증 및 테스트 데이터셋에서   
add\_negative\_train\_samples=False, < 각 기존 엣지에 대한   
edge\_types=("user", "rates", "movie"), < 음성 샘플 (negative samples)의 수(2)를 정의합니다   
V rev\_edge\_types=("movie", "rev\_rates", "user"),   
메시지 전달 및 학습을 위한 기존 엣지를 정의합니다   
학습 데이터셋에서 음성 학습 샘플이 없도록(False) 설정합니다   
메시지 전달을 위한 역방향 엣지를 정의하지만 학습을 위한 것은 아닙니다

transforms.RandomLinkSplit 함수는 ("user", "rates", "movie") 관계의 엣지를 학습, 검증, 테스트 엣지로 무작위 분할합니다. 전통적인 데이터셋 분할과 비교할 때, GNN의 경우에는 다른 요인들도 고려해야 합니다. 예를 들어 disjoint\_train\_ratio 매개변수는 학습 엣지를 다시 두 개의 서로 다른 그룹으로 세분합니다.

메시지 전달에 사용되는 엣지로, edge\_index 변수에 저장됩니다

감독에 사용되는 엣지로, edge\_label\_index 변수에 저장됩니다

이 두 유형의 엣지 간 차이는 다음 학습 데이터 구조에 반영되어 있습니다.

목록 12.19 학습 데이터의 세부 사항   
학습 데이터:   
===   
HeteroData(   
user={   
node\_id=[610]   
},   
movie={

```prolog
node_id=[9742],
x=[9742, 20]
},
(user, rates, movie)={
edge_index=[2, 56469],
edge_label=[24201],
edge_label_index=[2, 24201]
},
(movie, rev_rates, user)={
edge_index=[2, 56469]
}
```

학습 세트의 크기는 80,669이며, 이는 100,836(전체 엣지 수)의 80%에 해당합니다. 그러나 학습 HeteroData는 edge\_index에 대해 56,469(80,669의 70%)에 해당하는 값을, edge\_label\_index에 대해 24,201(80,669의 30%)에 해당하는 값을 보고합니다. 메시지 전달에 사용되는 엣지와 감독에 사용되는 엣지 간 중복을 피하기 위해 이러한 엣지 집합들이 서로 분리되어 있다는 점을 기억하는 것이 중요합니다.

또한 이종 그래프의 맥락에서 역방향 엣지는 rev\_edge\_types 매개변수로 지정됩니다. 역방향 엣지는 메시지 전달 (message passing)에 사용되지만 링크 예측 모델을 학습하는 데는 사용되지 않습니다. 다음 HeteroData 객체는 우리의 검증 및 테스트 데이터셋 구조를 보여 줍니다.

#### Listing 12.20 검증 및 테스트 데이터셋의 세부 사항


검증 데이터:   
HeteroData(   
user={   
node\_id=[610]   
},   
movie={   
node\_id=[9742],   
x=[9742, 20]   
},   
(user, rates, movie)={   
edge\_index=[2, 80670],   
edge\_label=[30249],   
edge\_label\_index=[2, 30249]   
},   
(movie, rev\_rates, user)={   
edge\_index=[2, 80670]   
}   
)   
테스트 데이터:   
HeteroData(   
user={   
node\_id=[610]   
},

movie={   
node\_id=[9742],   
x=[9742, 20]   
},   
(user, rates, movie)={   
edge\_index=[2, 90753],   
edge\_label=[30249],   
edge\_label\_index=[2, 30249]   
},   
(movie, rev\_rates, user)={   
edge\_index=[2, 90753]   
}   
)

이 목록에 표시된 차원은 검증 및 테스트 데이터셋에서 어떤 엣지가 메시지 전달에 사용되고 어떤 엣지가 모델의 적합도를 평가하는 데 사용되는지 이해할 수 있게 해 줍니다. 검증 데이터셋의 경우 edge\_index 크기는 80,670으로, 학습 데이터셋의 엣지 수와 같습니다. edge\_label\_index 크기는 30,249이며, 이는 10,083(100,836의 10%)에 해당하고, 그 결과 RandomLinkSplit에 의해 20,166개의 음성 엣지 (negative edge)가 생성됩니다.

이 맥락에서 80,670개의 학습 엣지는 메시지 전달에 사용되고, 30,249개의 엣지는 검증 데이터셋에서 링크 예측 작업에 대한 모델을 평가하는 데 사용됩니다. 테스트 세트에도 유사한 원리가 적용됩니다. 메시지 전달에 사용되는 엣지(90,753)는 학습 엣지(80,670)와 검증 엣지(10,083)를 포함합니다.

데이터셋을 분할한 후 다음 단계는 우리의 GNN에 입력하기에 적합한 서브그래프를 생성할 수 있는 미니배치 로더 (mini-batch loader)를 정의하는 것입니다. 이 단계는 소규모 그래프에는 필수적이지 않을 수 있지만, CPU 또는 GPU 메모리 용량을 초과하는 더 큰 그래프에서 GNN을 사용하는 데 중요합니다.

이를 위해 우리는 PyG의 loader.LinkNeighborLoader 컴포넌트를 사용하여 입력 엣지 집합에서 엣지 표본을 선택합니다. 다음 목록에 표시된 것처럼, 이는 각 반복에서 이웃 수를 샘플링함으로써 이 목록의 모든 노드로부터 서브그래프를 구성합니다.

#### 목록 12.21 각 반복에서 이웃 수 샘플링


```python
from torch_geometric.loader import LinkNeighborLoader
edge_label_index = train_data["user", "rates", "movie"].edge_label_index
edge_label = train_data["user", "rates", "movie"].edge_label
train_loader = LinkNeighborLoader( Samples at most 20 neighbors
data=data, in the first hop and at most 10
num_neighbors=[20, 10], < neighbors in the second hop
neg_sampling_ratio=2, <
edge_label_index=(("user", "rates", "movie"), edge_label_index),
edge_label=edge_label,
batch_size=128, < Sets the batch size in terms Defines the ratio of
shuffle=True < of the number of edges negative samples (2:1)for each existing edge
Random order created in the batch
of edges
```

이 과정은 검증 및 테스트 데이터셋에서도 수행된다는 점에 유의해야 합니다. 데이터 준비 단계를 설명한 후에는 링크 예측 과제를 처리하기 위한 아키텍처를 이해할 수 있습니다.

#### 12.2.4 인코더-디코더 아키텍처


링크 예측 시스템 역시 인코더-디코더 아키텍처에 기반합니다. 그러나 이 절에서 논의하는 아키텍처는 입력 데이터의 특성, 즉 이종 그래프 구조 (heterogeneous graph structure)와 수행할 다운스트림 과제 (downstream task), 즉 추천을 위한 링크 예측에 기반한 특정 구성 요소들로 이루어져 있으며, 그림 12.9는 관련 단계의 개요를 제공합니다. 다음 목록은 이 시나리오에서의 인코더-디코더 아키텍처를 구현합니다.

![](images/ko/figure-12-9-ko.png)  
그림 12.9 영화 추천을 위한 링크 예측 시스템에서 이종 그래프 데이터를 처리하는 파이프라인. 데이터는 노드 유형으로 사용자와 영화를 포함하고, 사용자-영화 평점을 엣지로 포함합니다. 임베딩이 생성되며, 사용자 임베딩은 모델이 학습하고 영화 임베딩은 장르 특징으로 초기화됩니다. 데이터는 세 가지 이종 GNN 인코더(H-GraphConv, H-GAT, H-SAGE)를 사용하여 처리되며, 이 인코더들은 이웃 특징을 집계하여 노드 표현을 생성합니다. 그런 다음 내적은 사용자와 영화 간의 적합성을 정량화하며, 점수가 높을수록 상호작용 가능성이 더 크다는 것을 나타냅니다. 마지막으로 점수는 사용자와 영화 사이의 링크 가능성을 나타내는 확률로 변환됩니다.

#### 목록 12.22 링크 예측을 위한 인코더–디코더 아키텍처


```python
super().__init__()
self.embedding = MovieLensEmbedding(
data["user"].num_nodes,
data["movie"].num_nodes, Initializes the embedding representation
hidden_channels for user and movie nodes using the
) < MovieLensEmbedding class
self.gnn = gnn_model(
data.metadata(), Initializes the model architecture
hidden_channels, (a heterogeneous version of
hidden_channels, modified GraphGCN, GAT, and SAGE)
hidden_channels
) < Initializes the final layer (decoder),
which performs a dot-product operation
self.classifier = DotProduct() < of user and movie embeddings
Performs forward propagation
def forward(self, data): <
and computes the prediction
x_dict = self.embedding(data)
x_dict = self.gnn(x_dict, data.edge_index_dict)
pred = self.classifier(
x_dict["user"],
x_dict["movie"],
data["user", "rates", "movie"].edge_label_index
)
return pred
```

MovieLensLinkPredictor의 forward 메서드는 데이터가 인코딩–디코딩 과정으로 전파되는 방식을 보여줍니다. 인코딩 단계는 사용자 및 영화 데이터 표현을 향상하기 위해 두 단계를 결합합니다.

임베딩 생성—이 단계는 사용자와 영화 모두에 대해 임베딩을 생성하여 그 특징의 표현력을 향상합니다. 사용자는 내재적 특징과 연결되어 있지 않기 때문에, 사용자 임베딩은 모델로부터 학습됩니다. 영화의 경우 장르를 인코딩한 특징 벡터가 임베딩 과정의 입력으로 사용되어, 각 영화 노드에 의미 있는 초기 표현을 제공합니다.

이종 GNN 모델—임베딩은 이종 GNN 모델을 사용하여 갱신됩니다. 이 모델은 그래프 구조를 사용해 이웃 노드의 정보를 집계함으로써 노드 표현을 정제합니다. 구체적으로 사용자 임베딩은 사용자가 상호작용한 영화의 정보로 갱신되고, 영화 임베딩은 해당 영화와 상호작용한 사용자의 정보로 갱신됩니다. 이러한 양방향 정보 교환은 임베딩이 그래프의 관계적 구조를 효과적으로 포착하도록 보장합니다.

디코딩 단계는 학습된 임베딩을 사용하여 예측을 수행합니다. 이는 내적 연산을 사용하여 구현되며, 사용자 노드와 영화 노드의 임베딩을 결합해 유사도 점수를 계산합니다. 이 점수는 사용자와 영화 사이의 링크(평점) 가능성을 나타냅니다. 내적은 모델이 학습된 임베딩을 기반으로 사용자–영화 관계를 예측할 수 있게 합니다. 이제 링크 예측 시스템에서 인코더와 디코더 구현을 분석해 보겠습니다.

#### 인코더


이 절의 시작 부분에서 소개한 바와 같이, 인코더 구성 요소는 두 단계로 이루어집니다. 첫 번째 단계는 임베딩을 생성하고, 두 번째 단계는 이종 GNN 모델을 적용합니다. 임베딩 생성 단계는 다음에 제시되어 있습니다.

#### 목록 12.23 임베딩 생성을 위한 클래스


```python
import torch
class MovieLensEmbedding(torch.nn.Module):
def __init__(self, user_input_dim, movie_input_dim, out_dim):
super().__init__()
self.movie_lin = torch.nn.Linear(20, out_dim)
self.user_emb = torch.nn.Embedding(user_input_dim, out_dim)
self.movie_emb = torch.nn.Embedding(movie_input_dim, out_dim)
def forward(self, data):
return {
"user": self.user_emb(data["user"].node_id),
"movie": self.movie_lin(data["movie"].x) +
self.movie_emb(data["movie"].node_id),
}
```

사용자 특징과 영화 특징에는 서로 다른 임베딩 생성 접근법을 적용합니다. 사용자에 대해서는 단일 단계 접근법을 사용하여, 전체 사용자 노드 수(user\_input\_dim)와 같은 행 수를 가지며 out\_dim 매개변수로 정의되는 열 수를 갖는 임베딩 행렬을 생성합니다. 이 맥락에서 사용자는 자신과 연관된 초기 특징을 전혀 갖지 않으며, 해당 임베딩은 학습 중에 오직 이 행렬로부터만 학습됩니다.

영화에 대해서는 두 단계 접근법을 사용합니다. 먼저 영화 장르를 인코딩하는 20차원 벡터로 표현된 입력 특징에 선형 변환을 적용합니다. 그런 다음 이 변환의 출력을 임베딩 층과 결합합니다. 이 방법은 초기 특징의 표현력을 향상시켜, 모델이 학습된 임베딩과 변환된 특징 표현을 모두 포착할 수 있게 합니다.

인코딩 단계의 두 번째 단계에서는 이종 GNN (heterogeneous GNN)을 사용합니다. 이는 금융 거래 그래프에서의 노드 분류와 달리, 인코더가 여러 유형의 노드와 엣지를 관리하도록 설계되어 있음을 의미합니다(다음 목록 참조).

#### 리스팅 12.24 이종 기본 인코더의 구현


```python
import torch
from torch_geometric.nn import to_hetero Defines the model’s
initialization
class HeteroBaseModel(torch.nn.Module):
def init__(self, metadata, input_dim, hidden_dim, out_dim, base_model):
super(HeteroBaseModel, self).__init__() < Calls the superclass
initializer to set up
the model
```

```python
Initializes the base homogeneous Converts the base homogeneous
model with the specified input, model into a heterogeneous model
hidden, and output dimensions using the provided metadata
→ self.base_model = base_model(input_dim, hidden_dim, out_dim)
self.hetero_model = to_hetero(self.base_model, metadata=metadata)
def forward(self, x_dict, edge_index_dict):
return self.hetero_model(x_dict, edge_index_dict) <
Defines the forward pass function Propagates the heterogeneous node features
for the heterogeneous graph model and edge indices through hetero_model to
produce embeddings for each node type
```

이 이종 기본 클래스는 동종 버전보다 더 많은 구성 요소를 포함합니다. 먼저 PyG의 to\_hetero() 함수를 사용하여 우리의 기본 GNN 동종 모델을 자동으로 이종 GNN 모델로 변환하며, 이 모델은 이후 집계 연산에 사용됩니다. 이 함수에는 두 가지 매개변수, 즉 기본 GNN 모델과 메타데이터 집합이 필요합니다. 기본 GNN 모델을 사용하는 방식에 대한 직관을 높이기 위해, 우리 SAGE 모델의 이종 버전에 대한 다음 구현을 살펴보겠습니다.

```python
Listing 12.25 Implementation of the HeteroSAGE model
class HeteroSAGE(HeteroBaseModel):
def init__(self, metadata, input_dim, hidden_dim, out_dim):
super(HeteroSAGE, self).__init__(
metadata,
input_dim,
hidden_dim,
out_dim,
SAGE
)
```

이 경우 HeteroSAGE 클래스의 초기화에는 금융 거래 맥락에서 동종 그래프를 처리하기 위해 채택한 SAGE 클래스를 인수로 필요로 합니다. 이 SAGE 클래스는 PyG 라이브러리가 제공하는 두 개의 SAGE-Conv 계층으로 구성된 GNN 모델임을 상기하십시오.

HeteroSAGE와 다른 모든 이종 모델의 구조는 각각에 대응하는 동종 버전에 기반하며, 이종 그래프의 각 엣지 유형에 적용됩니다. 우리의 시나리오에서는 작업과 데이터의 특성상 단일 엣지 유형인 ("user", "rates", "movie")만 존재합니다. 그러나 대부분의 복잡한 시나리오에는 여러 엣지 유형이 있으며, 우리는 각 엣지 유형에 어떤 컨볼루션 계층을 적용할지 결정할 수 있습니다.

PyG가 제공하는 HeteroData 객체의 metadata 메서드는 엣지 및 노드 유형의 집합을 정의하며, 이는 to\_hetero() 함수에 전달되는 두 번째 인수입니다. 따라서 우리는 이종 GNN 모델에 이웃의 특징 집계를 적용하는 데 어떤 엣지 유형이 사용되는지 알려줄 수 있습니다. 다시 말해, 컨볼루션 연산은 메타데이터에 지정된 노드와 엣지에 의해 구동됩니다.

#### 디코더


디코더 구성요소는 GNN 인코더에서 도출된 사용자 임베딩과 영화 임베딩 사이의 내적을 계산하여 사용자와 영화 간의 적합성을 판단합니다. 내적은 각각 학습된 특징 표현을 기반으로 이러한 적합성을 정량화합니다. 더 높은 내적 값은 특정 영화에 대한 잠재 사용자의 관심과 같이 상호작용 또는 평점이 발생할 가능성이 더 강함을 나타냅니다.

이 디코딩 함수는 PyTorch의 F.binary\_cross\_entropy\_with\_logits 함수와 함께 사용되며, 이 함수는 시그모이드 활성화와 이진 교차 엔트로피 손실을 통합합니다. 시그모이드 활성화는 내적 점수를 확률로 변환하여 링크 존재 가능성으로 해석할 수 있게 합니다. 그런 다음 이진 교차 엔트로피 손실은 이러한 예측 확률과 실제 상호작용 레이블 간의 차이를 측정하여, 추천 정확도를 향상하도록 모델의 최적화를 이끕니다.

#### 12.2.5 평가 및 분석


우리의 분석에서는 링크 예측을 위한 세 가지 종단 간 모델을 비교했습니다. 각 모델은 서로 다른 GNN 인코더인 GCN, GAT, SAGE를 사용합니다. 표 12.10은 각 종단 간 모델의 파라미터 수와 총 학습 시간을 초 단위로 보여 주며, 우리는 T4 Colab 머신에서 55 에포크 이후 이러한 결과를 얻었습니다.

참고 우리의 GCN 구현은 그래프 유형에 따라 서로 다른 PyG 연산자에 해당합니다. 동질 그래프의 경우, 우리는 차수 정규화를 갖는 표준 GCN 계층인 GCNConv를 사용했습니다. 이질 그래프의 경우, 우리는 HeteroConv 래퍼와 더 자연스럽게 통합되는 밀접히 관련된 변형인 GraphConv를 사용했습니다. GCNConv는 단일 노드 유형과 단일 엣지 유형을 가정하므로 이질 그래프에 직접 적용할 수 없는 반면, GraphConv는 이질적 설정에서 필요한 별도의 루트 및 이웃 변환을 지원합니다.

표 12.10 서로 다른 GNN 인코더를 사용하는 링크 예측 모델의 파라미터 수와 학습 시간
<table><tr><td>인코더</td><td>파라미터</td><td>학습 시간 (초)</td></tr><tr><td>GCN</td><td>713,408</td><td>826</td></tr><tr><td>GAT</td><td>1,066,880</td><td>956</td></tr><tr><td>SAGE</td><td>713,408</td><td>777</td></tr></table>

결과는 SAGE 모델이 가장 효율적이고, GAT 모델이 가장 비효율적이며, GCN 모델은 나머지 두 모델 사이에 위치함을 나타냅니다. 모델 파라미터 수는 노드 분류에 사용된 모델들보다 상당히 많습니다. 이는 사용자와 영화 특징의 표현력을 향상하기 위해 임베딩 계층을 추가한 것과, 데이터를 처리하기 위해 이질 GNN 모델을 사용한 것 등 여러 이유 때문입니다. 짐작할 수 있듯이, 파라미터 수는 학습 시간에 직접적인 영향을 미치며, 동일한 인프라에서 노드 분류를 수행할 때보다 훨씬 더 깁니다.

#### 정밀도, 재현율, F1-점수


영화 추천을 위한 링크 예측 과제에서 GCN, GAT, SAGE 모델의 성능을 평가하기 위해, 노드 분류 과제에서 채택했던 것과 동일한 지표를 사용하여 이들의 동작을 분석할 수 있습니다.

정밀도—모델이 사용자와 영화 사이의 링크를 예측할 때(예: 영화를 추천할 때), 그 예측은 얼마나 자주 정확합니까?

재현율—모든 사용자–영화 링크(예: 사용자가 좋아할 수 있는 영화) 중에서 모델은 얼마나 많은 링크를 성공적으로 식별합니까?

F1-점수—모델은 정밀도(추천의 정확성)와 재현율(가능한 한 많은 관련 추천을 찾는 능력) 사이의 균형을 얼마나 효과적으로 맞춥니까?

그림 12.10은 모델 학습 중 검증 데이터셋에서 이러한 지표들의 추세를 보여줍니다.

![](images/ko/figure-12-10a-ko.png)

![](images/ko/figure-12-10b-ko.png)

![](images/ko/figure-12-10c-ko.png)  
그림 12.10 학습 중 검증 데이터셋에서 GNN 모델의 정밀도, 재현율, F1-점수 추세

추천을 위한 링크 예측의 맥락에서 앞서 제기한 질문들에 답해 보겠습니다.

정밀도—모델들이 사용자가 영화에 평점을 매길지 여부(링크가 존재하는지)를 예측할 때, SAGE 모델이 가장 높은 정밀도를 보입니다. SAGE는 거짓 양성을 최소화하는 데 가장 신뢰할 수 있으며, 사용자가 영화에 평점을 매겨 상호작용할지 여부를 효과적으로 예측합니다. 영화 추천 과제에서 이는 사용자가 제안된 영화에 평점을 매길 가능성이 낮은, 관련성이 낮은 추천이 더 적다는 것을 의미합니다. GCN 모델은 약간 낮은 성능을 보이지만 높은 정밀도를 유지하므로, 신뢰성이 필수적인 경우 좋은 대안이 됩니다. GAT 모델은 가장 낮은 정밀도를 보이며, 에포크 전반에 걸쳐 더 큰 변동성을 나타내는데, 이는 사용자 참여에 대해 더 많은 잘못된 예측을 한다는 것(즉, 사용자가 평점을 매길 가능성이 낮은 영화에 평점을 매길 것이라고 예측함)을 시사합니다.

재현율—실제 사용자–영화 링크(즉, 사용자가 평점을 매길 모든 영화)를 식별하는 데 있어, GCN 모델이 가장 높은 재현율을 달성합니다. 이는 GCN이 사용자가 상호작용할 수 있는 영화를 식별하는 데 가장 포괄적이며, 가장 많은 실제 평점을 성공적으로 포착한다는 것을 나타냅니다. SAGE 모델은 약간 더 낮은 재현율로 그 뒤를 따르며, 이는 GCN에 비해 사용자가 평점을 매길 영화를 약간 더 많이 놓친다는 의미입니다. 그러나 GAT 모델은 재현율에서 어려움을 겪어 실제 사용자–영화 링크의 상당 부분을 놓치며, 이는 포괄적인 추천을 제공하는 능력을 저하시킵니다.

F1-score—SAGE 모델은 가장 높은 F1-score를 달성하여 정확한 추천을 제공하는 것과 사용자 선호의 포괄성을 보장하는 것 사이에서 최상의 균형을 이룹니다. 이는 SAGE가 관련 없는 예측을 최소화하면서 사용자가 어떤 영화에 평점을 매길지 예측하는 데 특히 효과적이게 합니다. GCN 모델은 높은 재현율의 이점을 바탕으로 우수한 성능을 보이지만, 정밀도가 약간 낮아 전반적인 균형이 약화됩니다. GAT 모델은 정밀도와 재현율이 낮기 때문에 경쟁력 있는 F1-score를 달성하는 데 어려움을 겪으며, 사용자–영화 상호작용의 전체 범위를 포착하는 데 신뢰성이 떨어집니다.

#### 혼동 행렬


그림 12.11은 테스트 데이터셋에서 GCN, GAT, SAGE 모델의 혼동 행렬(confusion matrix)을 보여 주며, 각 모델이 사용자 평점을 얼마나 효과적으로 예측하는지에 대한 상세한 통찰을 제공합니다. SAGE는 존재하지 않는 링크를 식별하는 데 강한 성능을 보이며, 사용자가 평점을 매기지 않을 영화의 94.6%를 올바르게 예측합니다(진음성: 19,084). 그러나 SAGE는 사용자가 실제로 평점을 매길 영화의 71.5%를 식별하는 반면(진양성: 7,211), 실제 평점의 28.5%를 놓칩니다(위음성: 2,872). 이는 SAGE가 사용자가 평점을 매길 가능성이 낮은 영화를 걸러내는 데 효과적이지만, 때때로 잠재적 평점을 간과한다는 것을 나타냅니다. 특히 SAGE는 위양성(1,082, 즉 5.4%)을 최소화하는데, 이는 사용자가 참여할 가능성이 낮은 영화를 거의 추천하지 않음을 의미하며, 매우 정밀한 평점 예측을 보장합니다.

![](images/ko/figure-12-11a-ko.png)

![](images/ko/figure-12-11b-ko.png)

![](images/ko/figure-12-11c-ko.png)  
그림 12.11 테스트 데이터셋에서 GCN, GAT, SAGE 모델의 혼동 행렬

GCN 모델은 사용자가 평점을 매길 영화와 매기지 않을 영화를 식별하는 데 강한 균형을 제공합니다. 이 모델은 사용자가 평점을 매기지 않을 영화의 91.7%를 성공적으로 식별하고(진음성: 18,488), 실제 평점의 78.6%를 포착합니다(진양성: 7,924). GCN은 SAGE보다 놓친 평점의 비율이 더 낮지만(위음성: 2,159), 위양성 비율은 약간 더 높습니다(1,678, 즉 8.3%). 이는 GCN이 잠재적 평점을 찾는 데 더 우수하지만, 때때로 사용자가 평점을 매기지 않을 수 있는 영화를 제안한다는 것을 의미합니다.

GAT 모델은 사용자가 평점을 매기지 않을 영화의 88.3%를 올바르게 예측하고(진음성: 17,808), 실제 평점의 75.4%를 식별합니다(진양성: 7,607). 그러나 위음성 수(2,476, 즉 24.6%)가 GCN보다 높아, 사용자가 평점을 매길 영화를 더 많이 놓친다는 것을 의미합니다. 또한 위양성 수(2,358, 즉 11.7%)가 가장 높아, 사용자가 참여할 가능성이 낮은 영화를 추천할 가능성이 더 크다는 것을 나타냅니다.

이러한 결과는 세 모델 모두 사용자 평점을 예측하는 데 각기 다른 정도로 좋은 성능을 보이지만, SAGE는 정밀도 측면에서 두드러져 사용자가 관련 없는 영화 제안을 받을 가능성을 줄인다는 것을 보여 줍니다. GCN은 잠재적 평점을 포착하는 것과 관련 없는 추천을 피하는 것 사이에서 가장 좋은 균형을 제공합니다. GAT는 사용자가 평점을 매기지 않을 수 있는 영화를 과도하게 추천하는 경향이 있습니다.

#### 요약


그래프 신경망은 노드 분류 (node classification)와 링크 예측 (link prediction) 같은 그래프 기반 ML 과제의 근본적인 문제를 해결합니다.

과제 도메인의 차이에도 불구하고, 우리는 여러 단계로 구성된 일반적인 인코더–디코더 (encoder–decoder) 프레임워크를 정의할 수 있으며, 이 프레임워크는 그래프 구조 데이터를 추론에 적합한 모델로 처리합니다. 이 프레임워크에서 인코더는 그래프 합성곱 네트워크 (graph convolutional network, GCN), 그래프 어텐션 네트워크 (graph attention network, GAT), 또는 GraphSAGE (SAGE)와 같은 GNN 아키텍처입니다. 디코더는 학습된 표현에 과제별 함수를 적용합니다.

그래프 데이터의 복잡한 관계를 포착하는 GNN의 능력은 사기 탐지와 추천 시스템 같은 실제 문제에서 매우 가치가 높으며, 다양한 도메인 전반에서 폭넓게 활용될 수 있게 합니다.
