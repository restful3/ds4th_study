---
lang: ko
format:
  html:
    toc: true
    embed-resources: true
    theme: cosmo
---

# 그래프 특징 공학: 수동 및 반자동 접근법

### 이 장에서 다루는 내용


그래프의 노드와 관계에 대한 수동 특징 공학 기법

그래프 표현에서 도메인 전문성과 반자동 추출을 결합하는 방법

특징 공학의 실제 응용

그래프에서 머신러닝(ML)의 성공은 근본적인 과제, 즉 그래프 요소(노드, 관계, 전체 그래프)를 ML 알고리즘이 처리할 수 있는 벡터로 어떻게 효과적으로 표현할 것인가에 달려 있습니다. 흔히 벡터화(vectorization) 또는 특징화(featurization)라고 불리는 이 표현 단계는 모델이 얼마나 잘 학습하고 예측할 수 있는지를 결정합니다.

로지스틱 회귀와 랜덤 포레스트 같은 전통적 접근법부터 정교한 딥러닝 모델에 이르기까지 현대 ML 알고리즘은 잘 확립되어 있지만, 그래프 구조를 직접 처리할 수는 없습니다. 대신 수치 입력 벡터가 필요합니다. 이러한 벡터의 품질은 노드 분류, 관계 예측, 전체 그래프 분석 등 어떤 다운스트림 작업이든 그 성능에 직접적인 영향을 미칩니다.

이 장에서는 이러한 벡터 표현을 만드는 기술과 과학을 탐구하며, 수동 접근법에서 자동화 접근법으로 나아갑니다. 우리는 먼저 수동 특징 공학에서 출발하여, 도메인 지식과 그래프 속성에 기반한 해석 가능한 특징을 설계합니다. 이 직접적인 접근법은 시간이 많이 들지만, 무엇이 효과적인 표현을 만드는지에 대한 통찰을 제공하고 특정 특징이 다른 특징보다 더 잘 작동하는 이유를 이해하는 데 도움을 줍니다.

이후 점차 더 자동화된 기법을 소개합니다. 그러나 특징 추출이 더 자동화될수록 특징은 해석하기 어려워지는 경향이 있습니다. 이는 다음과 같은 스펙트럼을 형성합니다.

수동 특징은 해석 가능성이 높지만 만드는 데 노동집약적입니다(9장에서 간단히 소개했으며, 이 장에서 심층적으로 논의합니다).

반자동 특징은 해석 가능성과 효율성 사이의 균형을 이룹니다(이 장에서도 다룹니다).

완전 자동 특징은 생성하기 효율적이지만 해석하기 더 어렵습니다(11장과 12장에서 논의합니다).

전통적인 데이터셋에서 특징은 실제 세계에서 직접 측정한 값입니다. 예를 들어 날씨 예측 데이터셋에는 강수량, 온도 범위, 풍속과 같은 측정 가능한 속성이 포함될 수 있습니다. 모델 학습 중에는 특징(측정된 속성)과 레이블(실제 날씨)을 모두 알고 있는 행을 사용합니다. 예측 시에는 특징만 있고 날씨 레이블을 결정해야 하는 새로운 데이터에 학습된 모델을 적용합니다.

하지만 그래프 기반 ML에서는 그래프 구조 자체에서 특징을 구성해야 합니다. 물리적 측정값 대신, 노드, 관계 또는 전체 그래프의 의미 있는 속성을 수치 값으로 포착해야 합니다.

이러한 그래프 기반 특징을 만드는 데에는 두 가지 근본적인 접근법이 있습니다:

 이전 장에서 소개한 특징 공학은 그래프 속성과 도메인 지식을 바탕으로 수동으로 설계된 특징에 의존합니다. 이러한 특징은 해석 가능성이 매우 높지만 생성하는 데 시간이 많이 걸리며, 복잡한 작업에 필요한 모든 관련 패턴을 포착하지 못할 수 있습니다. 그래프 기반 특징의 일반적인 예로는 노드 차수, 클러스터링 계수, 중심성 척도가 있습니다.

 이와 달리 표현 학습 (representation learning)은 그래프 구조로부터 특징 표현을 자동으로 학습합니다. 이 접근법은 인간의 개입을 최소화하며, 학습을 통해 특정 작업에 적응할 수 있습니다. 수동 공학보다 복잡한 패턴을 더 효과적으로 포착하는 경우가 많지만, 일반적으로 해석하기 더 어려운 특징을 생성합니다. 이 접근법은 다음 두 장에서 다룹니다.

특징 공학의 과제와 한계를 이해하면 표현 학습이 왜 점점 더 중요해졌는지 이해하는 데 도움이 됩니다. 수동 특징 공학은 두 가지 핵심 이유로 여전히 가치가 있습니다. 첫째, 인간이 이해하고 검증할 수 있는 해석 가능한 특징을 생성합니다. 둘째, 그래프 표현을 효과적으로 만드는 요인에 대한 통찰을 제공하여 자동화된 접근법의 설계를 뒷받침합니다.

수동으로 추출한 특징의 또 다른 중요한 장점은 그래프에 대한 자율적 추론을 수행하는 대규모 언어 모델 (LLMs)과의 호환성입니다. 이러한 특징은 잘 이해된 그래프 알고리즘과 속성에 기반하므로, LLM은 이를 효과적으로 해석하고 추론할 수 있습니다.

이 장에서는 실제 컨설팅 프로젝트에서 도출한 세 가지 실용적인 특징 공학 접근법을 살펴봅니다. 이러한 예시는 그래프 기반 ML에서 수동 특징 추출의 강점과 한계를 모두 보여줍니다.

### 10.1 수동 노드 특징


사람들의 네트워크가 있다고 가정해 보겠습니다. 즉, 사람을 나타내는 노드와 이 개인들 사이의 모든 연결을 나타내는 관계로 구성된 그래프입니다(그림 10.1 참조). 이 사람들 가운데 일부는 알려진 사기범입니다. 우리는 이들의 전체 목록을 가지고 있지 않으므로, 우리의 작업은 노드를 분류하여 아직 식별되지 않은 사기범을 알아내거나 사람들이 사기 행위의 피해자가 될 위험을 판단하는 것입니다.

![](images/11ac0e57b4f8cc1d823e70b008dad19f041425bfbced127b80ec2541fa21d035.jpg)

이 네트워크에는 두 가지 유형의 노드가 포함됩니다. 검은색 노드는 알려진 사기범(노드 D, E, F, I)을 나타내고, 흰색 노드는 정상 사용자 또는 아직 식별되지 않은 노드를 나타냅니다. 노드 사이의 모든 연결은 무방향 엣지로 표현됩니다. 다음 목록은 Python과 NetworkX를 사용하여 이 네트워크를 생성하는 방법을 보여줍니다.

그림 10.1 흰색 노드는 정상 개인 또는 사기범 여부가 알려지지 않은 개인을 나타내고, 검은색 노드는 알려진 사기범을 나타내는 예시 소셜 네트워크  
```python
import networkx as nx
import matplotlib.pyplot as plt
def create_fraud_network(): Initializes the empty
G = nx.Graph() < undirected graph
Fraudulent nodes
fraudsters = ['D', 'E', 'F', 'I'] < (black in figure 10.1)
#### Add all nodes first
nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
Defines all
nodes in the
G.add_nodes_from(nodes) Sets the fraudster attribute for network
each node, based on the list
for node in G.nodes():
G.nodes[node]['is_fraudster'] = node in fraudsters
```

```python
edges = [
('A', 'B'), ('A', 'G'), ('A', 'H'), ('A', 'I'), ('A', 'O'),
('A', 'T'), ('B', 'D'), ('B', 'C'), ('D', 'E'), ('D', 'F'),
('D', 'G'), ('E', 'F'), ('F', 'G'), ('G', 'I'), ('H', 'K'),
('I', 'K'), ('I', 'N'), ('K', 'J'), ('L', 'M'), ('L', 'N'),
('N', 'M'), ('O', 'P'), ('O', 'Q'), ('Q', 'R'), ('Q', 'S')
] < Defines all edges
in the network
G.add_edges_from(edges)
Returns the
return G < complete graph
```

다음 호출을 사용하면 이 장에서 수행하는 이후의 모든 분석에 이 네트워크를 사용할 수 있습니다.

목록 10.2 사용 예   
G = create\_fraud\_network() < 이 그래프 객체는   
우리의 모든 메트릭 계산에 사용할 수 있습니다.   
degree\_metrics = compute\_degree\_metrics(G)   
뒤에서 정의할 예시 메트릭   
triangle\_metrics = compute\_triangle\_metrics(G)

우리의 목표는 주어진 노드가 사기범인지 여부를 말해 주는 분류기를 구현하는 것입니다. 일반적으로 이 작업은 로지스틱 회귀, 베이지안 분류기, 의사결정나무, 랜덤 포레스트 등 잘 알려진 고전적 분류 알고리즘을 사용하여 수행됩니다. 이들은 모두 지도 알고리즘이므로, 학습 중 입력은 각 데이터 포인트(이 경우 노드)와 관련된 특징 집합 및 레이블(여기서는 사기범/비사기범)입니다. 학습을 통해 ML 모델이 구축된 후, 분류기는 일정한 가능도로 레이블을 할당할 수 있게 됩니다.

이러한 알고리즘을 사용하려면 그림 10.2에 나타낸 것처럼 각 노드를 학습과 예측 중 좋은 지표로 사용할 수 있는 특징 집합으로 “표현”해야 합니다. 각 노드에 대해 많은 흥미로운 정보를 추출할 수 있습니다.

국소 특징 (local features)은 노드의 1-홉 이웃 또는 에고 중심 네트워크 (egonet), 즉 특정 노드와 그 즉시 이웃을 고려하여 추출할 수 있는 특징입니다. 에고넷의 중심은 에고 (ego)이고, 주변 노드는 타자 (alters)입니다. 이러한 국소 특징은 에고 노드 주변의 n차 이웃, 즉 해당 노드로부터 n홉 떨어진 노드를 고려할 수도 있습니다.

 전역 특징은 전체 네트워크 또는 그 상당 부분에서 각 노드의 역할을 측정합니다(에고넷이나 n차 이웃이 아닙니다). 이 범주에는 매개 중심성 (betweenness centrality), 근접 중심성 (closeness centrality), PageRank, 고유벡터 중심성 (Eigenvector centrality)과 같은 중심성 지표가 포함됩니다. 이러한 척도는 네트워크에서 노드의 영향력과 해당 노드가 다른 노드들에 의해 어떻게 영향을 받을 수 있는지를 포착합니다. (이러한 중심성 알고리즘에 익숙하지 않다면, 참고 자료로 우리의 이전 저서인 Graph-Powered Machine Learning [1] 또는 [2]를 권장합니다.)

우리는 각 노드를 나타내는 특징을 식별하기 위해 지표를 사용할 것입니다. 또한 분류의 최종 품질을 개선하기 위해 일부 지표를 맞춤화하고, 특징화 (featurization) 과정이 우리의 필요에 맞게 조정될 수 있음을 보여줄 것입니다. 우리의 접근법은 즉각적인 이웃에 기반한 특징에서 시작하여 네트워크 전반의 패턴을 검토하기 전에, 지역 지표에서 전역 지표로 진행됩니다. 각 경우에 지표와 그 중요성을 정의하고, 자동 추출을 위한 코드를 제시하며, 결과를 표로 표시할 것입니다.

![](images/4bbbe8941168f2152f6700022c402cc1c231d7d0896de6ea4d5be48c08cfab49.jpg)  
노드를 특징 벡터로 변환하며, 각 특징은 지표와 알고리즘을 통해 특정 특성을 포착합니다. 그 결과 벡터는 원래 노드의 핵심 속성을 보존하는 수치적 표현으로 기능합니다.  
그림 10.2 노드 특징 추출: 지표와 그래프 알고리즘을 사용하여 노드를 주요 특성을 포착하는 수치적 특징 벡터로 변환하기

우리는 이러한 특징을 점진적으로 구축하면서, 각각의 새로운 지표가 노드 표현에 어떻게 하나의 차원을 추가하는지 보여줄 것입니다. 이러한 체계적인 접근법을 통해 네트워크 구조와 노드 행동의 점점 더 복잡한 패턴을 포착할 수 있습니다.

#### 10.1.1 차수


노드의 차수는 해당 노드가 몇 개의 이웃을 가지는지를 나타냅니다. 우리의 예시 사례에서는 사기성 직접 이웃과 정상 직접 이웃의 수를 구분하고자 합니다. 이를 사기성 차수와 정상 차수라고 부릅니다(간단히 사기 차수와 정상 차수로 줄여 부릅니다). 이 두 지표는 전역 차수와 함께 노드의 직접 연결을 더 잘 표현합니다. 예를 들어, 나에게 10개의 직접 이웃이 있고 그들이 모두 사기성이라면, 모든 이웃이 정상인 경우보다 내가 사기 행위자일 가능성이 더 높습니다. 다음 목록은 일반적인 그래프에서 이러한 값을 계산합니다.

각 노드의 차수 지표를 저장할 딕셔너리를 초기화합니다.

```python
for node in G.nodes():
Calculates the total number
neighbors = list(G.neighbors(node))
total_degree = len(neighbors) < of neighbors (total degree)
fraud_degree = sum(1 for neighbor in neighbors
if G.nodes[neighbor].get('is_fraudster', False))
legit_degree = total_degree - fraud_degree < Counts neighbors
marked as fraudsters
degree_metrics[node] = { using the node
'total_degree': total_degree, attribute is_fraudster
'fraud_degree': fraud_degree,
'legit_degree': legit_degree Calculates the legit
} degree as total minus
fraudulent neighbors
return degree_metrics < Returns a dictionary
containing all degree
def get_node_degrees(G, node): metrics for each node
metrics = compute_degree_metrics(G)
return metrics.get(node, {
'total_degree': 0,
'fraud_degree': 0,
'legit_degree': 0 Gets degree metrics
}) < for a specific node
```

이 코드는 그래프의 노드들이 사기성 노드를 식별하기 위해 'is\_fraudster' 불리언 속성으로 표시되어 있다고 가정합니다. 표 10.1은 관련 값을 포함합니다. 예를 들어, 노드 D는 총 네 개의 직접 이웃을 가지며, 그중 두 개는 사기성이고 두 개는 정상입니다.

표 10.1 그림 10.1의 사기 그래프에서 전체 차수, 사기 차수, 정상 차수
<table><tr><td rowspan=1 colspan=1>노드</td><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>B</td><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>F</td><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>H</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>전체 차수사기 차수정상 차수</td><td rowspan=1 colspan=1>615</td><td rowspan=1 colspan=1>312</td><td rowspan=1 colspan=1>101</td><td rowspan=1 colspan=1>422</td><td rowspan=1 colspan=1>220</td><td rowspan=1 colspan=1>321</td><td rowspan=1 colspan=1>431</td><td rowspan=1 colspan=1>202</td><td rowspan=1 colspan=1>404</td><td rowspan=1 colspan=1>101</td></tr><tr><td rowspan=1 colspan=1>노드</td><td rowspan=1 colspan=1>K</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>M</td><td rowspan=1 colspan=1>N</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>P</td><td rowspan=1 colspan=1>Q</td><td rowspan=1 colspan=1>R</td><td rowspan=1 colspan=1>S</td><td rowspan=1 colspan=1>T</td></tr><tr><td rowspan=3 colspan=1>전체 차수사기 차수정상 차수</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>3</td><td rowspan=2 colspan=1>10</td><td rowspan=2 colspan=1>10</td><td rowspan=3 colspan=1>101</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td></tr></table>

#### 연습문제

값을 직접 손으로 계산하고, 개념이 명확한지 확인하기 위해 이를 검토하십시오. 이들은 계산하기 매우 간단한 척도입니다. 나중에 다른 척도들은 코드로 실행해야 할 것입니다.

#### 10.1.2 삼각형


그래프 이론에서 삼각형은 모두 서로 연결된 세 개의 노드로 구성된 부분그래프입니다. 따라서 세 개의 노드 A, B, C가 있을 때, A-B, A-C, B-C 쌍 사이에 관계가 존재하면 이들은 삼각형을 형성합니다(그림 10.3 참조).

![](images/28a245a464648a6959d1a9d081f0aba6c739e69f82d6f5714b654ed3d22d2315.jpg)  
그림 10.3 연결된 세 노드의 예: 각 노드가 다른 두 노드와 연결되어 있으면 이들은 삼각형을 구성합니다.

한 노드의 에고넷 (egonet)에 삼각형이 존재한다는 것은 대상 노드가 이웃들과 강한 연결을 가지고 있음을 나타냅니다. 여러분과 가까운 사람들을 생각해 보십시오. 여러분의 친구들은 아마 서로도 친구일 것입니다. 이것이 바로 삼각형이 긴밀히 연결된 개인 집단의 영향력 효과를 드러내는 이유입니다. 하나의 노드와 두 개의 타자 (alter)가 삼각형을 형성하는 경우를 생각해 보십시오. 우리의 예에서 두 타자가 모두 사기적이라면, 해당 삼각형은 사기적이라고 결론 내릴 수 있으며, 그 반대도 마찬가지입니다. 타자 하나만 사기적이라면, 그 삼각형은 반사기적이라고 합니다.

다음 코드는 우리 그래프에서 전체 삼각형 수와 사기적, 정상, 반사기적 삼각형을 계산합니다.

```python
Listing 10.4 Computing triangle metrics in a fraud detection network
import networkx as nx
Initializes a dictionary
def compute_triangle_metrics(G): to store triangle
triangle_metrics = {} < metrics for each node
for node in G.nodes():
triangles = []
neighbors = list(G.neighbors(node)) Finds triangles by
checking whether
for i in range(len(neighbors)): two neighbors of
for j in range(i + 1, len(neighbors)): the target node
if G.has_edge(neighbors[i], neighbors[j]): are connected
triangles.append((neighbors[i], neighbors[j]))
```

total\_triangles = len(triangles)   
fraud\_triangles = 0   
카운트를 초기화합니다   
legit\_triangles = 0   
semi\_fraud\_triangles = 0 각 삼각형을   
다른 두 노드의   
사기 여부에 따라 분류합니다   
for n1, n2 in triangles: <   
n1\_fraud = G.nodes[n1].get('is\_fraudster', False)   
n2\_fraud = G.nodes[n2].get('is\_fraudster', False)   
다른 두 노드가 모두 사기범이면   
if n1\_fraud and n2\_fraud: <   
해당 삼각형을 사기적으로 계산합니다   
fraud\_triangles += 1   
elif not n1\_fraud and not n2\_fraud: <   
다른 두 노드가 모두 정상이면   
legit\_triangles += 1 해당 삼각형을 정상으로 계산합니다   
else: < 다른 노드 하나는 사기적이고   
semi\_fraud\_triangles += 1 하나는 정상인 경우 해당 삼각형을   
반사기적으로 계산합니다   
triangle\_metrics[node] = {   
'total\_triangles': total\_triangles,   
'fraud\_triangles': fraud\_triangles,   
'legit\_triangles': legit\_triangles,   
'semi\_fraud\_triangles': semi\_fraud\_triangles   
}   
각 노드에 대한 모든 삼각형 척도를   
return triangle\_metrics <   
포함하는 딕셔너리를 반환합니다   
def get\_node\_triangles(G, node):   
metrics = compute\_triangle\_metrics(G)   
return metrics.get(node, {   
'total\_triangles': 0,   
'fraud\_triangles': 0,   
'legit\_triangles': 0,   
'semi\_fraud\_triangles': 0 특정 노드에 대한   
}) < 삼각형 척도를 가져옵니다  
표 10.2에는 우리 사기 그래프의 값이 포함되어 있습니다.

표 10.2 그림 10.1의 사기 그래프에 대한 삼각형 척도
<table><tr><td rowspan=1 colspan=1>노드</td><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>B</td><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>F</td><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>H</td><td rowspan=1 colspan=1>I</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=4 colspan=1>전체 삼각형사기 삼각형정상 삼각형준사기 삼각형</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=4 colspan=1>0000</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=3 colspan=1>101</td><td rowspan=3 colspan=1>100</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=2 colspan=1>00</td><td rowspan=1 colspan=1>0</td><td rowspan=2 colspan=1>01</td><td rowspan=2 colspan=1>00</td><td rowspan=2 colspan=1>10</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>노드</td><td rowspan=1 colspan=1>K</td><td rowspan=1 colspan=1>L</td><td rowspan=1 colspan=1>M</td><td rowspan=1 colspan=1>N</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>P</td><td rowspan=1 colspan=1>Q</td><td rowspan=1 colspan=1>R</td><td rowspan=1 colspan=1>S</td><td rowspan=1 colspan=1>T</td></tr><tr><td rowspan=1 colspan=1>전체 삼각형</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=2 colspan=1>1O</td><td rowspan=2 colspan=1>00</td><td rowspan=2 colspan=1>00</td><td rowspan=2 colspan=1>00</td><td rowspan=2 colspan=1>00</td><td rowspan=2 colspan=1>00</td><td rowspan=2 colspan=1>00</td></tr><tr><td rowspan=1 colspan=1>사기 삼각형</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>o</td><td rowspan=1 colspan=1>o</td></tr><tr><td rowspan=1 colspan=1>정상 삼각형</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>준사기 삼각형</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr></table>

#### 연습문제


다시 한번, 값을 직접 계산하고, 개념이 확실한지 확인하십시오.

#### 10.1.3 밀도


밀도 (density)는 노드들이 서로에게 어떻게 영향을 미칠 수 있는지를 나타내는 또 다른 척도입니다. 이는 그래프의 노드들이 연결되어 있는 정도를 측정합니다. N개의 노드로 이루어진 완전 연결 그래프가 주어졌을 때, 다음 공식은 가능한 간선의 총수를 계산합니다.

$$
{\binom{N}{2}} = {\frac{N(N-1)}{2}}
$$

이 경우 각 노드는 네트워크의 다른 모든 노드와 연결됩니다. 밀도는 이러한 가능한 간선 중 실제 그래프에서 관측되는 간선의 비율을 측정합니다. 따라서 M이 그래프의 간선 수라면, 전체 네트워크의 밀도는 다음 공식으로 계산할 수 있습니다.

$$
d = {\frac{M}{\binom{N}{2}}} = {\frac{2M}{N(N-1)}}
$$

각 노드에 대해서도 그 에고넷 (egonet)의 밀도를 고려하여 밀도를 계산할 수 있습니다. 예를 들어 노드 A가 대상 노드라고 가정해 보겠습니다. 이 노드의 에고넷에는 7개의 노드가 있으므로, 에고넷에서 가능한 간선의 총수는 $7(7-1)/2=21$입니다. 관측된 간선의 수는 7개이며, $d=7/21 \sim 0.33$입니다. 이 예에서는 계산이 단순하지만, 다음 목록에는 상황이 더 어려워질 때 사용할 코드가 포함되어 있습니다. 이 척도는 그 특성상 우리가 고려하는 사기 도메인과 관련된 특정 값을 포함하지 않습니다(사기 밀도나 정상 밀도는 존재하지 않습니다).

![](images/d95463828199b2a6ca575f2cf8d234c650adb11196222df9d4eb05e9a50a1fb8.jpg)

![](images/db4104162aca5068f22fa07721b45c3c2e0b3d8e6a1bbc8370a131d6e34a44c6.jpg)  
우리의 예제 그래프에서 이 목록을 실행하면 표 10.3의 값이 얻어집니다.

표 10.3 그림 10.1의 사기 그래프에 대한 밀도 측정값
<table><tr><td rowspan=1 colspan=1>노드</td><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>B</td><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>F</td><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>H</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>밀도</td><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>노드</td><td rowspan=1 colspan=1>K</td><td rowspan=1 colspan=1>L</td><td rowspan=1 colspan=1>M</td><td rowspan=1 colspan=1>N</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>P</td><td rowspan=1 colspan=1>Q</td><td rowspan=1 colspan=1>R</td><td rowspan=1 colspan=1>S</td><td rowspan=1 colspan=1>T</td></tr><tr><td rowspan=1 colspan=1>밀도</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td></tr></table>

지금까지 살펴본 측정값들은 값을 계산하기 위해 한 노드의 에고넷 (egonet)을 사용합니다. 이어지는 측정값들은 계산 중 전체 네트워크를 고려합니다.

#### 10.1.4 측지 경로(또는 최단 경로)


측지 경로 또는 최단 경로는 두 노드 사이의 최소 거리를 나타냅니다. 우리는 이 측정값을 사용하고 이를 도메인의 특정 요구에 맞게 조정하여 다운스트림 알고리즘의 입력이 될 수 있는 특성을 식별할 수 있습니다.

우리의 예에서는 사기 노드와 정상 노드 사이에 경로가 있는지, 그 길이가 얼마나 되는지, 그리고 네트워크에 이러한 경로가 몇 개 포함되어 있는지를 알고자 합니다. 두 노드 사이에 더 많은 경로가 존재하고 그 경로들이 짧을수록 사기 행위가 대상 노드에 영향을 미칠 가능성이 더 높습니다. 이러한 고려를 바탕으로 우리는 사기 노드까지의 최단 경로(측지 경로)를 사용하기로 결정합니다. 또한 특정 거리에서 한 노드를 둘러싼 사기 노드의 수를 알아야 하므로, 임의의 사기 노드까지 1, 2, 또는 3홉인 경로의 수를 고려합니다.

이번에는 코드를 보여주기 전에, 수작업으로 계산할 수 있는 몇 가지 예부터 시작하겠습니다. 노드 A는 단일 홉(직접 연결)을 통해 노드 I와 연결되어 있습니다. 따라서 측지 거리는 1입니다. 다른 사기 노드와의 다른 직접 연결은 없습니다. 즉, 1홉 경로의 수는 하나입니다. 2홉 경로는 A-G-I, A-B-D, A-G-D, A-G-F의 네 개가 있으며, 이런 식으로 계속됩니다. 표 10.4를 참조하십시오.

표 10.4 그림 10.1의 사기 그래프에서의 측지 경로 (geodesic paths)
<table><tr><td rowspan=1 colspan=1>노드</td><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>B</td><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>F</td><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>H</td><td rowspan=1 colspan=1>—</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>측지 경로</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=3 colspan=1>2o1</td></tr><tr><td rowspan=1 colspan=1>1-홉 경로</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0</td><td rowspan=2 colspan=1>06</td></tr><tr><td rowspan=1 colspan=1>2-홉 경로</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>3-홉 경로</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>노드</td><td rowspan=1 colspan=1>K</td><td rowspan=1 colspan=1>L</td><td rowspan=1 colspan=1>M</td><td rowspan=1 colspan=1>N</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>P</td><td rowspan=1 colspan=1>Q</td><td rowspan=1 colspan=1>R</td><td rowspan=1 colspan=1>S</td><td rowspan=1 colspan=1>T</td></tr><tr><td rowspan=1 colspan=1>측지 경로</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>1-홉 경로</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>o</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>o</td><td rowspan=1 colspan=1>o</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>2-홉 경로</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>3-홉 경로</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>4</td></tr></table>

우리 네트워크는 단순하므로 3-홉 경로까지 계산하는 것도 복잡하지 않습니다. 그러나 실제 네트워크에서는 수작업 계산이 불가능하며 실용적이지 않습니다.

측지 경로 (geodesic paths)를 계산하는 것은 계산 비용이 많이 듭니다. 다양한 알고리즘을 사용할 수 있으며, 우리의 경우에 가장 적합한 알고리즘 중 하나는 다익스트라 알고리즘 (Dijkstra’s)입니다.

참고 다익스트라 알고리즘은 가장 작은 잠정 거리를 가진 미방문 노드를 반복적으로 선택하고, 해당 노드를 통해 각 미방문 이웃까지의 거리를 계산한 뒤, 그 노드를 방문한 것으로 표시함으로써 그래프에서 노드 간 최단 경로를 찾습니다. 이 알고리즘은 한 번에 하나의 정점씩 최단 경로 트리를 효율적으로 구축합니다 [3].

리스팅 10.6에는 측지 거리와 관련된 특징을 자동으로 추출하는 코드가 포함되어 있습니다. 이 코드는 단일 소스에서 출발하는 다익스트라 알고리즘 구현을 갖춘 networkx 라이브러리를 사용합니다. 현재 GDS 라이브러리(https:// github.com/neo4j/graph-data-science)의 Neo4j에도 몇 가지 다익스트라 구현이 있으며, 그중 하나는 소스 노드와 해당 노드에서 도달 가능한 모든 노드 사이의 최단 경로를 계산합니다.

#### 목록 10.6 사기 탐지 네트워크에서 측지 경로 지표 계산


```python
import networkx as nx
from collections import defaultdict
Initializes a dictionary
def compute_geodesic_metrics(G, max_hops=3): to store path metrics
path_metrics = {} < for each node
fraudster_nodes = [n for n, attr in G.nodes(data=True)
if attr.get('is_fraudster', False)]
Identifies all
fraudulent nodes
for node in G.nodes(): in the graph
if G.nodes[node].get('is_fraudster', False):
```

```python
geodesic_path = 0 < If the node is fraudulent, the distance
hop_counts = defaultdict(int) to the nearest fraudster is 0.
else:
paths_to_fraudsters = [] < For nonfraudulent nodes, calculates
hop_counts = defaultdict(int) paths to all fraudsters
for fraudster in fraudster_nodes:
try:
Uses Dijkstra's algorithm (via > path = nx.shortest_path(G, node, fraudster)
networkx) to find the shortest path_length = len(path) - 1
path to each fraudster paths_to_fraudsters.append(path_length)
if path_length <= max_hops:
hop_counts[path_length] += 1 Counts the number of
except nx.NetworkXNoPath: paths for each hop
continue distance up to max_hops
geodesic_path = min(paths_to_fraudsters)
if paths_to_fraudsters else float('inf') Finds the shortest path
length to any fraudster
path_metrics[node] = {
'geodesic_path': geodesic_path,
'#1-hop_paths': hop_counts[1],
'#2-hop_paths': hop_counts[2], Stores metrics, including
'#3-hop_paths': hop_counts[3] shortest path and count of
} < paths at each hop distance
return path_metrics < Returns a dictionary containing
all path metrics for each node
def get_node_paths(G, node):
metrics = compute_geodesic_metrics(G)
return metrics.get(node, {
'geodesic_path': float('inf'),
'#1-hop_paths': 0,
'#2-hop_paths': 0,
'#3-hop_paths': 0 Gets path metrics
}) < for a specific node
```

이 구현은 시작 노드에서 사기 행위자 노드까지의 측지 거리만, 그리고 사전에 정의된 거리까지만 계산합니다.

#### 10.1.5 근접성


근접 중심성 (closeness centrality)은 한 노드가 다른 모든 노드에 얼마나 “가까운지”를 나타냅니다. 이는 네트워크에서 한 노드로부터 다른 모든 노드까지의 평균 거리를 측정하며, 여기서 노드 간 거리는 두 노드 사이의 측지 경로 또는 최단 경로(이전 절에서 설명함)로 계산됩니다. 노드가 N개인 네트워크가 주어졌을 때, 노드 i에서 다른 노드들까지의 평균 측지 거리 또는 “원거리성(farness)”은 다음과 같이 계산됩니다.

$$
g(v_{i}) = \frac{\sum_{j=1}^{N} \mathbf{\Phi}_{(j \neq i)} d(v_{i}, \ v_{j})}{N - 1}
$$

각 부분을 살펴보겠습니다.

분자 $\sum_{j=1, (j \neq i)}^{N} d(v_{i}, v_{j})$는 모든 최단 경로 거리를 합산합니다.

$d(\boldsymbol{v}_{i}, \boldsymbol{v}_{j})$는 노드 $v_{i}$와 다른 노드 $v_{j}$ 사이의 최단 경로 거리를 나타냅니다.

– 합산은 $j = 1$부터 $N$까지 진행되어 네트워크의 모든 노드를 포괄합니다. 합에서 $j \neq i$는 노드 자신까지의 거리(0이 됨)를 제외한다는 뜻입니다.

분모 $(N - 1)$은 다음을 나타냅니다.

– N은 네트워크의 전체 노드 수입니다.

– 노드에서 자기 자신까지의 거리는 포함하지 않으므로 1을 뺍니다.

– 따라서 이는 네트워크에서 다른 노드들의 수를 제공합니다.

따라서 이 공식은 노드 $v_{i}$에서 다른 모든 노드까지의 모든 최단 경로를 합산하고 이를 다른 노드의 수로 나누어 평균을 계산합니다. $g(v_{i})$ 값이 낮을수록 해당 노드가 일반적으로 네트워크의 다른 노드에 더 가깝다는 것을 의미하며, 값이 높을수록 더 멀리 떨어져 있는 경향이 있음을 시사합니다.

예를 들어, 어떤 노드가 직접 연결을 많이 가지고 있다면 다른 노드까지의 최단 경로가 작아지는 경향이 있어 $g(v_{i})$ 값이 낮아집니다. 이는 해당 노드가 네트워크에서 잘 연결되어 있으며 중심적인 위치에 있음을 나타냅니다. 사기 사용 사례에서 사기 노드의 $g(v_{i})$ 값이 낮다면, 사기가 네트워크를 통해 쉽게 확산되어 다른 노드를 더 빠르게 오염시킬 수 있습니다.

근접 중심성은 네트워크에서 더 중심적인 노드에 더 높은 값을 부여하기 때문에 원거리성의 역수입니다. 공식은 다음과 같습니다.

$$
{\mathrm{closeness\ centrality}}(v_{i}) = \left({\frac{\sum_{j=1}^{N} \mathsf{\Gamma}_{(j \neq i)} d(v_{i}, \ v_{j})}{N - 1}}\right)^{-1}
$$

두 가지 문제가 발생할 수 있습니다. 첫째, 네트워크의 모든 노드에 대한 근접 중심성 값이 서로 가까울 수 있으므로, 차이를 확인하기 위해 소수점 이하 자릿수를 살펴봐야 하는 경우가 많습니다. 둘째, 어떤 노드가 다른 노드에 도달할 수 없을 때(두 노드 사이에 경로가 존재하지 않을 때), 두 노드 사이의 거리는 무한대입니다. 이 문제를 극복하기 위해 근접 중심성은 도달할 수 없는 노드까지의 거리를 제외합니다. 따라서 어떤 노드에 대해 근접 중심성을 계산할 때, 계산은 전체 네트워크를 사용하지 않고 해당 노드에서 도달 가능한 네트워크 부분만 사용합니다.

다음 코드는 networkx 그래프를 사용하여 표현된 네트워크의 근접 중심성 (closeness centrality)을 계산합니다.

#### 목록 10.7 사기 탐지 네트워크에서 근접 중심성 계산하기


```python
import networkx as nx
from collections import defaultdict
def compute_closeness_metrics(G): Initializes a dictionary to store
closeness_metrics = {} < closeness values for each node
```

```python
for node in G.nodes():
total_distance = 0 Uses networkx to calculate
reachable_nodes = 0 shortest paths to all nodes
shortest_paths = nx.single_source_shortest_path_length(G, node) <
for other_node, distance in shortest_paths.items():
Excludes self if other_node != node:
total_distance += distance Counts reachable nodes
reachable_nodes += 1 < and sums up the distances
n = len(G.nodes()) - 1 Total nodes minus self
if reachable_nodes > 0 and n > 0:
closeness = (reachable_nodes / n) * (reachable_nodes /
total_distance) < Calculates the normalized
else: closeness centrality considering
closeness = 0.0 disconnected components
closeness_metrics[node] = round(closeness, 2) < Rounds the closeness
to two decimal places
return closeness_metrics < Returns a dictionary containing
closeness values for all nodes
def get_node_closeness(G, node):
metrics = compute_closeness_metrics(G)
return metrics.get(node, 0.0) < Gets closeness value
for a specific node
def analyze_closeness_distribution(G):
metrics = compute_closeness_metrics(G)
values = list(metrics.values())
stats = {
'max_closeness': max(values),
'min_closeness': min(values),
'avg_closeness': sum(values) / len(values),
'most_central_node': max(metrics.items(), key=lambda x: x[1])[0],
'least_central_node': min(metrics.items(), key=lambda x: x[1])[0]
} <
Analyzes the distribution
of closeness values
return stats
```

코드는 먼저 networkx의 경로 탐색 알고리즘을 사용하여 대상 노드에서 네트워크의 다른 모든 노드까지의 최단 경로를 결정합니다. 이 단계는 중심성 계산에 필요한 기본 거리를 제공합니다.

그러나 실제 네트워크는 완전히 연결되어 있지 않을 수 있습니다. 이러한 일반적인 상황을 처리하기 위해, 코드는 도달 가능한 노드만 고려하는 정규화 전략을 포함합니다. 이 접근법은 연결되지 않은 구성 요소가 중심성 값을 왜곡하는 것을 방지하면서도 부분적으로 연결된 네트워크에 대해 의미 있는 측정값을 제공합니다.

근접 중심성의 계산은 앞서 소개한 공식을 따릅니다. 결과를 더 실용적이고 해석 가능하게 만들기 위해, 네트워크 전반의 근접 값 분포를 이해하는 데 도움이 되는 분석 기능을 포함했습니다.

이는 네트워크 구조에서 중심점 역할을 하는 노드를 식별하려고 할 때 특히 유용할 수 있습니다.

각 노드를 이러한 방식으로 처리함으로써, 우리는 각 노드가 네트워크의 나머지 노드에 비해 얼마나 중심적인지에 대한 포괄적인 관점을 얻으며, 더 쉬운 비교와 분석을 위해 값은 0과 1 사이로 정규화됩니다(표 10.5 참조). 이 구현은 이론적 정확성과 실제 적용 가능성 사이의 균형을 이루므로, 연구와 실제 응용 모두에 적합합니다.

표 10.5 그림 10.1의 사기 그래프에 대한 근접성 지표
<table><tr><td rowspan=1 colspan=1>노드</td><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>B</td><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>F</td><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>H</td><td rowspan=1 colspan=1>_</td><td rowspan=1 colspan=1>J</td></tr><tr><td rowspan=1 colspan=1>근접성</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.39</td><td rowspan=1 colspan=1>0.28</td><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=1>0.26</td><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1>0.43</td><td rowspan=1 colspan=1>0.37</td><td rowspan=1 colspan=1>0.45</td><td rowspan=1 colspan=1>0.26</td></tr><tr><td rowspan=1 colspan=1>노드</td><td rowspan=1 colspan=1>K</td><td rowspan=1 colspan=1>L</td><td rowspan=1 colspan=1>M</td><td rowspan=1 colspan=1>N</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>P</td><td rowspan=1 colspan=1>Q</td><td rowspan=1 colspan=1>R</td><td rowspan=1 colspan=1>S</td><td rowspan=1 colspan=1>T</td></tr><tr><td rowspan=1 colspan=1>근접성</td><td rowspan=1 colspan=1>0.34</td><td rowspan=1 colspan=1>0.26</td><td rowspan=1 colspan=1>0.26</td><td rowspan=1 colspan=1>0.34</td><td rowspan=1 colspan=1>0.40</td><td rowspan=1 colspan=1>0.29</td><td rowspan=1 colspan=1>0.31</td><td rowspan=1 colspan=1>0.24</td><td rowspan=1 colspan=1>0.24</td><td rowspan=1 colspan=1>0.34</td></tr></table>

노드 A는 네트워크의 다른 모든 노드와 가장 밀접하게 연결되어 있습니다. 노드 R과 S는 다른 모든 노드로부터 가장 멀리 떨어져 있습니다.

#### 10.1.6 매개성 (Betweenness)


매개 중심성 (betweenness centrality)은 근접 중심성과는 다른 관점에서 네트워크 내 노드의 중요성을 이해하는 데 도움을 줍니다. 근접 중심성이 한 노드가 다른 노드들에 얼마나 빨리 도달할 수 있는지를 측정하는 반면, 매개 중심성은 한 노드가 다른 노드들 사이에서 얼마나 자주 다리 역할을 하는지를 측정합니다. 구체적으로, 이는 한 노드가 다른 노드 쌍들 사이의 최단 경로에 나타나는 횟수를 정량화합니다.

네트워크의 임의의 노드 쌍에 대해 정보나 영향력은 대체로 그들 사이의 최단 경로를 따라 흐를 가능성이 큽니다. 특정 노드가 이러한 최단 경로에 자주 나타난다면, 그 노드는 높은 매개 중심성을 가지며 따라서 많은 다른 노드들 사이의 정보 흐름을 잠재적으로 통제할 수 있습니다. 수학적으로 노드 v의 매개 중심성은 다음과 같이 계산됩니다.

$$
\mathrm{betweenness}(v) = \sum_{s,t\ (s \neq t \neq v)} {\frac{\sigma_{st}(v)}{\sigma_{st}}}
$$

여기서 $\sigma_{st}$는 노드 s와 t 사이의 최단 경로의 총수를 나타내며, $\sigma_{st}(\boldsymbol{v})$는 그 경로들 중 노드 v를 통과하는 경로의 수를 나타냅니다. 다음 목록에 나타난 것처럼, 이 합은 v와 같지 않은 모든 노드 쌍 s와 t에 대해 계산됩니다.

```python
Listing 10.8 Computing betweenness centrality in a fraud detection network
import networkx as nx
from collections import defaultdict
Initializes a dictionary
def compute_betweenness_metrics(G, normalized=True): to store betweenness
betweenness_metrics = {} < values for all nodes
```

betweenness = nx.betweenness\_centrality(   
G,   
normalized=normalized, networkx의 구현을 사용해   
endpoints=False < 매개성을 계산함   
값을 소수점 셋째 자리로   
반올림하여 가독성을   
for node in G.nodes(): 높이고 저장함   
betweenness\_metrics[node] = round(betweenness[node], 3) <   
return betweenness\_metrics < 모든 노드의 매개성 값을   
포함하는 딕셔너리를 반환함   
def analyze\_betweenness\_distribution(G):   
metrics = compute\_betweenness\_metrics(G)   
values = list(metrics.values())   
return {   
'max\_betweenness': max(values), 평균보다 높은 매개성을 가진   
'min\_betweenness': min(values), 노드를 핵심 교량으로 식별함   
'avg\_betweenness': sum(values) / len(values),   
'key\_bridges': [node for node, score in metrics.items()   
if score > sum(values) / len(values)] <   
}   
def get\_node\_betweenness(G, node):   
metrics = compute\_betweenness\_metrics(G) 특정 노드의 매개성   
값을 가져옴   
return metrics.get(node, 0.0) <   
def identify\_potential\_bottlenecks(G, threshold=0.5):   
metrics = compute\_betweenness\_metrics(G)   
bottlenecks = {node: score for node, score in metrics.items()   
if score > threshold} < 임곗값을 기준으로 잠재적 병목을   
return bottlenecks 식별함

이 코드는 networkx의 매개 중심성 알고리즘을 사용하면서 분석과 해석을 위한 실용적 기능을 추가합니다. 계산된 매개성 값은 기본적으로 정규화되며, 이는 값이 0과 1 사이에 오도록 스케일링되어 서로 다른 네트워크 간 비교를 더 쉽게 만든다는 뜻입니다. 1에 가까운 값은 해당 노드가 많은 최단 경로에 나타나며 따라서 정보 흐름을 통제할 높은 잠재력을 가진다는 것을 나타냅니다.

표 10.6의 결과를 살펴보면 흥미로운 패턴을 확인할 수 있습니다. 매개 중심성 (betweenness centrality)이 104인 노드 A는 네트워크에서 중요한 연결 다리로 보이며, 많은 다른 노드들 사이의 정보 흐름을 잠재적으로 통제할 수 있습니다. 반대로 노드 C, E, J, L, M, P, R, S, T는 모두 매개 값이 0으로, 어떤 노드 쌍 사이에서도 연결 다리 역할을 하지 않음을 나타냅니다.

구현에는 네트워크에서 잠재적 병목 지점과 핵심 연결 다리를 식별하는 데 도움이 되는 추가 분석 도구가 포함되어 있습니다. 이러한 도구는 네트워크 구조의 취약성을 이해하거나 사기 탐지 맥락에서 추가 모니터링이 필요할 수 있는 노드를 식별하려 할 때 유용할 수 있습니다.

표 10.6 그림 10.1의 사기 그래프의 매개 지표
<table><tr><td rowspan=1 colspan=1>노드</td><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>B</td><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>F</td><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>H</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>J</td></tr><tr><td rowspan=1 colspan=1>매개성</td><td rowspan=1 colspan=1>104</td><td rowspan=1 colspan=1>24.67</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>13.83</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>6.167</td><td rowspan=1 colspan=1>35.3</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>노드</td><td rowspan=1 colspan=1>K</td><td rowspan=1 colspan=1>L</td><td rowspan=1 colspan=1>M</td><td rowspan=1 colspan=1>N</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>P</td><td rowspan=1 colspan=1>Q</td><td rowspan=1 colspan=1>R</td><td rowspan=1 colspan=1>S</td><td rowspan=1 colspan=1>T</td></tr><tr><td rowspan=1 colspan=1>매개성</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>34</td><td rowspan=1 colspan=1>63</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr></table>

#### 10.1.7 PageRank


PageRank는 들어오는 연결의 구조를 기반으로 노드의 중요도를 측정하는 강력한 지표입니다. 원래 Google의 창립자들이 웹 페이지의 순위를 매기기 위해 개발했지만, 사기 탐지를 포함한 다른 많은 네트워크 분석 맥락에서도 가치가 있음이 입증되었습니다. 더 단순한 중심성 척도와 달리, PageRank는 연결의 양뿐만 아니라 그 질도 고려합니다. 즉, 높은 순위를 가진 다른 노드들과 연결된 노드는 더 높은 PageRank 점수를 받게 됩니다.

우리의 사기 탐지 맥락에서는 PageRank를 두 가지 흥미로운 방식으로 조정할 수 있습니다. 첫째, 모든 연결을 동일하게 고려하는 기본 PageRank를 계산합니다. 그런 다음 알려진 사기 노드에서 오는 연결에 더 큰 가중치를 부여하는 사기 가중 PageRank를 계산합니다. 다음 목록에 제시된 이 이중 접근법은 네트워크에서 노드의 일반적 중요성과 사기 활동과의 구체적 관계를 모두 이해하는 데 도움이 됩니다 [4].

```python
Listing 10.9 Computing PageRank variations for fraud detection
import networkx as nx
import numpy as np
def compute_pagerank_metrics(G, fraud_weight=2.0, damping_factor=0.85):
pagerank_metrics = {} < Initializes a dictionary to store
base_pagerank = nx.pagerank( both PageRank variations
G,
alpha=damping_factor,
personalization=None,
weight=None Calculates standard PageRank
) < using networkx implementation
fraud_personalization = {}
for node in G.nodes(): < Creates a personalization
if G.nodes[node].get('is_fraudster', False): dictionary that gives higher
fraud_personalization[node] = fraud_weight weight to fraudulent nodes
else:
fraud_personalization[node] = 1.0
fraud_pagerank = nx.pagerank(
G,
alpha=damping_factor,
personalization=fraud_personalization,
weight=None Calculates fraud-weighted
< PageRank using personalization
```

```python
for node in G.nodes():
pagerank_metrics[node] = {
'pagerank_base': round(base_pagerank[node], 3),
'pagerank_fraud': round(fraud_pagerank[node], 3)
} < Stores both PageRank
values for each node
return pagerank_metrics < Returns a dictionary
containing all
PageRank metrics
def get_node_pagerank(G, node):
metrics = compute_pagerank_metrics(G)
return metrics.get(node, {
'pagerank_base': 0.0,
'pagerank_fraud': 0.0 Gets the PageRank values
}) < for a specific node
```
예시 그래프의 각 노드에 대한 결과는 표 10.7에 제시되어 있습니다.

표 10.7 그림 10.1의 사기 그래프에 대한 PageRank 지표
<table><tr><td>노드</td><td>A</td><td>B</td><td>C</td><td>D</td><td>E</td><td>F</td><td>G</td><td>H</td><td>-</td><td></td></tr><tr><td>기본 PageRank</td><td>0.108</td><td>0.057</td><td>0.023</td><td>0.068</td><td>0.036</td><td>0.051</td><td>0.067</td><td>0.04</td><td>0.07</td><td>0.024</td></tr><tr><td>사기 PageRank</td><td>0.087</td><td>0.063</td><td>0.018</td><td>0.168</td><td>0.114</td><td>0.145</td><td>0.109</td><td>0.023</td><td>0.094</td><td>0.011</td></tr><tr><td>노드</td><td>K</td><td>L</td><td>M</td><td>N</td><td>0</td><td>P</td><td>Q</td><td>R</td><td>S</td><td>T</td></tr><tr><td>기본 PageRank</td><td>0.06</td><td>0.041</td><td>0.041</td><td>0.057</td><td>0.066</td><td>0.026</td><td>0.75</td><td>0.028</td><td>0.028</td><td>0.022</td></tr><tr><td>사기 PageRank</td><td>0.039</td><td>0.016</td><td>0.016</td><td>0.034</td><td>0.02</td><td>0.005</td><td>0.011</td><td>0.003</td><td>0.003</td><td>0.012</td></tr></table>

노드 A는 가장 높은 기본 PageRank(0.108)를 가지며, 이는 네트워크 구조에서 그 일반적 중요성을 나타냅니다. 그러나 사기 가중 PageRank를 살펴보면 노드 D가 가장 중요한 노드(0.168)로 부상하며, 이는 기본 PageRank가 가장 높지는 않음에도 사기 활동과 더 강한 연결을 갖고 있음을 시사합니다.

기본 PageRank와 사기 가중 PageRank 사이의 이러한 차이는 가치 있는 통찰을 제공할 수 있습니다. 기본 PageRank에 비해 사기 가중 PageRank가 크게 증가한 노드들은 더 면밀한 조사를 필요로 할 수 있는데, 이는 해당 노드들이 전체 네트워크상 위치가 시사하는 것보다 알려진 사기 노드들과 더 실질적인 연결을 갖고 있기 때문입니다.

#### 10.1.8 예측


특징 추출을 계속할 수도 있지만, 이 시점에서는 과정이 어떻게 작동하는지 분명해졌을 것입니다. 이 과정은 반복적이어야 하며, 분류기 (classifier)가 충분하다고 판단되는 예측 품질에 도달할 때까지 특징은 변경되고 증가할 수 있습니다. 이 절에서는 지금까지 추출한 특징을 사용하고, 9장에서 사용했던 동일한 알고리즘인 로지스틱 회귀 (logistic regression)를 사용하여 예측을 수행합니다. 다음 목록은 각 노드에 대한 특징 벡터를 추출하고 다음 단계를 위해 데이터셋을 준비합니다.

```python
Listing 10.10 Creating node features
import pandas as pd
import numpy as np
def create_node_features_dataset(G):
degree_metrics = compute_degree_metrics(G)
triangle_metrics = compute_triangle_metrics(G)
density_metrics = compute_density_metrics(G)
path_metrics = compute_geodesic_metrics(G)
closeness_metrics = compute_closeness_metrics(G)
betweenness_metrics = compute_betweenness_metrics(G) Computes all metrics
pagerank_metrics = compute_pagerank_metrics(G) < for each node
features_dict = {}
for node in G.nodes():
features_dict[node] = {
'total_degree': degree_metrics[node]['total_degree'],
'fraud_degree': degree_metrics[node]['fraud_degree'],
'legit_degree': degree_metrics[node]['legit_degree'],
'total_triangles': triangle_metrics[node]['total_triangles'],
'fraud_triangles': triangle_metrics[node]['fraud_triangles'],
'legit_triangles': triangle_metrics[node]['legit_triangles'],
'semi_fraud_triangles':
triangle_metrics[node]['semi_fraud_triangles'],
'density': density_metrics[node],
'geodesic_path': path_metrics[node]['geodesic_path'],
'paths_1hop': path_metrics[node]['#1-hop_paths'],
'paths_2hop': path_metrics[node]['#2-hop_paths'],
'paths_3hop': path_metrics[node]['#3-hop_paths'],
'closeness': closeness_metrics[node],
'betweenness': betweenness_metrics[node],
'pagerank_base': pagerank_metrics[node]['pagerank_base'],
'pagerank_fraud': pagerank_metrics[node]['pagerank_fraud'],
#### Label
'is_fraudster': G.nodes[node]['is_fraudster']
} <
Creates a comprehensive
feature set for each node
#### Convert to DataFrame
df = pd.DataFrame.from_dict(features_dict, orient='index') <
Converts to a pandas
return df
DataFrame for easier
manipulation
```

이제 각 노드에 대한 특징이 pandas DataFrame에 있으므로, 다음 목록에 보인 것처럼 이를 분할하여 일부는 학습에 사용하고 일부는 학습된 모델의 품질을 검증하는 데 사용할 수 있습니다.

```python
Listing 10.11 Training a fraud classifier and evaluating its accuracy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
def train_fraud_classifier(G):
df = create_node_features_dataset(G)
X = df.drop('is_fraudster', axis=1) Separates features (X)
y = df['is_fraudster'] < from labels (y)
X_train, X_test, y_train, y_test = train_test_split( Splits the data into
X, y, test_size=0.2, random_state=42, stratify=y training (80%) and
) < testing (20%) sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) Scales the features to
X_test_scaled = scaler.transform(X_test) < normalize their ranges
clf = LogisticRegression(random_state=42, max_iter=1000) Trains the logistic
clf.fit(X_train_scaled, y_train) regression classifier
y_pred = clf.predict(X_test_scaled) < Calculates feature
importance
feature_importance = pd.DataFrame({
'feature': X.columns,
'importance': abs(clf.coef_[0]) Collects all
}).sort_values('importance', ascending=False) < evaluation metrics
results = {
'classification_report': classification_report(y_test, y_pred),
'confusion_matrix': confusion_matrix(y_test, y_pred),
'feature_importance': feature_importance,
'model': clf,
'scaler': scaler Makes
} predictions
return results
```

계층화 분할 (stratified split, 매개변수 stratify를 yes로 설정하여 얻음)은 학습 세트와 테스트 세트가 원래 데이터셋과 동일한 사기 노드 및 정상 노드 비율을 유지하도록 보장합니다. StandardScaler는 모든 특징이 동일한 스케일에 놓이도록 보장하며, 이는 로지스틱 회귀에서 중요합니다. 특징 중요도 분석은 사기 노드를 탐지하는 데 어떤 지표가 가장 유용한지 이해하는 데 도움을 줍니다.

모델을 확보했으므로, 이제 다음 함수를 사용하여 한 노드가 사기꾼일 가능성을 예측할 준비가 되었습니다.

```python
def predict_fraud_probability(G, node, trained_results):
df = create_node_features_dataset(G)
node_features = df.loc[node].drop('is_fraudster') < Gets node
features
scaled_features = trained_results['scaler'].transform(
node_features.values.reshape(1, -1)
) < Scales
features
fraud_prob =
trained_results['model'].predict_proba(scaled_features)[0][1] <
Gets the
return fraud_prob
probability
```

모든 함수가 준비되었습니다. 이제 간단한 분석과 테스트를 실행하여 어떤 노드가 사기꾼을 나타내는지 또는 정상인을 나타내는지를 보여 주는 다섯 가지 가장 중요한 특징을 드러낼 수 있습니다.

```python
Listing 10.13 Obtaining the results of the entire process
G = create_fraud_network()
results = train_fraud_classifier(G) < Creates and analyzes
the network
print("Classification Report:")
print(results['classification_report']) < Prints a
classification report
print("\nConfusion Matrix:")
print(results['confusion_matrix']) < Prints a
confusion matrix
print("\nTop 5 Most Important Features:")
print(results['feature_importance'].head()) < Shows the top five most
important features
node_of_interest = 'A'
prob = predict_fraud_probability(G, node_of_interest, results)
print(f"\nFraud probability for node {node_of_interest}: {prob:.3f}") <
Example: Gets the fraud
probability for a specific node
```

결과는 살펴보지 않겠습니다. 통계적으로 의미가 있을 만큼 충분히 크지 않은 샘플 그래프에서 얻은 것이기 때문입니다. 그러나 이 과정은 더 크고 더 현실적인 그래프에도 적용할 수 있습니다. 이 접근법의 장점은 과정이 복잡하고 광범위한 특징 설계와 신중한 고려가 필요했음에도 불구하고, 전적으로 사용자의 통제하에 있으며 추출된 각 특징은 그래프를 살펴보는 것만으로도 쉽게 설명할 수 있다는 점입니다. 이는 과정의 투명성을 높이며, 예를 들어 데이터베이스의 크기가 제한적이거나 설명 가능성이 중요한 경우에 이 방법이 타당한 이유입니다.

### 10.2 수동 관계 특징


구조적 특징을 통해 노드를 표현하는 방법을 살펴본 후, 이제 그래프 ML의 또 다른 근본적인 과제, 즉 노드 간 관계를 어떻게 표현할 것인지에 주목합니다. 노드는 그래프의 주요 요소이지만, 노드 사이의 연결에는 우리가 포착하고 분석해야 하는 정보가 담겨 있는 경우가 많습니다.

그래프의 요소들 사이에서 잠재적 상호작용을 예측하고자 하는 상황을 상상해 보십시오. 바로 여기에서 관계 예측 (relationship prediction, 링크 예측 (link prediction)으로도 알려짐)이 등장합니다. 두 단백질이 상호작용할 수 있는지, 고객이 어떤 제품에 관심을 가질 수 있는지, 또는 어떤 약물이 질병을 치료할 수 있는지를 예측하려는 경우든, 우리는 본질적으로 동일한 질문을 하고 있습니다. 두 노드가 주어졌을 때, 그들 사이에 관계가 존재할 가능성은 얼마나 되는가?

노드 분류 (node classification)와 유사하게, 관계 예측은 그래프 요소를 ML 알고리즘이 처리할 수 있는 특징으로 변환할 것을 요구합니다. 그러나 개별 노드를 표현하는 대신, 노드 간 잠재적 연결의 특성을 포착해야 합니다. 이는 이진 분류 (binary classification) 과제, 즉 관계가 존재하는지 여부를 예측하는 과제로 접근할 수도 있고, 다중 클래스 분류 (multiclass classification) 과제, 즉 존재할 수 있는 관계의 유형을 예측하는 과제로 접근할 수도 있습니다.

이러한 개념을 설명하기 위해, 우리는 약물 재창출 (drug repurposing), 즉 기존 약물의 새로운 용도를 찾는 실제 응용 사례를 살펴보겠습니다. 이 과제는 약물과 질병 사이의 링크 예측 문제로 모델링할 수 있으며, 여기서 약물은 보다 공식적으로 화합물 (compounds)이라고도 합니다. 우리의 목표는 잠재적 치료 관계를 예측하는 것입니다. 이 예시를 통해, 노드 특징 공학 (feature engineering)에 대해 살펴본 개념을 바탕으로 그래프에서 관계의 의미 있는 표현을 만드는 다양한 전략을 검토합니다. 9장에서 가져온 그림 10.4는 노드 분류와 링크 예측이라는 두 과제를 비교하고, 두 노드의 표현을 결합하여 관계의 표현을 얻는 방법을 예고합니다.

노드 분류에서 핵심적인 차이는, 예측하거나 학습 중에 사용할 링크를 나타내기 위해 노드 쌍을 입력으로 사용한다는 점입니다. 따라서 링크 예측이 정확하려면, 입력으로 사용되는 노드 쌍 사이의 가능한 연결을 효과적으로 표현하는 방법이 필요합니다. 우리는 두 가지 서로 다른 접근법을 사용할 수 있습니다.

노드 기반 결합—소스 노드와 대상 노드의 특징 벡터를 결합하여 관계 표현을 도출합니다. 예를 들어 노드 표현이 [1,2,3]과 [4,5,6]이라면, 이들의 잠재적 연결을 표현하기 위해 연결 (concatenation)이나 원소별 곱셈 (element-wise multiplication)과 같은 연산을 통해 결합할 수 있습니다. 이것이 그림 10.4에 제시된 경우입니다.

 경로 기반 특징—노드 특징에 의존하는 대신, 그래프에서 노드들이 연결되는 방식을 분석하여 관계를 특성화합니다. 각 특징은 노드 사이의 서로 다른 경로 패턴을 나타내며, 예를 들어 2-홉 경로의 수나 특정 메타경로 (metapath)의 존재 여부와 같은 요소를 통해 관계의 구조적 맥락을 포착하는 벡터를 생성합니다.

노드 기반 조합은 노드 임베딩과 잘 작동하는 반면, 경로 기반 특징은 복잡한 네트워크 패턴을 포착하는 데 뛰어납니다. 이제 각각을 자세히 살펴보겠습니다.

![](images/0d56c43a7fe97044f9f7aa9dcac863bf0cc454d443256e3689e310233b42253c.jpg)  
그림 10.4 노드 분류 과정과 비교한 일반적인 관계 예측 과정

#### 10.2.1 노드 기반 표현


가장 직관적인 접근법은 연결될 수 있는 두 노드의 특징 표현을 결합하는 것입니다. 각 노드는 자신을 설명하는 특징 집합을 가지며(지문과 유사합니다), 우리는 이러한 특징을 병합하여 두 노드 사이의 연결이 어떤 모습일 수 있는지를 설명하고자 합니다. 이러한 결합은 그래프에서 실제로 연결되어 있는지 여부와 관계없이 모든 노드 쌍에 대해 작동해야 합니다. 이는 링크를 예측할 때 기존 연결과 잠재적 연결을 모두 평가해야 하기 때문에 중요합니다.

다음은 링크 예측을 염두에 두고 벡터를 결합하는 데 가장 흔히 사용되는 기법입니다.

연결 (Catenate)—두 벡터를 끝과 끝이 이어지도록 결합합니다. 길이가 n인 벡터 u와 v에 대해, u의 원소 뒤에 v의 원소를 배치하여 길이가 2n인 새 벡터를 만듭니다. 예를 들어 u = [1,2]이고 $\mathbf{v} = [3,4]$이면, 이들의 연결 결과는 [1,2,3,4]입니다. 이는 원래 정보를 모두 보존하지만 결과 벡터의 차원을 두 배로 만듭니다.

평균—두 입력 벡터의 원소별 평균(mean)을 취해 새 벡터를 만듭니다. 각 위치 $i$에 대해 $(\mathbf{u}[i] + \mathbf{v}[i]) / 2$를 계산합니다. 이는 원래 차원을 유지하면서 두 벡터 사이의 중심 경향을 포착합니다. 예를 들어 u = [2,4]이고 v = [4,8]이면, 이들의 평균은 [3,6]입니다.

리스팅 10.14 두 노드 표현을 하나의 링크 표현으로 결합하기   
def catenate(u, v):   
return u + v   
def operator\_avg(u, v):   
return (u + v) / 2.0   
def operator\_l1(u, v):   
return np.abs(u - v)   
def operator\_l2(u, v):   
return (u - v) \*\* 2   
def operator\_hadamard(u, v):   
return u \* v   
링크 예측의 품질은 노드 표현의 품질에 직접적으로 의존합니다. 합성 기법 또한 최종 품질에 영향을 미칩니다. 선택을 도와주는 일반적인 규칙은 없으므로, 벤치마크를 통해 사용자의 시나리오에 적합한 접근법이 무엇인지에 대한 지표를 얻을 수 있습니다.

L1(맨해튼 거리)—두 벡터의 대응하는 원소 사이의 절댓값 차이를 계산합니다. 각 위치 i에 대해 |u[i] – v[i]|를 계산합니다. 이는 각 차원에서 벡터들이 얼마나 다른지를 포착하며, 비유사성을 측정하는 데 유용합니다. 예를 들어 u = [1,4]이고 v = [3,1]이면, 이들의 L1 결합은 [2,3]입니다.

L2(유클리드 거리)—대응하는 원소 사이의 제곱 차이를 계산합니다. 각 위치 i에 대해 (u[i] – v[i])²를 계산합니다. L1과 마찬가지로 벡터 간 차이를 포착하지만, 제곱 연산으로 인해 더 큰 차이를 더 강조합니다. 예를 들어 u = [1,4]이고 v = [3,1]이면, 이들의 L2 결합은 [4,9]입니다.

아다마르(Hadamard, 원소별 곱)—두 벡터의 대응하는 원소를 곱합니다. 각 위치 i에 대해 u[i] × v[i]를 계산합니다. 이 연산은 두 벡터의 값이 곱셈 방식으로 결합되어야 하는 관련 수량을 나타낼 때 특히 유용합니다. 예를 들어 u = [2,4]이고 v = [3,1]이면, 이들의 아다마르 곱은 [6,4]입니다.

다음 목록은 이러한 각 방법을 구현하는 방법을 보여줍니다.

#### 10.2.2 경로 기반 특징


일부 기법은 출발 노드와 도착 노드를 고려하되, 이들의 표현을 사용하지 않거나 이들만을 배타적으로 사용하지 않는 방식으로 노드 쌍 표현을 설명합니다. 이러한 기법 중 다수는 관계를 적절히 표현하기 위해 수작업을 필요로 합니다.

표현의 수작업 과정은 일반적으로 도메인에 특화되어 있으므로, 해당 도메인과 우리가 달성하려는 목표에 대한 어느 정도의 이해가 필요합니다. 예를 들어 4장에서와 같이 생의학 지식 그래프와 그것이 신약 발견 또는 약물 재창출 (drug repurposing)에 어떻게 도움이 될 수 있는지를 살펴보겠습니다. 우리는 다시 Hetionet 데이터셋을 사용할 것인데, 이는 약물, 질병, 유전자, 증상을 나타내는 50,000개 이상의 노드를 포함하는 19개의 공개 데이터베이스 정보를 통합한 것입니다.

참고 모든 예제를 따라 하고 있다면 이미 데이터베이스가 있어야 합니다. 그렇지 않다면 4장으로 돌아가 데이터베이스를 만드는 방법을 확인하십시오.

Himmelstein 등 [5]은 Hetionet 데이터셋을 사용하여 약물 재창출에서 중요한 진전을 이루었습니다. 계산 분석을 통해 이들은 이러한 요소들 사이에 200만 개 이상의 관계를 확립했습니다. 그리고 이들의 연구는 실질적인 결과를 낳았습니다. 이들은 흡연 중독과 간질 치료에 잠재력을 보인, 우울증과 알코올 중독에 사용되는 기존 약물을 식별했습니다. 이제 이들의 방법론적 접근을 높은 수준에서 살펴보겠습니다.

4장에서 제시된 Hetionet 스키마를 생각해 보십시오. 여기서는 그림 10.5에 다시 제시되어 있습니다. 목표는 화합물과 질병의 네트워크 연결성을 치료 확률로 변환하는 ML 모델을 학습하는 것입니다 [6, 7]. 연구자들은 노드 간 네트워크 연결성을 반영하는 각 노드 쌍, 즉 화합물과 질병만의 표현을 만들기 위해, 화합물에서 질병으로 이동하고 길이가 2에서 4인 모든 메타경로 (metapath)를 평가했습니다. 그림 10.6은 유전자와 질병 사이의 예를 사용하여 메타그래프 (metagraph)와 메타경로의 개념을 다시 보여줍니다.

![](images/9f72e601d64381937258fdc261660d54f490fce0430c607a3cd549a5bcd18fde.jpg)  
그림 10.5 4장에서 제시된 Hetionet 스키마

![](images/d4ea42f61c5f4309eb5953747fc60cc436fe5b56fb583ccd856413c503f6c695.jpg)  
그림 10.6 Hetionet의 메타그래프와 메타경로 예

그림 10.5의 스키마에서 시작하면, 화합물과 질병 사이의 일부 메타경로가 표 10.8에 나열되어 있습니다. 이러한 메타경로는 화합물과 질병 사이의 가능한 경로 중 일부만을 나타냅니다. 이들은 직접 연결이 존재하는지 여부와 관계없이 각 화합물–질병 쌍에 대한 특징을 형성합니다.

표 10.8 약물과 질병을 연결하는 메타경로 (metapath)의 예 [5]
<table><tr><td>메타경로</td><td>길이</td><td>약어</td></tr><tr><td>Compound—binds-Gene—associates-Disease</td><td>2</td><td>CbGaD</td></tr><tr><td>Compound-downregulates-Gene-upregulates-Disease</td><td>2</td><td>CdGuD</td></tr><tr><td>Compound-resembles-Compound—treats-Disease</td><td>2</td><td>CrCtD</td></tr><tr><td>Compound—binds-Gene—binds-Compound-treats-Disease</td><td>3</td><td>CbGbCtD</td></tr><tr><td>Compound—binds-Gene—expresses-Anatomy—localizes-Disease</td><td>3</td><td>CbGeAID</td></tr><tr><td>Compound—binds-Gene—interacts-Gene—interacts—Gene—associates-Disease</td><td>4</td><td>CbGiGiGaD</td></tr><tr><td>Compound—binds-Gene—participates—Pathway-participates-Gene —associates-Disease</td><td>4</td><td>CbGpPWpGaD</td></tr></table>

예를 들어, 화합물 메트포르민과 제2형 당뇨병이라는 질병을 고려하면, 각 메타경로(CbGaD, CdGuD, CrCtD 등)에 대한 값을 계산해야 합니다. 그림 10.7을 참조하십시오. 가장 단순한 접근법은 화합물과 질병 사이의 서로 다른 경로 인스턴스를 세는 것입니다. 그러나 연결성이 매우 높은 노드가 집계를 지배하는 경우, 단순히 경로를 세는 것은 오해를 불러일으킬 수 있습니다. 예를 들어, 어떤 유전자가 많은 생물학적 과정에 관여한다면 자연스럽게 더 많은 경로에 나타나겠지만, 이것이 반드시 화합물과 질병 사이의 더 강하거나 더 의미 있는 관계를 나타내는 것은 아닙니다.

각 값은 경로 인스턴스의 출발점과 도착점으로서 특정 화합물과 질병에 대해 계산된 메타경로와 관련됩니다.  
![](images/37ef50cd89c02e5f7ec99660bbd42442f17c277f5a63f507dd95f671bf0fd62c.jpg)

이러한 편향을 해결하기 위해, 우리는 차수 가중 경로 수 (degree-weighted path count, DWPC; 4장에서 더 자세히 논의함)를 사용하며, 이는 노드 차수에 기반한 감쇠 인자를 적용합니다. DWPC를 계산할 때,

각 경로는 중간 노드의 차수에 반비례하여 가중치가 부여됩니다.

연결이 많은 노드는 최종 점수에 더 적게 기여합니다.

감쇠 효과는 더 구체적이고 초점이 맞춰진 생물학적 경로를 강조하는 데 도움이 됩니다.

예를 들어, 메트포르민과 제2형 당뇨병 사이에 두 개의 경로가 있다고 가정해 보겠습니다.

하나는 연결성이 높은 유전자(차수 100)를 통과하는 경로입니다.

다른 하나는 더 특이적인 유전자(차수 10)를 통과하는 경로입니다.

두 번째 경로는 DWPC 점수에 더 크게 기여하여 잠재적인 생물학적 중요성을 더 잘 반영합니다. 이 접근법은 생물학적 네트워크의 허브 노드에서 발생하는 잡음을 줄이면서 의미 있는 치료적 관계를 식별하는 데 도움이 되기 때문에, 특히 약물 재창출 (drug repurposing)에서 효과적인 것으로 입증되었습니다.

다음 목록은 Neo4j를 사용하여 CbGaD(Compound–binds–Gene–associates–Disease) 메타경로에 대해 메트포르민과 제2형 당뇨병 사이의 DWPC를 계산합니다.

목록 10.15 CbGaD에 대한 메트포르민과 제2형 당뇨병 사이의 DWPC   
MATCH path = (c:Compound)-[:BINDS\_CbG]-(g)-[:ASSOCIATES\_DaG]-(d:Disease)   
WHERE c.name = 'Metformin' AND d.name = 'type 2 diabetes mellitus'   
WITH   
[   
count{(v)-[:BINDS\_CbG]-()},

목록 10.17 CbGpPWpGaD에 대한 DWPC 값

count{()-[:BINDS\_CbG]-(g)},   
count{(g)-[:ASSOCIATES\_DaG]-()},   
count{()-[:ASSOCIATES\_DaG]-(d)}   
]   
AS degrees, path, d   
WITH   
d.identifier AS disease\_id,   
d.name AS disease\_name,   
count(path) AS PC,   
sum(reduce(pdp = 1.0, d in degrees| pdp \* d ^ -0.4)) AS DWPC   
RETURN   
disease\_id, disease\_name, PC, DWPC

이를 우리의 Hetionet 데이터베이스에서 실행하면 값 0.0007을 얻습니다. 우리는 이 값을 Metformin–Type 2 Diabetes 쌍에 대한 벡터에서 CbGaD 특징의 값으로 사용할 것입니다.

#### 연습문제


기존 직접 연결과 존재하지 않는 직접 연결을 고려하여, 다른 화합물과 질병에 대해 listing 10.15의 쿼리를 실행하십시오. 그런 다음 다른 메타경로 (metapath)를 테스트하도록 쿼리를 변경하십시오.

Listings 10.16과 10.17은 두 가지 추가 예를 제공합니다.

#### Listing 10.16 CbGeAlD에 대한 DWPC 값


MATCH path = (c:Compound)-[:BINDS\_CbG]-(g:Gene)<-[:EXPRESSES\_AeG]-   
[CA](a:Anatomy)<-[:LOCALIZES\_DlA]-(d:Disease)   
WHERE c.name = 'Metformin' AND d.name = 'type 2 diabetes mellitus'   
WITH   
[   
count{(c)-[:BINDS\_CbG]-()},   
count{()-[:BINDS\_CbG]-(g)},   
count{(g)<-[:EXPRESSES\_AeG]-()},   
count{()-[:EXPRESSES\_AeG]-(a)},   
count{(a)<-[:LOCALIZES\_DlA]-()},   
count{()-[:LOCALIZES\_DlA]-(d)}   
] AS degrees, path, d   
WITH   
d.identifier AS disease\_id,   
d.name AS disease\_name,   
count(path) AS PC,   
sum(reduce(pdp = 1.0, d in degrees| pdp \* d ^ -0.4)) AS DWPC   
RETURN disease\_id, disease\_name, PC, DWPC

MATCH path = (c:Compound)-[:BINDS\_CbG]-(g1:Gene)-[:PARTICIPATES\_GpPW]-   
>(pw:Pathway)<-[:PARTICIPATES\_GpPW]-(g2:Gene)-[:ASSOCIATES\_DaG]-   
(d:Disease)   
WHERE c.name = 'Metformin' AND d.name = 'type 2 diabetes mellitus'

WITH   
[   
count{(c)-[:BINDS\_CbG]-()},   
count{()-[:BINDS\_CbG]-(g1)},   
count{(g1)-[:PARTICIPATES\_GpPW]->()},   
count{()-[:PARTICIPATES\_GpPW]->(pw)},   
count{(pw)<-[:PARTICIPATES\_GpPW]-()},   
count{()<-[:PARTICIPATES\_GpPW]-(g2)},   
count{(g2)-[:ASSOCIATES\_DaG]-()},   
count{()-[:ASSOCIATES\_DaG]-(d)}   
] AS degrees, path, d   
WITH   
d.identifier AS disease\_id,   
d.name AS disease\_name,   
count(path) AS PC,   
sum(reduce(pdp = 1.0, d in degrees| pdp \* d ^ -0.4)) AS DWPC   
RETURN disease\_id, disease\_name, PC, DWPC

가능한 모든 메타경로에 걸쳐 모든 가능한 화합물–질병 쌍에 대한 DWPC 값을 계산하는 것은 계산 비용이 많이 들며, 잡음이 많거나 오해를 불러일으키는 결과로 이어질 수 있습니다. 이러한 문제를 해결하기 위해 Himmelstein과 동료들 [5]은 복잡성을 줄이고 모델 성능과 무관하거나 잠재적으로 해로울 수 있는 특징을 제거하기 위한 2단계 접근법을 개발했습니다.

1 메타경로 축소—그들은 알려진 치료 관계와 비치료 관계에서의 빈도를 분석하여 가장 중요한 메타경로를 식별하는 통계적 방법을 개발했습니다. 이 분석은 높은 예측력을 유지하면서 관련 메타경로의 수를 1,026개에서 709개로 줄였습니다.

2 쌍 선택—그들은 도메인 지식과 차수 기반 확률 분석을 사용하여 가장 유망한 화합물–질병 쌍을 식별함으로써 분석을 더욱 정교화했습니다. 이러한 축소는 계산 오버헤드를 줄였을 뿐만 아니라, 가장 관련성 높은 쌍에 집중함으로써 분류기 성능도 향상시켰습니다.

노드 표현을 결합하는 방식은 앞서 논의한 것처럼 더 단순한 접근법을 제공하지만, 수동으로 추출한 노드 특징을 사용할 때는 성능이 좋지 않은 경우가 많습니다. 뒤에서 살펴보겠지만, 이 접근법은 자동으로 추출된 특징을 사용할 때 더 효과적이 됩니다.

#### 그래프 특징 공학에 LLM 사용하기


우리가 이 책 전반에서 여러 차례 보았듯이, LLM은 복잡한 과업에서 지원을 제공할 수 있습니다. 그래프 특징 공학도 그중 하나입니다. LLM은 복잡한 패턴을 이해하고 이를 실행 가능한 코드로 변환해야 하는 과업에서 뛰어납니다. 이는 그래프 데이터베이스를 다룰 때 특히 가치가 있는데, 쿼리를 작성하려면 도메인과 쿼리 언어 모두에 대한 깊은 이해가 필요한 경우가 많기 때문입니다. 예를 들어, 우리의 약물 재창출 사례에서 LLM은 다음을 수행할 수 있습니다.

쿼리 생성—LLM은 메타경로 (metapath)에 대한 고수준 설명을 최적화된 Cypher 쿼리로 변환할 수 있습니다.

(계속)

특징 공학—즉시 명확하게 드러나지 않을 수 있는 관련 패턴과 관계를 제안할 수 있습니다.

코드 생성—쿼리 결과를 실행하고 처리하는 데 필요한 인프라를 만드는 데 도움을 줄 수 있습니다.

예를 들어, 관계 표현을 위한 특징을 추출하기 위해 약물 재창출 네트워크의 여러 메타경로에 대한 Cypher 쿼리를 생성하고자 한다고 가정해 보겠습니다. 다음은 LLM과 함께 사용할 수 있는 효과적인 프롬프트입니다.

![](images/c36e9b2b77bf4278b3875e2b6cebe4f5de03ac01eeee03c1b78872b07dfeae4b.jpg)

AN 당신은 Neo4j와 Cypher 쿼리를 전문으로 하는 그래프 데이터베이스 전문가입니다. 저는 약물 재창출 프로젝트를 진행하고 있으며, 메타경로 분석을 위한 쿼리 생성에 도움이 필요합니다.

다음을 제공하겠습니다.

그래프 스키마(apoc.meta.schema()에서 얻은 것)

 CbGaD에 대한 쿼리 예시

 Compound 노드와 Disease 노드 사이의 메타경로 목록

테스트를 위한 샘플 화합물 및 질병 이름

각 메타경로에 대해 다음을 수행하십시오.

Path Count(PC)와 Degree-Weighted Path Count(DWPC, 감쇠 계수 0.4 사용)를 모두 계산하는 Cypher 쿼리를 생성하십시오.

경로의 각 노드에 대한 차수 계산을 포함하십시오.

disease\_id, disease\_name, PC, DWPC를 반환하십시오.

스키마는 다음과 같습니다.

{여기에 스키마 정의 또는 첨부 파일로 제공}

DWPC에 대한 쿼리 예시는 다음과 같습니다.

MATCH path = (c:Compound)-[:BINDS\_CbG]-(g)-[:ASSOCIATES\_DaG]-   
(d:Disease)   
WHERE c.name = 'Metformin' AND d.name = 'type 2 diabetes mellitus   
WITH   
count{(v)-[:BINDS\_CbG]-()},   
count{()-[:BINDS\_CbG]-(g)},   
count{(g)-[:ASSOCIATES\_DaG]-()},   
count{()-[:ASSOCIATES\_DaG]-(d)}   
AS degrees, path, d   
WITH   
d.identifier AS disease\_id,   
d.name AS disease\_name,   
count(path) AS PC,   
sum(reduce(pdp = 1.0, d in degrees| pdp \* d ^ -0.4)) AS DWPC   
RETURN   
disease\_id, disease\_name, PC, DWPC

다음 메타경로에 대한 쿼리를 생성해 주십시오.

CbGaD (Compound-binds-Gene-associates-Disease)

CdGuD (Compound-downregulates-Gene-upregulates-Disease)

{source compound}를 화합물로 사용하고 {destination disease}를 질병으로 사용하십시오.

이는 Cypher 쿼리를 생성하는 데 사용할 수 있는 프롬프트의 예시입니다. 그런 다음 최종 표현을 생성하기 위한 Python 코드를 작성할 수 있습니다. 즉시 유용한 결과를 반환하도록 프롬프트를 변경할 수도 있지만, 우리의 목표는 LLM을 창의적으로 활용하는 방법에 대한 씨앗을 심는 것이었습니다.

### 10.3 반자동 특징 추출


10.1절과 10.2절에서는 노드와 관계 모두에 대한 수동 특징 공학 (feature engineering)을 살펴보았으며, 도메인 지식이 의미 있는 표현의 생성을 어떻게 안내할 수 있는지 보여주었습니다. 우리는 구조적 지표와 도메인 특화 패턴을 신중하게 선택하면 사기 탐지와 약물 재창출 같은 작업에 효과적인 특징을 만들 수 있음을 보았습니다. 그러나 이러한 수동 접근법은 깊은 통찰을 제공하지만 상당한 어려움도 수반합니다. 광범위한 도메인 전문성이 필요하고, 구현에 시간이 많이 걸리며, 새로운 사용 사례마다 맞춤화해야 합니다.

수동 특징 공학의 장점, 즉 해석 가능성, 신뢰성, 예측 가능성을 유지하면서도 특징 선택 과정의 상당 부분을 자동화할 수 있다면 어떨까요? 바로 여기서 ReFeX (Recursive Feature eXtraction) [8]가 등장하며, 완전한 수동 특징 공학과 11장 및 12장에서 살펴볼 복잡한 신경망 접근법 사이의 중간 지점을 제공합니다. ReFeX는 그래프에서 관련 구조적 특징을 자동으로 식별하고 추출합니다. 블랙박스 신경망 접근법과 달리, ReFeX의 과정은 투명하며 도메인 전문가가 이해하고 검증할 수 있는 해석 가능한 특징을 생성합니다. 이러한 투명성은 특정 예측이 왜 이루어졌는지 설명해야 할 때 특히 중요합니다. 예를 들어, 어떤 계정이 의심스럽다고 표시된 이유를 정당화해야 하는 사기 탐지 시스템을 상상해볼 수 있습니다.

ReFeX의 또 다른 장점은 서로 다른 그래프 전반에서 일관성을 갖는다는 점입니다. ReFeX가 생성하는 특징은 서로 다른 네트워크 간에, 또는 시간이 지남에 따라 동일한 네트워크의 서로 다른 스냅샷 간에도 의미 있게 비교될 수 있습니다. 이는 더 복잡한 신경망 접근법에서는 거의 가능하지 않은 일입니다. 이러한 속성은 그래프 구조가 어떻게 진화하는지 추적하거나 서로 다른 네트워크 전반의 패턴을 비교해야 하는 애플리케이션에서 ReFeX를 특히 유용하게 만듭니다.

이러한 장점을 보여주기 위해, 먼저 작은 예제에서 ReFeX 특징을 수동으로 계산하면서 ReFeX가 어떻게 작동하는지 살펴보겠습니다. 그런 다음 수동 특징 공학이 비현실적인 더 크고 실제적인 시나리오에 ReFeX를 어떻게 적용할 수 있는지 살펴보겠습니다.

ReFeX는 그래프 구조에서 지역 특징과 에고넷 (egonet) 특징을 재귀적으로 생성하여 작동하며, 우리가 이전에 수동으로 계산했던 많은 지표를 자동으로 포착합니다. 이 접근법의 핵심 장점은 다음과 같습니다.

효율성—재귀적 구조 특징의 자동 추출

일관성—특징 생성을 위한 체계적 접근법

해석 가능성—명확한 구조적 의미를 유지하는 생성된 특징

확장성—특징 품질을 유지하면서 더 큰 그래프를 처리할 수 있는 능력

그러나 이 과정에서도 인간의 감독은 여전히 필수적입니다. 도메인 전문가는 다음을 수행할 수 있습니다.

생성된 특징의 관련성을 검증합니다.

특징 선택을 안내하기 위해 도메인 지식을 통합합니다.

사기 탐지에 대한 특징의 기여를 이해하고 설명합니다.

특정 요구 사항에 따라 특징 추출 과정을 수정합니다.

이러한 하이브리드 접근법은 이후 완전 자동 특징 학습 방법에 대한 논의의 토대를 마련합니다. 그곳에서 우리는 딥러닝 기법이 인간의 개입 없이 어떻게 특징을 추출할 수 있는지 살펴볼 것이며, 잠재적으로 더 정교한 패턴 인식을 얻는 대신 해석 가능성을 어느 정도 포기하게 됩니다.

ReFeX는 로컬(노드 기반) 특징과 노드의 이웃에서 오는 특징(에고넷 (egonet) 기반)을 재귀적으로 결합하여 특징을 계산합니다. 이러한 방식으로 알고리즘은 주어진 노드가 연결된 노드들의 정체성이 아니라 그 노드들의 종류를 나타내는 “행동적” 정보를 포착하는 지역 특징을 생성합니다. 서로 다른 그래프를 가로질러 마이닝할 때 중요한 것은 여러분이 누구를 알고 있는지, 또는 누구와 관계를 맺고 있는지입니다.

ReFeX 과정은 두 가지 기본 규칙에 기반합니다.

 구조적—특징 행렬 F의 구성은 노드나 링크에 대한 추가 속성 정보를 요구해서는 안 됩니다.

효과적—좋은 노드 특징은 (1) 그러한 속성이 사용 가능할 때 노드 속성을 예측하는 데 도움이 되어야 하며, (2) 그래프 간에 전이 가능해야 합니다(예: 그래프가 시간에 따라 변화할 때).

이상적인 특징 집합은 데이터 마이닝 작업에 도움이 되어야 합니다. 대표적인 작업에는 노드 분류(일부 레이블이 주어진 후), 그래프 노드의 탈익명화 (de-anonymization), 전이 학습 (transfer learning)이 포함됩니다. 그림 10.8은 이 과정의 입력과 출력을 매우 단순한 방식으로 나타냅니다.

ReFeX는 노드 레이블이나 유형을 고려하지 않고 그래프의 순수한 구조적 측면, 즉 노드와 관계에 대해 작동합니다. 이러한 토폴로지에 대한 초점은 알고리즘이 구조적 패턴을 식별할 수 있게 합니다. 특징 추출 과정은 세 가지 주요 단계로 이루어집니다.

1 로컬 특징 추출은 즉각적인 노드 특성에 초점을 맞춥니다. 이 수준에서의 주요 지표는 가중 및 비가중 변형을 모두 포함하는 노드 차수 (node degree)입니다. 방향 그래프로 작업할 때 ReFeX는 진입 차수와 진출 차수를 별도로 계산합니다. 가중 그래프의 경우, 입사 간선 가중치의 합으로 가중 차수를 계산하여 노드 연결성에 대한 더 미묘한 관점을 제공합니다.

![](images/e22f7a0802bed8d604e72b61736c91799b6057e6c69cef9ee606efcc925642b3.jpg)  
그림 10.8 ReFeX는 각 노드를 서로 다른 규모에서의 노드 토폴로지 특징을 나타내는 벡터로 변환합니다 [8].

2 에고넷 특징 검토는 각 노드의 즉각적인 이웃을 분석합니다. 이 수준에서 ReFeX는 유입 에고넷 간선 수, 유출 에고넷 간선 수, 전체 에고넷 간선 수를 포함한 지표를 계산합니다. 가중 그래프로 작업할 때는 에고 네트워크 (ego network) 내부 연결의 강도를 포착하기 위해 이러한 지표의 가중 변형도 계산합니다.

3 재귀적 특징 추출은 요약 통계의 재귀적 적용을 통해 기존 특징을 집계합니다. 이 과정은 점점 더 복잡한 구조적 패턴을 포착하기 위해 집계 함수(합/평균)의 조합을 사용합니다. 예를 들어, degree(sum)(mean)(mean)(sum)과 같은 특징은 즉각적인 이웃을 넘어 확장되는 지역 구조 패턴을 포착할 수 있습니다. 방향 그래프에서 ReFeX는 유입 경로와 유출 경로에 대해 이러한 재귀적 특징을 별도로 계산하여 네트워크의 방향성 패턴에 대한 포괄적인 관점을 제공합니다.

알고리즘의 재귀적 특성은 지수적으로 증가하는 수의 특징을 생성할 수 있습니다. 이러한 복잡성을 관리하기 위해 ReFeX는 여러 가지 가지치기 (pruning) 기법을 사용합니다.

상관 분석 (Correlation analysis)—상관관계가 높은 특징 쌍을 식별하고 제거합니다.

로그 binning (Logarithmic binning)—효율적인 비교를 위해 특징값을 이산 구간에 매핑합니다.

임계값 기반 가지치기 (Threshold-based pruning)—지정된 임계값보다 차이가 작은 특징을 제거합니다.

#### 10.3.1 ReFeX 수동 수행


![](images/54c9e5cc8136995e504a3b5558e5ace9f5ba9f2837024dfe6a7dcf30406aaa5c.jpg)  
그림 10.9 우리의 단순한 사기 네트워크. 이 경우 ReFeX는 노드 유형을 고려하지 않으므로 노드의 색상은 무시됩니다.

그림 10.9에 표시된 작은 그래프 데이터베이스에 ReFeX를 수동으로 적용해 보겠습니다. 단순화를 위해 무방향 및 비가중 그래프를 고려하며, 가지치기 단계는 수행하지 않습니다. 표 10.9는 각 노드의 차수를 보여 줍니다.

표 10.9 그림 10.9의 노드 차수
<table><tr><td>노드</td><td>차수</td></tr><tr><td>A</td><td>6</td></tr><tr><td>B</td><td>3</td></tr><tr><td>C</td><td>1</td></tr><tr><td>D</td><td>4</td></tr><tr><td>E</td><td>2</td></tr><tr><td>..•</td><td>··</td></tr></table>

이 결과는 10.1.1절의 결과와 동일합니다. 이는 방향 그래프가 아니므로, 진입 차수와 진출 차수를 계산할 수 없습니다.

다음 단계는 각 노드에 대한 에고넷 (egonet)을 고려하는 것입니다. 예를 들어 노드 A의 경우, 에고넷은 A 자체와 그 이웃인 B, G, H, I, O, T로 구성됩니다. 에고넷의 전체 노드 수는 7개입니다(A와 여섯 이웃). 내부 간선의 총수는 7개입니다(A를 이웃과 연결하는 여섯 간선에 G와 I 사이의 간선 하나를 더한 값). 에고넷의 진입/진출 간선 총수는 9개입니다(이는 A의 에고넷에 있는 노드를 외부 노드와 연결하는 간선입니다: B→C,D; G→D,F; H→K; I→K,N; O→P,Q). 표 10.10은 각 노드에 대한 총값을 나열합니다.

표 10.10 그림 10.9의 노드에 대한 에고넷 구조의 세부 정보
<table><tr><td rowspan=1 colspan=1>노드의 에고넷</td><td rowspan=1 colspan=1>노드 수</td><td rowspan=1 colspan=1>내부 간선 수</td><td rowspan=1 colspan=1>진입/진출 간선 수</td></tr><tr><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>7</td><td rowspan=6 colspan=1>95143·.·</td></tr><tr><td rowspan=1 colspan=1>B</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6</td></tr><tr><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>3</td><td rowspan=2 colspan=1>3…·</td></tr><tr><td rowspan=1 colspan=1>·.·</td><td rowspan=1 colspan=1>·…·</td></tr></table>

ReFeX의 마지막 단계에는 재귀적 특징 집계 (recursive feature aggregation)가 포함되며, 이는 더 넓은 네트워크 특성을 점진적으로 포착하는 다단계 과정입니다. 알고리즘은 각 반복에서 점점 더 먼 이웃으로부터 특징을 집계하여, 각 노드의 구조적 맥락에 대한 더 포괄적인 관점을 만듭니다. 그림 10.10은 예시 그래프를 사용하여 이 과정을 설명합니다.

![](images/acaf2efc0c26103c81ea111e1c977a483fc7d7a835d498f8be5234489cd20f7c.jpg)

$$
\begin{array}{rl}
& \mathrm{neighbor\_sum\_of\_sums} = \Theta_{\mathsf{WMS}} = \Theta_{\mathsf{MPS}} = \Theta_{\mathsf{MPS}} = \Theta_{\mathsf{MPS}} = \mathsf{T}_{\mathsf{MPS}} + \mathsf{T}_{\mathsf{MPS}} = \mathsf{T}_{\mathsf{MPS}} + \mathsf{T}_{\mathsf{MPS}} = \mathsf{T}_{\mathsf{MPS}} \\
& \qquad\quad \mathrm{neighbor\_mean\_of\_sums} = 6 \mathrm{nej} / \mathsf{G} = 11.5
\end{array}
$$

![](images/526d498ce6be24541cbeeb5bf9b67b6402c75307f79547bc8cfb4090ffc4a0e8.jpg)  
그림 10.10 ReFeX 과정의 몇 차례 반복을 보여 주는 예시입니다. 각 반복에서 각 노드는 자신의 값(차수, 이전 합계 등)을 이웃에게 전달하며, 이웃은 이를 집계합니다.

노드 A를 예로 들어 과정을 살펴보겠습니다.

1 첫 번째 반복:

– 노드 A에는 여섯 개의 이웃(H, I, G, B, T, O)이 있습니다.

– 지역 특징: degree(A) = 6

– SUM 연산자를 사용한 첫 번째 집계:

$$
\begin{array}{ll}
{{\mathrm{sum}(\mathrm{neighbor\_degrees}) = \mathrm{degree}(H) + \mathrm{degree}(I) + \mathrm{degree}(G) + \mathrm{degree}(B)}} \\
{{\qquad\quad + \mathrm{degree}(T) + \mathrm{degree}(O)}} \\
{{\mathrm{sum}(\mathrm{neighbor\_degrees}) = 2 + 4 + 4 + 3 + 1 + 9 = 17}}
\end{array}
$$

2 두 번째 반복:

– 첫 번째 반복에서 얻은 집계값을 사용합니다.

– 이제 각 이웃은 자신의 첫 번째 반복 합계(agg)를 가지고 있습니다.

– 두 번째 집계:

$$
\begin{array}{rl}
& {\operatorname{sum}(\mathrm{neighbor\_aggregates}) = \operatorname{agg}(B) + \operatorname{agg}(G) + \operatorname{agg}(H) + \operatorname{agg}(I) + \operatorname{agg}(O) + \operatorname{agg}(T)} \\
& {\operatorname{sum}(\mathrm{neighbor\_aggregates}) = 11 + 17 + 9 + 16 + 10 + 6 = 69}
\end{array}
$$

또한 ReFeX는 집계에 MEAN 연산자를 사용할 수도 있습니다. 노드 A의 경우 이는 다음과 같은 값을 제공합니다.

첫 번째 반복:

$$
\mathrm{mean}(\mathrm{neighbor\_degrees}) = (3 + 4 + 2 + 4 + 3 + 1) / 6 = 17 / 6 \approx 2.83
$$

두 번째 반복:

$$
\mathrm{mean}(\mathrm{neighbor\_aggregates}) = (11 + 17 + 9 + 16 + 10 + 6) / 6 = 69 / 6 = 11.5 \times 10^{-4}
$$

표 10.11은 이러한 반복 이후 노드 A에 대한 결과를 보여 줍니다(가지치기 전).

표 10.11 노드 A에 대해 계산된 특징
<table><tr><td>특징 유형</td><td>특징</td><td>값</td></tr><tr><td>로컬 특징</td><td>차수</td><td>6</td></tr><tr><td>에고넷 특징</td><td>에고넷의 엣지 수</td><td>7</td></tr><tr><td>재귀 특징, 첫 번째 반복</td><td>이웃 차수의 합</td><td>17</td></tr><tr><td></td><td>이웃 차수의 평균</td><td>2.83</td></tr><tr><td>재귀 특징, 두 번째 반복</td><td>이웃 합의 합</td><td>69</td></tr><tr><td></td><td>이웃 합의 평균</td><td>11.5</td></tr></table>

#### 10.3.2 코드로 ReFeX 자동 수행하기


알고리즘의 전체 구현은 이 책의 코드 저장소에서 확인할 수 있습니다.   
다음 목록은 코드에서 가장 관련성이 높은 부분을 보여 줍니다.

```python
Listing 10.18 ReFeX implementation (key parts)
import networkx as nx
import numpy as np
from collections import defaultdict from sklearn.preprocessing import StandardScaler Initializes ReFeX with an
iteration limit and
correlation threshold
class ReFeX:
def init__(self, max_iterations=2, correlation_threshold=0.95):
self.max_iterations = max_iterations
self.correlation_threshold = correlation_threshold <
def extract_features(self, G): Extracts basic node-level featu
features = self._extract_local_features(G) (degrees) and uses them to
initialize the feature matrix
egonet_features = self._extract_egonet_features(G)
Adds features based onegonet V features = np.column_stack((features, egonet_features)) for iteration in range(self.max_iterations): features iteratively Generates and adds recursive
properties new_features = self._generate_recursive_features(G, features)
features = np.column_stack((features, new_features)) <
features = self._prune_features(features) < Removes redundant
return features features based on
correlation
def _extract_local_features(self, G):
"""Extract local (node-level) features"""
n_nodes = G.number_of_nodes()
features = np.zeros((n_nodes, 3))
for idx, node in enumerate(G.nodes()):
#### Degree features
features[idx, 0] = G.degree(node)
#### In-degree and out-degree for directed graphs
if G.is_directed():
features[idx, 1] = G.in_degree(node) Computes local
features[idx, 2] = G.out_degree(node) features including
else: degree metrics
features[idx, 1] = features[idx, 2] = G.degree(node) <
return features
Calculates egonet-based
def _extract_egonet_features(self, G): < features for each node
n_nodes = G.number_of_nodes()
features = np.zeros((n_nodes, 3))
for idx, node in enumerate(G.nodes()):
ego = nx.ego_graph(G, node, radius=1)
features[idx, 0] = ego.number_of_nodes()
features[idx, 1] = ego.number_of_edges()
features[idx, 2] = nx.density(ego)
Generates recursive
features using sum and
return features
mean aggregations
def _generate_recursive_features(self, G, current_features): <
n_nodes = G.number_of_nodes()
```

```python
CHAPTER 10 Graph feature engineering: Manual and semiautomated approaches
n_features = current_features.shape[1]
new_features = np.zeros((n_nodes, n_features * 2))
for idx, node in enumerate(G.nodes()):
neighbors = list(G.neighbors(node))
if not neighbors:
continue
neighbor_feats =
current_features[[list(G.nodes()).index(n) for n in neighbors]]
new_features[idx, :n_features] = np.sum(neighbor_feats, axis=0)
new_features[idx, n_features:] = np.mean(neighbor_feats,
axis=0)
return new_features
Identifies highly correlated features
def _prune_features(self, features): < for removal, and removes them
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features) Standardizes features
corr_matrix = np.corrcoef(scaled_features.T) < Computes the
correlation matrix
to_remove = set()
Finds highly
for i in range(corr_matrix.shape[0]): <
correlated features
for j in range(i + 1, corr_matrix.shape[1]):
if abs(corr_matrix[i, j]) > self.correlation_threshold:
to_remove.add(j)
keep_features = list(set(range(features.shape[1])) - to_remove) <
return features[:, keep_features] ≤ Returns a pruned Keeps
feature matrix uncorrelated
features
```

#### 연습문제


이 책의 코드 저장소에는 Neo4j에서 사용할 수 있는 Hetionet 데이터베이스에 연결하는 전체 코드가 포함되어 있습니다. 직접 실행해 보십시오.

ReFeX는 자동 특징 추출 (automated feature extraction)을 향한 중요한 진전으로, 수동 특징 공학 (feature engineering)과 완전 자율적인 표현 학습 (representation learning) 기법 사이의 중요한 중간 지점을 차지합니다. 순수한 그래프 구조에 초점을 맞춘다는 점은 더 복잡한 접근법을 이해하기 위한 훌륭한 기반을 제공하며, 구조적 패턴만으로도 노드와 그 이웃의 의미 있는 특성을 포착할 수 있음을 보여 줍니다. ReFeX는 특징 추출을 자동화하지만 투명성과 해석 가능성을 유지합니다. 그 계산은 단계별로 추적하고 검증할 수 있으므로, 자신의 특징 공학 과정을 이해하고 검증해야 하는 실무자에게 매우 유용한 도구입니다. 또한 ReFeX의 결정론적 (deterministic) 특성은 일관성을 보장합니다. 동일한 입력은 항상 동일한 출력을 생성합니다. 이러한 예측 가능성은 재현성이 필수적인 운영 환경에서 특히 가치가 있습니다. 더 나아가 그래프 구조가 변경될 때 ReFeX는 특징 행렬 전체를 완전히 다시 생성하도록 요구하는 대신, 영향을 받은 특징만 선택적으로 재계산할 수 있게 합니다. 이러한 효율성은 ReFeX를 동적 그래프 환경에 특히 적합하게 만듭니다.

그러나 ReFeX에도 한계가 있습니다. 구조적 특징에 의존하기 때문에 노드 속성이나 엣지 유형을 직접 통합할 수 없습니다. 또한 가지치기 (pruning)가 계산 복잡도를 관리하는 데 도움이 되기는 하지만, 최적의 특징 선택을 보장하기 위해 때로는 사람의 감독이 필요합니다.

다음 장에서는 현대의 자율적 표현 학습 기법이 이러한 한계를 어떻게 해결하는지 살펴보겠습니다. 다만 그 과정에서 특정 응용에서 ReFeX를 가치 있게 만드는 해석 가능성과 결정론적 특성의 일부를 희생하게 됩니다. ReFeX의 강점과 한계를 이해하는 것은 이러한 더 발전된 방법들이 도입하는 혁신과 트레이드오프를 평가하는 데 필수적인 맥락을 제공할 것입니다.

#### 요약


그래프에서 수동 및 반수동 특징 공학 (feature engineering)은 ML 작업의 기반을 제공하며, 해석 가능성과 자동화 사이의 균형을 맞춥니다.

수동 특징 공학은 국소 지표와 전역 척도를 결합하여, 즉각적인 연결과 더 넓은 네트워크 패턴을 모두 포착하는 의미 있는 노드 표현을 생성합니다.

노드 지표를 결합하면 해석 가능한 의사결정 과정을 유지하면서 패턴을 식별하는 데 도움이 됩니다.

관계 표현은 노드 기반 조합(연결, 평균화, 거리) 또는 경로 기반 특징(메타경로, 구조적 패턴)을 통해 접근할 수 있습니다.

 노드 기반 관계 표현은 연결, 평균화, L1/L2 거리, 아다마르 곱 등 다양한 조합 방법을 사용할 수 있습니다.

경로 기반 특징은 메타경로 분석을 통해 노드 간의 구조적 패턴을 포착합니다.

DWPC (degree-weighted path count)는 노드 차수를 고려하면서 연결 관련성을 측정하는 정교한 접근 방식을 제공합니다.

도메인 전문성은 지표 선택부터 결과 검증에 이르기까지 노드 표현과 관계 표현 모두에서 특징 공학을 안내합니다.

ReFeX와 같은 반자동 접근 방식은 해석 가능한 특징을 자동으로 생성하면서 도메인 지식을 통합할 수 있는 선택지를 보존함으로써 중간 지점을 제공합니다.

 수동 접근 방식과 반자동 접근 방식 중 무엇을 선택할지는 해석 가능성 요구, 계산 자원, 사용 가능한 도메인 전문성에 따라 달라집니다.

특징 가지치기와 선택은 두 접근 방식 모두에서 여전히 필수적이며, 특정 ML 작업에 대한 상관성과 관련성을 신중하게 고려해야 합니다.