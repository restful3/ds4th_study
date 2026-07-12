# Graph Flashcards

## Card 1

**Front:** 그래프 표현 학습(GRL)이 수동 특징 공학(Manual feature engineering)에 비해 가지는 주요 장점은 무엇입니까?

**Back:** 딥러닝 기법을 통해 그래프 구조와 노드 속성으로부터 최적의 표현(임베딩)을 자동으로 학습합니다.

---

## Card 2

**Front:** 그래프 임베딩(Graph embedding)의 근본적인 변환 과정은 무엇입니까?

**Back:** 불연속적인 그래프 구조를 그래프의 구조적, 의미적 특성을 보존하는 밀집 벡터(Dense vector) 표현으로 변환합니다.

---

## Card 3

**Front:** GRL의 발전 과정 중 제1세대에 해당하는 전통적인 그래프 임베딩의 핵심 접근 방식은?

**Back:** 전통적인 수학 및 컴퓨터 과학에 기반하여 복잡한 구조를 단순화하는 고전적 차원 축소(Dimensionality reduction)에 집중했습니다.

---

## Card 4

**Front:** 자연어 처리의 word2vec 기술을 그래프 노드에 적용하여 의미 있는 관계를 포착하기 시작한 GRL의 세대는?

**Back:** 제2세대 (Node2Vec 등이 대표적임).

---

## Card 5

**Front:** 그래프 신경망(GNN)을 통해 원시 데이터로부터 특징을 자동으로 학습하는 현재의 GRL 세대는?

**Back:** 제3세대.

---

## Card 6

**Front:** 그래프 데이터가 이미지나 텍스트와 같은 정형 데이터와 구별되는 기하학적 특징은?

**Back:** 각 노드마다 연결된 이웃의 수와 구조가 다른 불규칙성(Irregularity)을 가집니다.

---

## Card 7

**Front:** 유클리드 공간(Euclidean space)에서 두 지점 사이의 거리를 계산하는 표준 공식은?

**Back:** 각 좌표 차이의 제곱합의 제곱근(표준 유클리드 거리 공식)을 사용합니다.

---

## Card 8

**Front:** 기업 조직도나 생물학적 분류와 같은 계층적 구조(Hierarchical structure)를 표현하는 데 가장 효율적인 비유클리드 공간은?

**Back:** 쌍곡선 공간(Hyperbolic space).

---

## Card 9

**Front:** 쌍곡선 공간(Hyperbolic space)에서 중심에서 멀어질수록 가용 공간이 늘어나는 방식은?

**Back:** 지수적(Exponentially)으로 증가합니다.

---

## Card 10

**Front:** 네트워크 전체에서 노드의 절대적 위치나 중앙성(Centrality)을 보존하는 데 집중하는 임베딩 유형은?

**Back:** 위치적 임베딩(Positional embeddings).

---

## Card 11

**Front:** 노드의 국부적인 연결 패턴이나 역할(Role)의 유사성을 보존하는 데 집중하는 임베딩 유형은?

**Back:** 구조적 임베딩(Structural embeddings).

---

## Card 12

**Front:** GNN이 위치적 임베딩보다 구조적 임베딩에 더 적합한 이유는?

**Back:** GNN은 이웃 노드로부터 정보를 집계하여 국부적인 연결 패턴을 인식하고 보존하는 데 능숙하기 때문입니다.

---

## Card 13

**Front:** 고정된 노드 세트에 대해 임베딩을 직접 최적화하며, 새로운 노드가 추가될 경우 다시 학습해야 하는 방식은?

**Back:** 변환적 학습(Transductive learning).

---

## Card 14

**Front:** 학습 과정에서 보지 못한 새로운 노드에도 적용할 수 있는 일반적인 규칙(매개변수 매핑)을 학습하는 방식은?

**Back:** 귀납적 학습(Inductive learning).

---

## Card 15

**Front:** 동적인 그래프 환경(예: 전자상거래 신규 사용자 추천)에서 반드시 필요한 학습 방식은?

**Back:** 귀납적 학습(Inductive learning).

---

## Card 16

**Front:** 인코더-디코더 모델에서 인코더(Encoder)의 입력값 두 가지는?

**Back:** 그래프의 인접 행렬(Adjacency matrix)로 표현된 구조와 노드별 특징(Features)입니다.

---

## Card 17

**Front:** 인코더-디코더 모델에서 디코더(Decoder)가 수행하는 핵심 작업은?

**Back:** 생성된 임베딩을 사용하여 원래 그래프의 연결 상태나 이웃 유사성 등의 속성을 재구성(Reconstruct)합니다.

---

## Card 18

**Front:** Node2Vec 알고리즘에서 네트워크의 국부적 구조와 광범위한 구조를 유연하게 탐색하기 위해 사용하는 기법은?

**Back:** 무작위 보행(Random walk).

---

## Card 19

**Front:** Node2Vec 디코더가 두 노드 사이의 관계를 예측할 때 확률을 계산하기 위해 사용하는 함수는?

**Back:** 소프트맥스(Softmax) 함수.

---

## Card 20

**Front:** 각 노드를 벡터에 직접 매핑하는 룩업 테이블(Lookup table) 방식을 사용하는 임베딩 유형은?

**Back:** 샤로 임베딩(Shallow embeddings).

---

## Card 21

**Front:** 샤로 임베딩(Shallow embeddings)의 치명적인 한계 중 하나로, 노드 수에 따라 매개변수가 비례하여 증가하는 성질은?

**Back:** 매개변수 비효율성(Parameter inefficiency).

---

## Card 22

**Front:** 샤로 임베딩이 새로운 노드에 대한 임베딩을 즉시 생성하지 못하는 이유는?

**Back:** 학습된 룩업 테이블에 포함되지 않은 노드에 대해서는 대응하는 벡터가 존재하지 않기 때문입니다(Transductive 성질).

---

## Card 23

**Front:** 지식 그래프(KG) 임베딩에서 단순히 연결 여부뿐만 아니라 무엇을 추가로 인코딩해야 합니까?

**Back:** 엔티티 간의 관계 유형(Relationship type)을 인코딩해야 합니다.

---

## Card 24

**Front:** 지식 그래프 임베딩 학습 시 계산 효율성을 높이기 위해 실제 관계가 없는 노드 쌍을 샘플링하는 기법은?

**Back:** 부정적 샘플링(Negative sampling).

---

## Card 25

**Front:** 지식 그래프 임베딩의 손실 함수에서 실제 관계에 높은 점수를 주고 가짜 관계에 낮은 점수를 주도록 조절하는 가중치 매개변수는?

**Back:** 감마($\gamma$).

---

## Card 26

**Front:** 부정적 샘플링 시 무작위 추출 대신 의미적으로 적절한 유형의 노드만 샘플링하는 기법은?

**Back:** 유형 제약 샘플링(Type-constrained sampling).

---

## Card 27

**Front:** 관계를 엔티티 벡터 간의 '변환(Translation)'으로 취급하여 $h + r \approx t$가 성립하도록 학습하는 모델은?

**Back:** TransE.

---

## Card 28

**Front:** 지식 그래프에서 비대칭적 관계(Asymmetric relations)를 처리하기 위해 복소수(Complex numbers)를 사용하는 모델은?

**Back:** ComplEx.

---

## Card 29

**Front:** GNN의 메시지 전달(Message passing) 프레임워크에서 매 라운드마다 각 노드가 수행하는 세 단계는?

**Back:** 이웃 메시지 수집, 메시지 처리(정보 추출), 자신의 표현 업데이트.

---

## Card 30

**Front:** GNN에서 $k$번의 메시지 전달 단계를 거친 후 노드 $u$의 표현이 포함하게 되는 정보의 범위는?

**Back:** 노드 $u$로부터 $k$홉($k$-hop) 거리 내에 있는 이웃들의 정보입니다.

---

## Card 31

**Front:** 기본적인 GNN 업데이트 식 $h_u^{(k)} = \sigma(W_{self}^{(k)} h_u^{(k-1)} + W_{neigh}^{(k)} \sum_{v \in N(u)} h_v^{(k-1)} + b^{(k)})$에서 $W$는 무엇을 의미합니까?

**Back:** 학습 가능한 가중치 행렬(Trainable parameter matrices)입니다.

---

## Card 32

**Front:** 메시지 전달 과정에서 자기 자신을 이웃의 일부로 취급하여 집계와 업데이트를 단순화하는 기법은?

**Back:** 셀프 루프(Self-loops) 추가.

---

## Card 33

**Front:** 노드마다 이웃의 수가 다른 문제를 해결하기 위해 이웃 정보를 합산하는 대신 평균을 내는 정규화 방식은?

**Back:** 평균 정규화(Mean normalization).

---

## Card 34

**Front:** Kipf와 Welling의 GCN 모델에서 사용하는, 송신 노드와 수신 노드의 차수(Degree)를 모두 고려하는 정규화 방식은?

**Back:** 대칭 정규화(Symmetric normalization).

---

## Card 35

**Front:** 대칭 정규화(Symmetric normalization)를 적용했을 때 인용 네트워크에서 얻을 수 있는 효과는?

**Back:** 수천 번 인용된 논문의 영향력이 너무 압도적이지 않도록 감소시켜 안정적인 학습을 돕습니다.

---

## Card 36

**Front:** 모든 이웃을 동일하게 대우하는 대신, 각 이웃의 중요도에 따라 서로 다른 가중치를 부여하는 기법은?

**Back:** 어텐션 메커니즘(Attention mechanisms).

---

## Card 37

**Front:** GAT(Graph Attention Network)에서 노드 $u$가 이웃 $v$로부터 받는 정보의 중요도를 나타내는 계수는?

**Back:** 어텐션 가중치($\alpha_{u,v}$).

---

## Card 38

**Front:** 여러 개의 독립적인 어텐션 메커니즘을 병렬로 운영하여 이웃 관계의 다양한 측면을 포착하는 기법은?

**Back:** 멀티헤드 어텐션(Multi-head attention).

---

## Card 39

**Front:** 트랜스포머(Transformer) 아키텍처의 어텐션 점수를 계산하기 위해 사용되는 세 가지 요소는?

**Back:** 쿼리(Query, Q), 키(Key, K), 값(Value, V).

---

## Card 40

**Front:** GNN 모델에서 노드의 차수(Degree)와 같은 구조적 정보를 명시적으로 입력하기 위해 사용하는 기법은?

**Back:** 구조적 인코딩(Structural encoding).

---

## Card 41

**Front:** GraphSAGE에서 이전 층의 노드 상태와 집계된 이웃 정보를 결합할 때 사용하는 방식은?

**Back:** 연결(Concatenation).

---

## Card 42

**Front:** GNN의 업데이트 단계에서 스킵 연결(Skip connections)이 기여하는 주요 기능은?

**Back:** 기울기 소실 문제를 완화하여 더 깊은 신경망 아키텍처를 학습할 수 있게 합니다.

---

## Card 43

**Front:** RNN에서 영감을 얻어, 기존 노드 표현을 얼마나 유지하고 새로운 정보를 얼마나 받아들일지 결정하는 업데이트 방식은?

**Back:** 게이트 업데이트(Gated updates).

---

## Card 44

**Front:** 여러 층(layer)에서 생성된 노드 표현들을 적응적으로 결합하여 국부적 정보와 전역적 정보를 동시에 활용하는 네트워크 구조는?

**Back:** 점핑 나리지 네트워크(Jumping Knowledge Networks).

---

## Card 45

**Front:** LLM이 노드나 엣지의 텍스트 정보를 수치형 벡터로 변환하여 GNN에 전달하는 역할 모델은?

**Back:** 인코더(Encoder)로서의 LLM.

---

## Card 46

**Front:** GNN이 구조적 정보를 처리하고 LLM이 최종적인 복잡한 추론이나 자연어 답변을 생성하는 역할 모델은?

**Back:** 예측기(Predictor)로서의 LLM.

---

## Card 47

**Front:** 대조 학습(Contrastive learning) 등을 통해 GNN의 구조 정보와 LLM의 텍스트 정보를 일치시키는 방식은?

**Back:** 정렬기(Aligner)로서의 LLM.

---

## Card 48

**Front:** 그래프 신경망에서 노드 표현이 층을 거듭할수록 서로 너무 비슷해져 구분이 불가능해지는 현상은?

**Back:** 오버스무딩(Over-smoothing) 문제.

---

## Card 49

**Front:** 지식 그래프 임베딩에서 $L = -\log(\sigma(DEC(z_u, \tau, z_v)))$ 항이 의미하는 바는?

**Back:** 실제로 존재하는 관계(양성 샘플)에 대해 높은 확률 점수를 부여하도록 장려합니다.

---
