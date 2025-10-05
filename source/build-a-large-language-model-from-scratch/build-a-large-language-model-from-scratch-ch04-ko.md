# 4 텍스트 생성을 위한 GPT 모델 직접 구현하기

## 이 장에서 다루는 내용

- 사람과 유사한 텍스트를 생성하도록 학습할 수 있는 GPT 계열 대규모 언어 모델(LLM) 구현
- 심층 신경망 학습을 안정화하기 위한 층 정규화(layer normalization)
- 심층 신경망에 쇼트컷(스킵, 잔차) 연결 추가
- 다양한 크기의 GPT 모델을 만들기 위한 트랜스포머 블록 구현
- GPT 모델의 파라미터 수와 저장 공간 계산

앞 장에서는 LLM의 핵심 구성 요소 가운데 하나인 멀티 헤드 어텐션을 직접 구현해 보았다. 이제 남은 구성 요소를 모두 완성하여 GPT와 유사한 모델을 조립하고, 다음 장에서 사람처럼 텍스트를 생성할 수 있도록 학습시킬 것이다.

그림 4.1에 나와 있듯이 LLM 아키텍처는 여러 구성 요소로 이루어진다. 이번 장에서는 전체 구조를 먼저 살펴본 뒤, 개별 구성 요소를 자세히 구현한다.

<img src="./image/fig_04_01.png" width=800>

그림 4.1 LLM을 코딩하는 세 단계. 이번 장은 1단계의 세 번째 단계, 즉 LLM 아키텍처 구현에 초점을 맞춘다.

# 4.1 LLM 아키텍처 코딩하기

GPT(Generative Pretrained Transformer)는 단어(토큰)를 한 번에 하나씩 생성하는 대규모 딥러닝 모델이다. 모델 크기가 매우 크지만, 구성 요소가 반복적으로 사용되기 때문에 생각보다 단순하다. 그림 4.2는 GPT 스타일 LLM의 상위 구조를 보여 준다.

우리는 이미 입력 토크나이징, 임베딩, 그리고 이전 장에서 마스킹된 멀티 헤드 어텐션 모듈을 다루었다. 이제 트랜스포머 블록을 포함한 GPT 모델의 핵심 구조를 구현해 보고, 다음 장에서 이를 학습시켜 자연스러운 텍스트를 생성하도록 만들 것이다.

앞선 예제에서는 설명을 쉽게 하기 위해 작은 임베딩 차원을 사용했지만, 이번에는 GPT-2 중 가장 작은 1억 2,400만 개 파라미터 모델을 목표로 한다. 이는 Radford 외(https://mng.bz/yoBq)에서 발표한 “Language Models Are Unsupervised Multitask Learners”에 소개된 모델이며, 논문에는 1억 1,700만 개라고 기재되어 있지만 이후 수정되었다. 6장에서 우리는 더 큰 GPT-2 모델(3억 4,500만, 7억 6,200만, 15억 4,200만 파라미터)에 대응하기 위해 사전학습된 가중치를 불러오는 방법을 살펴볼 것이다.

딥러닝과 GPT 같은 LLM에서 "파라미터"란 학습 가능한 모델 가중치를 의미한다. 파라미터는 학습 과정에서 손실 함수를 최소화하도록 조정되는 내부 변수로, 모델이 학습 데이터에서 패턴을 학습할 수 있게 해 준다.

<img src="./image/fig_04_02.png" width=800>

그림 4.2 GPT 모델 구조. 입력 임베딩 층 외에 이전 장에서 구현한 마스킹 멀티 헤드 어텐션을 포함하는 하나 이상의 트랜스포머 블록으로 이루어진다.

예를 들어, $(2,048 	imes 2,048)$ 크기의 가중치 행렬로 표현되는 신경망 층이 있다면, 행과 열의 곱인 2,048 × 2,048 = 4,194,304개의 파라미터를 가진다.

# GPT-2 vs. GPT-3

이번 장에서는 OpenAI가 공개한 GPT-2의 사전학습 가중치를 활용한다. GPT-3는 구조적으로 GPT-2와 동일하지만, 파라미터 수를 15억에서 1,750억으로 늘렸고 훨씬 더 많은 데이터를 사용해 학습했다. 현재 GPT-3의 가중치는 공개되어 있지 않다. 또한 GPT-2는 노트북에서도 실행이 가능하지만, GPT-3는 GPU 클러스터가 필요하다. Lambda Labs에 따르면 V100 데이터센터 GPU 한 대로 GPT-3를 학습하려면 355년, 소비자용 RTX 8000으로는 665년이 걸린다고 한다.

작은 GPT-2 모델의 설정은 다음과 같은 파이썬 딕셔너리로 정의한다.

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257, # 어휘 집합 크기
    "context_length": 1024, # 컨텍스트 길이
    "emb_dim": 768, # 임베딩 차원
    "n_heads": 12, # 어텐션 헤드 수
    "n_layers": 12, # 트랜스포머 블록 수
    "drop_rate": 0.1, # 드롭아웃 비율
    "qkv_bias": False # QKV 선형층의 편향 사용 여부
}
```

- `vocab_size`는 BPE 토크나이저(2장에서 다룸)가 사용하는 50,257개의 토큰을 의미한다.
- `context_length`는 위치 임베딩으로 표현할 수 있는 최대 토큰 수를 나타낸다.
- `emb_dim`은 각 토큰을 768차원 벡터로 변환하는 임베딩 차원이다.
- `n_heads`는 멀티 헤드 어텐션의 헤드 수(3장에서 구현).
- `n_layers`는 모델에 쌓을 트랜스포머 블록의 수.
- `drop_rate`는 드롭아웃 비율(0.1이면 은닉 유닛의 10%를 무작위로 꺼서 과적합 방지, 3장 참조).
- `qkv_bias`는 멀티 헤드 어텐션의 쿼리·키·값 선형층에 편향을 추가할지 여부다. 최신 LLM에서는 보통 편향을 사용하지 않으므로 False로 두었지만, 6장에서 OpenAI의 GPT-2 가중치를 로드할 때 다시 다룬다.

이 설정을 사용해 그림 4.3과 같은 GPT 뼈대(DummyGPTModel)를 구현한다. 이 모델은 전체 구조를 조망하고 이후 어떤 구성 요소를 구현해야 하는지 확인하는 데 도움이 된다.

<img src="./image/fig_04_03.png" width=800>

그림 4.3 GPT 아키텍처를 구현하는 순서. 먼저 GPT 백본(더미 모델)을 만들고, 이후 필요한 핵심 구성 요소를 각각 구현한 뒤 최종 트랜스포머 블록에 통합한다.

# 코드 4.1 GPT 모델 뼈대 클래스

```python
import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

트랜스포머 블록용 자리 표시자

```
class DummyTransformerBlock(nn.Module):  # 나중에 실제 구현으로 교체
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x
```

레이어 정규화 자리 표시자

```
class DummyLayerNorm(nn.Module):  # 나중에 실제 LayerNorm으로 교체
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
```

위 DummyGPTModel은 PyTorch의 `nn.Module`을 사용해 간단한 GPT 모형을 정의한 것이다. 토큰 및 위치 임베딩, 드롭아웃, 더미 트랜스포머 블록, 더미 레이어 정규화, 그리고 최종 선형 출력층(`out_head`)으로 구성된다. 설정 값은 위에서 정의한 `GPT_CONFIG_124M` 같은 딕셔너리에서 전달된다.

forward 메서드는 입력 토큰 인덱스를 받아 토큰/위치 임베딩을 더한 뒤 드롭아웃을 적용하고, 트랜스포머 블록을 거쳐 레이어 정규화를 수행하며, 마지막으로 선형층을 통해 단어 분포(logits)를 출력한다.

코드 4.1은 이미 동작하지만, 트랜스포머 블록과 레이어 정규화는 더미 클래스라 이후에 실제 구현으로 대체할 것이다.

이제 데이터를 준비하고 GPT 모델이 어떻게 동작하는지 간단히 살펴보자. 2장에서 만든 토크나이저를 사용하여 두 문장의 배치를 토크나이징한다.

```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print (batch)
```

<img src="./image/fig_04_04.png" width=800>

그림 4.4 입력 데이터가 토크나이징, 임베딩을 거쳐 GPT 모델로 전달되는 큰 흐름. DummyGPTModel에서는 토큰 임베딩이 모델 내부에서 처리된다. LLM에서는 입력 임베딩 차원과 출력 차원이 보통 동일하다. 여기서 출력 임베딩은 컨텍스트 벡터(3장 참조)를 의미한다.

두 문장을 토크나이징한 결과는 다음과 같다.

```
tensor([[6109, 3626, 6100, 345],
    [6109, 1110, 6622, 257]])
```

첫 번째 행은 첫 번째 문장, 두 번째 행은 두 번째 문장을 의미한다.

이제 1억 2,400만 개 파라미터를 가지는 DummyGPTModel 인스턴스를 만들고 방금 만든 배치를 넣어 보자.

```python
torch.manual_seed(123)  # for reproducible initial weights
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print(logits)
```

모델 출력(로짓)은 다음과 같다.

```
Output shape: torch.Size([2, 4, 50257])
tensor([[[-1.2034, 0.3201, -0.7130, ..., -1.5548, -0.2390, -0.4667],
    [-0.1192, 0.4539, -0.4432, ..., 0.2392, 1.3469, 1.2430],
    [ 0.5307, 1.6720, -0.4695, ..., 1.1966, 0.0111, 0.5835],
    [ 0.0139, 1.6755, -0.3388, ..., 1.1586, -0.0435, -1.0400]],
    [[-1.0908, 0.1798, -0.9484, ..., -1.6047, 0.2439, -0.4530],
    [-0.7860, 0.5581, -0.0610, ..., 0.4835, -0.0077, 1.6621],
    [ 0.3567, 1.2698, -0.6398, ..., -0.0162, -0.1296, 0.3717],
    [-0.2407, -0.7349, -0.5102, ..., 2.0057, -0.3694, 0.1814]]],
    grad_fn=<UnsafeViewBackward0>]
```

출력 텐서는 두 개의 샘플(두 문장)에 대응하는 두 행을 가진다. 각 문장은 4개의 토큰으로 구성되어 있으며, 각 토큰은 토크나이저 어휘 크기인 50,257차원 벡터이다.

각 차원은 어휘의 특정 토큰에 대응하므로 50,257차원이 된다. 이후 장에서 후처리 코드를 구현할 때 이 벡터를 다시 토큰 ID로 변환하고, 이를 텍스트로 디코딩한다.

이제 GPT 아키텍처의 전체 흐름을 확인했으므로, 개별 구성 요소를 구현해 DummyLayerNorm을 대체할 실제 레이어 정규화부터 살펴보자.

# 4.2 층 정규화로 활성값 정규화하기

레이어 수가 많은 딥 신경망을 학습할 때 기울기가 사라지거나 폭발하는 문제가 발생할 수 있다. 이는 학습을 불안정하게 만들고 가중치가 잘 조정되지 않아 모델이 데이터에서 패턴을 충분히 학습하지 못하게 한다.

(기울기 개념이 낯설다면 부록 A.4를 참고하면 도움이 된다.)

레이어 정규화는 이러한 문제를 완화하기 위해 각 층의 출력(활성값)이 평균 0, 분산 1이 되도록 정규화한다. 이렇게 하면 학습이 더 빠르고 안정적으로 진행된다. GPT-2와 최신 트랜스포머 아키텍처에서는 멀티 헤드 어텐션 앞뒤와 최종 출력층 앞에 레이어 정규화를 사용한다. 그림 4.5는 레이어 정규화의 개념을 시각적으로 보여 준다.

<img src="./image/fig_04_05.png" width=800>

그림 4.5 레이어 정규화 개념. 층의 여섯 개 출력(활성값)을 평균 0, 분산 1로 정규화한다.

그림 4.5의 예를 코드로 구현해 보자. 입력으로 다섯 개 특징을 가진 두 개의 샘플을 만들고, 5→6 선형층과 ReLU를 적용한다.

```python
torch.manual_seed(123)  # for reproducible initial weights
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print (out)
```

두 개의 5차원 샘플 생성

출력은 다음과 같다.

```
tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
    [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]],
    grad_fn=<ReluBackward0>)
```

첫 번째 행은 첫 번째 입력 샘플의 결과, 두 번째 행은 두 번째 입력 샘플의 결과다. 이 층은 선형층 뒤에 ReLU를 사용해 음수 입력을 0으로 클램핑하므로 결과에 음수가 없다.

레이어 정규화를 적용하기 전 평균과 분산을 살펴보자.

```python
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean:
", mean)
print("Variance:
", var)
```

결과는 다음과 같다.

```
Mean:
    tensor([[0.1324],
        [0.2170]], grad_fn=<MeanBackward1>)
Variance:
    tensor([[0.0231],
        [0.0398]], grad_fn=<VarBackward0>)
```

각 행의 평균과 분산이 출력되었다. `keepdim=True`를 사용해 차원을 유지하면 이후 계산이 편하다.

레이어 정규화를 적용하면 다음과 같다.

```python
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Normalized layer outputs:
", out_norm)
print("Mean:
", mean)
print("Variance:
", var)
```

정규화된 결과는 평균 0, 분산 1을 갖는다.

```
Normalized layer outputs:
    tensor([[ 0.6159, 1.4126, -0.8719, 0.5872, -0.8719, -0.8719],
        [-0.0189, 0.1121, -1.0876, 1.5173, 0.5647, -1.0876]],
        grad_fn=<DivBackward0>)
Mean:
    tensor([[-5.9605e-08],
        [1.9868e-08]], grad_fn=<MeanBackward1>)
Variance:
    tensor([[1.],
        [1.]], grad_fn=<VarBackward0>)
```

소수점 오차는 매우 작다.

`torch.set_printoptions(sci_mode=False)`를 사용하면 지수 표기 없이 값을 볼 수 있다.

```python
torch.set_printoptions(sci_mode=False)
print("Mean:
", mean)
print("Variance:
", var)
```

```
Mean:
    tensor([[ 0.0000],
    [ 0.0000]], grad_fn=<MeanBackward1>)
Variance:
    tensor([[1.],
        [1.]], grad_fn=<VarBackward0>)
```

이제 레이어 정규화를 PyTorch 모듈로 깔끔하게 구현해 보자.

## 코드 4.2 레이어 정규화 클래스

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones (emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

이 구현은 입력 텐서의 마지막 차원(임베딩 차원)을 정규화한다. `eps`는 분모가 0이 되는 것을 방지하기 위한 작은 값이다. `scale`과 `shift`는 학습 가능한 파라미터로, 학습 과정에서 모델이 더 나은 표현을 찾을 수 있도록 돕는다.

### 편향(biased) 분산

위 코드에서 `unbiased=False`를 사용했는데, 이는 분산 계산에서 표본 분산 보정(Bessel 보정)을 사용하지 않음을 의미한다. $n$개의 값에 대해 분산을 계산할 때 $n-1$이 아닌 $n$으로 나누는 방식이다. 임베딩 차원이 매우 크기 때문에 이 차이는 거의 영향을 미치지 않으며, GPT-2의 TensorFlow 구현과 호환되도록 하기 위해 동일한 설정을 사용했다.

```python
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("Mean:
", mean)
print("Variance:
", var)
```

출력은 다음과 같다.

```
Mean:
    tensor([[ -0.0000],
        [ 0.0000]], grad_fn=<MeanBackward1>)
Variance:
    tensor([[1.0000],
        [1.0000]], grad_fn=<VarBackward0>)
```

<img src="./image/fig_04_07.png" width=800>

그림 4.7 GPT 아키텍처를 구성하는 빌딩 블록. GPT 백본과 레이어 정규화를 구현했고, 이제 GELU 활성 함수와 피드포워드 네트워크를 다룰 차례다.

### 레이어 정규화 vs 배치 정규화

배치 정규화는 배치 차원 전체를 정규화하지만, 레이어 정규화는 특성 차원 전체를 정규화한다. LLM은 하드웨어나 사용 환경에 따라 배치 크기가 제약받는 경우가 많다. 레이어 정규화는 각 샘플을 독립적으로 정규화하므로 분산 학습이나 리소스가 한정된 환경에서도 안정적으로 동작한다.

# 4.3 GELU 활성함수를 사용하는 피드포워드 네트워크 구현

이번에는 트랜스포머 블록에서 사용되는 작은 신경망 구성 요소인 피드포워드 네트워크를 구현한다. 먼저 LLM에서 자주 사용하는 GELU(Gaussian Error Linear Unit) 활성 함수를 살펴보자.

```python
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

relu = nn.ReLU()
gel = GELU()

x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
plt.show()
```

<img src="./image/fig_04_08.png" width=800>

그림 4.8 GELU와 ReLU 활성 함수 출력 비교. x축은 입력, y축은 함수 출력이다.

GELU는 음수 입력에 대해서도 작은 비선을형 값을 유지하여 최적화가 더 원활하게 진행되도록 돕는다.

이제 이 GELU를 이용해 피드포워드 모듈을 구현해 보자.

## 코드 4.4 피드포워드 네트워크 모듈

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    def forward(self, x):
        return self.layers(x)
```

피드포워드 모듈은 두 개의 선형층과 GELU로 구성된다. GPT_CONFIG_124M에서 `emb_dim`은 768이므로 입력과 출력은 768차원을 유지하지만, 내부에서는 4배 확장(3,072차원) 후 다시 축소한다.

```python
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)
```

```python
# 배치 차원이 2인 예제 입력 생성
```

출력과 입력의 형태는 동일하다.

<img src="./image/fig_04_09.png" width=800>

그림 4.9 피드포워드 네트워크의 연결 구조. 첫 번째 선형층은 768차원을 3,072로 확장하고, 두 번째 선형층은 다시 768차원으로 축소한다.

<img src="./image/fig_04_10.png" width=800>

그림 4.10 피드포워드 네트워크 내부 확장과 축소 과정을 시각화. 다양한 배치 크기와 토큰 수를 처리할 수 있지만 임베딩 차원은 초기화 시 고정된다.

<img src="./image/fig_04_11.png" width=800>

그림 4.11 GPT 아키텍처 구축을 위한 빌딩 블록. 검은 체크는 구현이 완료된 부분을 나타낸다.

# 4.4 쇼트컷 연결 추가하기

쇼트컷(스킵, 잔차) 연결은 깊은 네트워크에서 발생하는 기울기 소실 문제를 완화하기 위해 제안되었다. 출력에 입력을 더해줌으로써 역전파 경로를 단축하고 기울기가 더 잘 전달되도록 한다(그림 4.12).

<img src="./image/fig_04_12.png" width=800>

그림 4.12 다섯 개 층으로 이루어진 깊은 신경망에서 쇼트컷 연결이 있는 경우와 없는 경우의 비교. 각 층에서 평균 절대 기울기를 비교해 볼 수 있다.

## 코드 4.5 쇼트컷 연결을 시연하는 신경망

```python
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
        ])
    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x
```

각 층을 거치면서 입력과 출력이 같은 크기일 때만 잔차를 더해 준다.

```python
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)  # for reproducible initial weights
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)
```

다음으로, 역전파 시 기울기를 출력하는 헬퍼 함수를 만들자.

```python
def print_gradients(model, x):
    output = model(x)  # Forward pass
    target = torch.tensor([[0.]])
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    loss.backward()  # Backward pass to calculate the gradients
    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
```

이 함수는 손실 함수로 MSE를 사용해 출력과 타깃(여기서는 0)을 비교하고, 역전파를 통해 각 파라미터의 기울기를 계산한 뒤, 각 선형층의 평균 절대 기울기를 출력한다.

```python
print_gradients(model_without_shortcut, sample_input)
```

```
layers.0.0.weight has gradient mean of 0.00020173587836325169
layers.1.0.weight has gradient mean of 0.0001201116101583466
layers.2.0.weight has gradient mean of 0.0007152041653171182
layers.3.0.weight has gradient mean of 0.001398873864673078
layers.4.0.weight has gradient mean of 0.005049646366387606
```

앞쪽 층으로 갈수록 기울기가 급격히 작아지는 기울기 소실 문제가 보인다.

```python
torch.manual_seed(123)  # for reproducible initial weights
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
)
print_gradients(model_with_shortcut, sample_input)
```

```
layers.0.0.weight has gradient mean of 0.22169792652130127
layers.1.0.weight has gradient mean of 0.20694105327129364
layers.2.0.weight has gradient mean of 0.32896995544433594
layers.3.0.weight has gradient mean of 0.2665732502937317
layers.4.0.weight has gradient mean of 1.3258541822433472
```

마지막 층의 기울기는 여전히 크지만, 앞쪽 층들이 거의 같은 크기를 유지해 기울기 소실이 발생하지 않는다. LLM 같은 매우 깊은 모델에서 잔차 연결이 필수적인 이유다.

# 4.5 트랜스포머 블록에서 어텐션과 선형층 연결하기

이번에는 GPT와 다른 LLM의 핵심 구성 요소인 트랜스포머 블록을 구현한다. GPT-2 소형 모델에서는 이 블록을 12번 반복한다. 멀티 헤드 어텐션, 레이어 정규화, 드롭아웃, 피드포워드 네트워크를 함께 사용한다.

<img src="./image/fig_04_13.png" width=800>

그림 4.13 트랜스포머 블록 구조. 각 토큰은 768차원 벡터로 표현되어 있으며, 어텐션과 피드포워드 네트워크를 거치면서 같은 형태를 유지한다.

```python
from chapter03 import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
```

레이어 정규화를 어텐션과 피드포워드 앞에 두는 방식(프리-레이어 노름)은 최신 LLM에서 일반적이다.

```python
torch.manual_seed(123)  # for reproducible initial weights
x = torch.rand(2, 4, 768)  # Sample input of shape [batch_size, num_tokens, emb_dim]
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)
```

출력 형태는 입력과 동일하다.

<img src="./image/fig_04_14.png" width=800>

그림 4.14 GPT 아키텍처를 구성하는 빌딩 블록. 검은 체크는 완료된 부분.

# 4.6 GPT 모델 코딩하기

이제 DummyTransformerBlock과 DummyLayerNorm을 실제 구현으로 교체하여 GPT-2 소형 모델을 완성한다. 5장에서는 이 모델을 사전학습시키고, 6장에서는 OpenAI가 공개한 가중치를 불러와 사용할 것이다.

<img src="./image/fig_04_15.png" width=800>

그림 4.15 GPT 모델의 전체 구조. 토크나이징한 텍스트가 토큰 임베딩과 위치 임베딩을 거쳐 여러 개의 트랜스포머 블록을 통과하고, 마지막에 레이어 노름과 선형 출력층을 거쳐 다음 토큰의 확률 분포(logits)를 생성한다.

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

트랜스포머 블록을 스택으로 구성했기 때문에 GPTModel 클래스는 비교적 간결하다. `__init__` 메서드는 임베딩 층, 트랜스포머 블록, 레이어 노름, 출력 선형층을 초기화한다. `forward`는 입력 토큰을 임베딩하고 위치 정보를 더한 후, 트랜스포머 블록과 레이어 노름을 거쳐 logits을 반환한다.

```python
torch.manual_seed(123)  # for reproducible initial weights
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("Input batch:
", batch)
print("
Output shape:", out.shape)
print(out)
```

이 코드는 입력 배치, 출력 텐서의 형태, logits을 순서대로 출력한다. 출력 텐서의 형태는 `[2, 4, 50257]`로, 두 문장 각각 4개의 토큰을 가지고 있으며 마지막 차원 50,257은 어휘 크기를 의미한다.

파라미터 수도 계산해 보자.

```python
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
```

```
Total number of parameters: 163,009,536
```

앞서 1억 2,400만 개 파라미터라고 했는데 왜 1억 6,300만 개가 되었을까? 이는 원래 GPT-2가 토큰 임베딩과 출력층 가중치를 동일하게 공유(weight tying)했기 때문이다. 두 층의 가중치 형태가 같으므로, 출력층 파라미터를 빼면 다음과 같이 1억 2,400만 개가 된다.

```python
total_params_gpt2 = (
    total_params - sum(p.numel()
    for p in model.out_head.parameters())
)
print(f"Number of trainable parameters "
    f"considering weight tying: {total_params_gpt2:,}"
)
```

```
Number of trainable parameters considering weight tying: 124,412,160
```

파라미터에 32비트 부동소수점을 사용하면 약 621.83MB의 저장 공간이 필요하다.

<img src="./image/fig_04_16.png" width=800>

이제 모델이 `[batch_size, num_tokens, vocab_size]` 형태의 숫자 텐서를 출력한다는 것을 확인했으므로, 이 값을 텍스트로 변환하는 코드를 만들어 보자.

# 연습문제 4.1 피드포워드와 어텐션 모듈의 파라미터 수 비교

피드포워드 모듈과 멀티 헤드 어텐션 모듈이 각각 몇 개의 파라미터를 갖는지 계산하고 비교해 보자.

# 연습문제 4.2 더 큰 GPT 모델 초기화하기

GPTModel 클래스를 수정하지 않고 설정만 변경하여 GPT-2 Medium(임베딩 1,024, 트랜스포머 블록 24개, 헤드 16개), GPT-2 Large(임베딩 1,280, 블록 36개, 헤드 20개), GPT-2 XL(임베딩 1,600, 블록 48개, 헤드 25개)을 구성하고, 각 모델의 파라미터 수도 계산해 보자.

<img src="./image/fig_04_16.png" width=800>

그림 4.16 GPT 모델이 "Hello, I am"이라는 컨텍스트에서 텍스트를 생성하는 과정을 단계별로 보여 준다. 각 반복마다 입력 컨텍스트에 새 토큰이 추가되며, 여섯 번째 반복에서 "Hello, I am a model ready to help"라는 문장을 완성한다.

<img src="./image/fig_04_17.png" width=800>

그림 4.17 다음 토큰을 생성하는 과정. 모델이 출력한 logits에 소프트맥스를 적용해 확률 분포를 만들고, 가장 큰 확률에 해당하는 토큰 ID를 선택해 텍스트로 디코딩한 뒤 입력 컨텍스트에 추가한다.

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

<img src="./image/fig_04_18.png" width=800>

위 코드는 단순한 생성 루프를 구현한 것으로, 컨텍스트를 자르고, 모델 예측을 수행해 확률이 가장 높은 토큰을 선택한 뒤 컨텍스트에 추가한다. 소프트맥스가 단조 함수(monotonic)라는 점에서, `torch.argmax`를 logits에 바로 적용해도 동일한 결과를 얻을 수 있다. 하지만 logits를 확률로 변환하는 전체 과정을 보여 주기 위해 소프트맥스를 포함했다.

학습된 LLM에서는 항상 가장 높은 확률의 토큰만 선택하면 출력이 단조로워지므로, 다음 장에서는 다양성과 창의성을 위한 확률적 샘플링 기법을 함께 사용할 것이다.

<img src="./image/fig_04_18.png" width=800>

그림 4.18 `generate_text_simple` 함수가 토큰을 하나씩 생성하여 컨텍스트에 덧붙이는 과정을 여섯 번의 반복으로 나타냈다.

이제 "Hello, I am"이라는 문장을 입력으로 사용하여 `generate_text_simple`을 실행해 보자.

```python
start_context = "Hello, I am"
encoded = tokenizer.encode (start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)
```

```
encoded: [15496, 11, 314, 716]
encoded_tensor.shape: torch.Size([1, 4])
```

모델을 평가 모드로 전환해 드롭아웃 같은 랜덤 요소를 비활성화한 뒤, 새 토큰 6개를 생성해 보자.

```python
model.eval()
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output:
", out)
```

생성된 텐서는 다음과 같다.

```
Output:
 tensor([[15496,    11,   314,   716,   257,  1448,   281,  3268,   286,  1294]],
       grad_fn=<CatBackward0>)
```

각 숫자는 토큰 ID를 의미하며, 2장에서 만든 토크나이저로 디코딩하면 다음 텍스트가 된다.

```python
print("Output text:
", tokenizer.decode(out.squeeze(0).tolist()))
```

```
Output text:
 Hello, I am a model ready to help
```

이 예제는 모델 가중치를 학습시키지 않았기 때문에 입력 문장에서 볼 법한 가장 단순한 문장을 출력했다. 현실적으로는 무작위 가중치로 초기화된 모델의 출력은 일관성이 없고 의미 없는 경우가 대부분이다. 이는 훈련이 얼마나 중요한지를 보여 준다.

# 연습문제 4.3 드롭아웃 설정을 세분화하기

이번 장에서는 `GPT_CONFIG_124M`에서 `drop_rate` 하나만을 사용했다. 임베딩, 쇼트컷, 멀티 헤드 어텐션에서 사용하는 드롭아웃 값을 각각 따로 설정하도록 코드를 수정해 보자.

## 요약

- 레이어 정규화는 각 층의 출력을 평균 0, 분산 1로 맞춰 학습을 안정화한다.
- 쇼트컷(스킵, 잔차) 연결은 기울기 소실 문제를 완화해 딥러닝 모델의 학습을 돕는다.
- 트랜스포머 블록은 마스킹 멀티 헤드 어텐션과 GELU 활성 함수를 사용하는 피드포워드 네트워크를 결합한 GPT의 핵심 구성 요소다.
- GPT 모델은 수백만~수십억 개 파라미터를 가진 트랜스포머 블록을 반복적으로 쌓아 만든다.
- GPT-2는 크기에 따라 1억 2천만, 3억 4천만, 7억 6천만, 15억 4천만 파라미터 모델이 있으며, 동일한 GPTModel 클래스로 구성할 수 있다.
- GPT와 같은 생성 모델은 출력 텐서를 softmax와 argmax 등을 사용해 토큰 ID로 변환하고, 이를 텍스트로 디코딩해 한 번에 한 단어(토큰)씩 텍스트를 생성한다.
- 학습되지 않은 GPT 모델은 일관성 없는 텍스트를 생성하므로, 훈련이 필수적이다.
