# LLM Classification Fine-tuning - Baseline

Kaggle Competition: [LLM Classification Fine-tuning](https://www.kaggle.com/competitions/llm-classification-finetuning)

## 목차
- [대회 개요](#대회-개요)
- [대회 배경 및 목적](#대회-배경-및-목적)
- [평가 지표](#평가-지표)
- [데이터셋 설명](#데이터셋-설명)
- [프로젝트 구조](#프로젝트-구조)
- [빠른 시작](#빠른-시작-quick-start)
- [Baseline 모델 구조](#baseline-모델-구조)
- [Baseline 코드 상세 설명](#baseline-코드-상세-설명)
- [중요 사항](#중요-사항)
- [성능 개선 아이디어](#성능-개선-아이디어)
- [문제 해결](#문제-해결)
- [FAQ](#faq-자주-묻는-질문)
- [참고 자료](#참고-자료)

## 대회 개요

**LLM Classification Fine-tuning**은 사용자가 두 개의 LLM 응답 중 어느 것을 선호할지 예측하는 대회입니다.

### 문제 정의
- **Task**: Head-to-Head 배틀에서 사용자 선호도 예측
- **데이터**: Chatbot Arena에서 수집된 실제 대화 데이터
- **목표**: 사용자가 선호하는 응답을 예측하는 모델 개발

### 출력 형식
3개 클래스에 대한 확률값 (합이 1):
- `winner_model_a`: 모델 A가 더 나은 확률
- `winner_model_b`: 모델 B가 더 나은 확률
- `winner_tie`: 비슷한 확률

### 대회 유형
- **Getting Started Competition**: 초보자를 위한 비경쟁 대회
- **Rolling Leaderboard**: 2개월 롤링 윈도우 (오래된 제출은 자동 제거)
- **무기한 운영**: 언제든 참가 가능
- **상금 없음**: 학습 및 경험 목적

## 대회 배경 및 목적

### 배경: Chatbot Arena
이 대회는 **Chatbot Arena** 데이터를 활용합니다:
- 사용자가 두 개의 익명 LLM과 대화
- 사용자가 더 선호하는 응답 선택
- 실제 사용자 선호도 데이터 수집

### 목적: RLHF의 Reward Model
이 대회는 **Reinforcement Learning from Human Feedback (RLHF)** 의 핵심 요소인 **Reward Model** 또는 **Preference Model** 개발과 직접 연관됩니다.

#### 기존 LLM의 한계
기존 LLM을 직접 사용하여 선호도를 예측할 때 발생하는 편향:
- **Position Bias**: 먼저 제시된 응답을 선호
- **Verbosity Bias**: 더 긴 응답을 선호
- **Self-Enhancement Bias**: 자신의 응답을 선호

#### 이 대회의 의의
- 편향을 극복하는 효과적인 선호도 예측 모델 개발
- 사용자 개인의 선호도에 맞춘 응답 생성 가능
- 더 사용자 친화적인 AI 대화 시스템 구축

## 평가 지표

### Log Loss (Cross-Entropy Loss)
```
Log Loss = -1/N * Σ Σ y_ij * log(p_ij)
```

**제출 형식**:
- 각 테스트 샘플에 대해 3개 클래스의 확률 예측
- 모든 확률의 합은 1이어야 함
- `eps=auto` 적용 (0 또는 1에 가까운 확률 보정)

**예시**:
```csv
id,winner_model_a,winner_model_b,winner_tie
136060,0.33,0.33,0.33
211333,0.40,0.35,0.25
1233961,0.25,0.50,0.25
```

**Log Loss 해석**:
- **낮을수록 좋음**: 0에 가까울수록 완벽한 예측
- **확률 기반**: 정확한 클래스뿐만 아니라 확률 분포의 정확도도 평가
- **불확실성 반영**: 여러 클래스에 확률을 분산시키면 안전하지만 점수는 낮아짐

## 데이터셋 설명

### 데이터 출처
**ChatBot Arena**의 실제 사용자 인터랙션 데이터를 사용합니다:
- 사용자(judge)가 하나 이상의 프롬프트를 두 개의 서로 다른 LLM에 제공
- 사용자가 더 만족스러운 응답을 제공한 모델을 선택
- 목표: 사용자의 선호도를 예측하고 주어진 prompt/response 쌍이 승자로 선택될 확률 결정

### 데이터 규모
- **학습 데이터**: 57,477행 (대회 설명에는 "약 55,000행"으로 명시)
- **테스트 데이터**: 약 25,000행 (실제 제출 시)
- **예제 테스트 데이터**: 3행 (제출 시 전체 테스트 세트로 대체됨)

⚠️ **경고**: 이 데이터셋에는 모욕적이거나 저속하거나 공격적으로 간주될 수 있는 텍스트가 포함되어 있습니다.

### 파일 구조

#### train.csv (학습 데이터)
| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `id` | int | 행의 고유 식별자 |
| `model_a` | string | 모델 A의 이름 (예: gpt-4, claude-2) |
| `model_b` | string | 모델 B의 이름 |
| `prompt` | string | 두 모델에 입력으로 제공된 프롬프트 |
| `response_a` | string | 모델 A가 프롬프트에 대해 생성한 응답 |
| `response_b` | string | 모델 B가 프롬프트에 대해 생성한 응답 |
| `winner_model_a` | float | 모델 A가 승리한 경우 1, 아니면 0 (목표 변수) |
| `winner_model_b` | float | 모델 B가 승리한 경우 1, 아니면 0 (목표 변수) |
| `winner_tie` | float | 비긴 경우 1, 아니면 0 (목표 변수) |

**예시**:
```csv
id,model_a,model_b,prompt,response_a,response_b,winner_model_a,winner_model_b,winner_tie
1,gpt-4,claude-2,"What is AI?","AI is..","Artificial Intelligence...",0,1,0
```

#### test.csv (테스트 데이터)
| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `id` | int | 행의 고유 식별자 |
| `prompt` | string | 두 모델에 입력으로 제공된 프롬프트 |
| `response_a` | string | 모델 A의 응답 |
| `response_b` | string | 모델 B의 응답 |

**주의**: `model_a`와 `model_b` 컬럼은 테스트 데이터에 포함되지 않습니다.

**예시**:
```csv
id,prompt,response_a,response_b
136060,"Explain quantum computing","Quantum computing uses...","A quantum computer..."
```

#### sample_submission.csv (제출 형식 예시)
| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `id` | int | test.csv의 id와 매칭 |
| `winner_model_a` | float | 모델 A가 승리할 확률 (0~1) |
| `winner_model_b` | float | 모델 B가 승리할 확률 (0~1) |
| `winner_tie` | float | 비길 확률 (0~1) |

**제약**: 각 행의 세 확률의 합은 1이어야 함.

**예시**:
```csv
id,winner_model_a,winner_model_b,winner_tie
136060,0.33,0.33,0.34
211333,0.25,0.50,0.25
```

### 데이터 특징

#### 1. 다중 클래스 확률 예측
- 단순 분류가 아닌 확률 분포 예측
- Soft labels 가능 (예: [0.4, 0.5, 0.1])

#### 2. 모델 정보 사용 불가 (테스트 시)
- 학습 데이터에는 `model_a`, `model_b` 포함
- 테스트 데이터에는 모델 정보 없음
- **전략**: 모델 이름에 의존하지 않는 일반화된 모델 개발 필요

#### 3. 텍스트 길이 변동성
- Prompt: 짧은 질문부터 긴 설명까지
- Response: 한 문장부터 여러 문단까지
- **고려사항**: MAX_LENGTH 설정 시 주의

#### 4. 다양한 도메인
- 기술, 과학, 일상 대화, 창작 등
- 다양한 질문 유형 (설명, 비교, 조언, 코딩 등)

### 데이터 활용 팁

#### 1. 탐색적 데이터 분석 (EDA)
```python
import pandas as pd

train = pd.read_csv('data/train.csv')

# 클래스 분포 확인
print(train[['winner_model_a', 'winner_model_b', 'winner_tie']].sum())

# 텍스트 길이 분포
train['prompt_len'] = train['prompt'].str.len()
train['response_a_len'] = train['response_a'].str.len()
train['response_b_len'] = train['response_b'].str.len()

print(train[['prompt_len', 'response_a_len', 'response_b_len']].describe())

# 모델 분포
print(train['model_a'].value_counts())
print(train['model_b'].value_counts())
```

#### 2. 데이터 전처리 고려사항
- **결측값**: 확인 및 처리
- **특수 문자**: 이모지, HTML 태그 등
- **긴 텍스트**: 토큰 제한으로 잘릴 수 있음
- **불균형**: 클래스 분포 확인

#### 3. Feature Engineering 아이디어
- 응답 길이 차이
- 어휘 다양성 (unique words)
- 감성 분석 점수
- 가독성 점수 (Flesch-Kincaid 등)
- 문법 오류 수

## 프로젝트 구조

```
llm-classification-finetuning/
├── README.md                # 이 파일
├── requirements.txt         # Python 패키지 의존성
├── setup_jupyter.sh         # Jupyter 커널 설정 스크립트
│
├── baseline.py             # Kaggle 제출용 (Kaggle 노트북에 복사)
├── baseline_local.py       # 로컬 테스트용
├── download_model.py       # DistilBERT 다운로드 및 압축 스크립트
├── exploration.ipynb       # 데이터 탐색용 Jupyter 노트북
│
├── data/                   # 데이터 디렉토리 (gitignore)
│   ├── train.csv          # 학습 데이터 (~176MB, 57,477행)
│   ├── test.csv           # 테스트 데이터
│   └── sample_submission.csv
│
├── models/                 # 모델 디렉토리 (gitignore)
│   ├── distilbert-base-uncased/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── ...
│   └── distilbert-base-uncased.zip  # Kaggle 업로드용
│
└── outputs/                # 로컬 실행 결과 (gitignore)
    ├── best_model.pt      # 학습된 모델 (로컬용)
    └── submission.csv     # 예측 결과 (로컬용)
```

## 빠른 시작 (Quick Start)

### 워크플로우 선택

#### 로컬에서 테스트 후 Kaggle 제출

로컬에서 코드를 검증한 후 Kaggle에서 실행합니다.

**1단계: 환경 설정**
```bash
# 가상환경 생성 및 활성화 (선택)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

**2단계: 모델 다운로드**
```bash
python download_model.py
```
이 스크립트는 `models/distilbert-base-uncased/` 폴더에 모델을 다운로드합니다 (~254 MB).
자동으로 모델을 `models/distilbert-base-uncased.zip` 에 압축 합니다.

**3단계: 데이터 준비**

```bash
kaggle competitions download -c llm-classification-finetuning
```
위 명령어로 Kaggle에서 대회 데이터(llm-classification-finetuning.zip)를 다운로드 하고 압축을 풀어 `data/` 폴더에 배치:
- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv`

**4단계: 데이터 탐색 (선택, 권장)**
```bash
# Jupyter 커널 설정
bash setup_jupyter.sh

# Jupyter Notebook 실행
jupyter notebook exploration.ipynb
```
- 노트북에서 커널 선택: `Kernel → Change Kernel → Python (LLM-Classification-FT)`
- 데이터 분포, 텍스트 길이, 모델 분포 등 상세 분석
- Feature Engineering 아이디어 도출

**5단계: 로컬에서 학습 테스트 (선택)**
```bash
python baseline_local.py
```
- 작은 데이터셋으로 테스트하거나 EPOCHS를 1로 설정하여 빠르게 검증
- 결과는 `outputs/` 폴더에 저장됨
- 이 단계는 코드 검증용이며, 생성된 모델은 Kaggle에 업로드할 수 없음

**6단계: Kaggle 노트북 생성 및 실행**
1. [대회 페이지](https://www.kaggle.com/competitions/llm-classification-finetuning)의 "Code" 탭으로 이동
2. "New Notebook" 클릭
3. **Settings → GPU T4 x2** 활성화 (필수)
4. **Add Data** 클릭:
   - Competition 데이터: `llm-classification-finetuning` 추가
   - Model upload: `distilbert-base-uncased.zip` 추가
5. 노트북에 [baseline.py](baseline.py)의 전체 코드를 복사/붙여넣기
6. **Run All** 클릭
7. 학습 완료 후 노트북을 대회에 제출

## Baseline 모델 구조

### 개요

- **모델**: DistilBERT (경량화된 BERT, 파라미터 66M)
- **입력 형식**: `prompt [SEP] response_a [SEP] response_b`
- **출력**: 3개 클래스에 대한 확률 벡터
- **학습 시간**: GPU T4 기준 약 15-20분 (1 epoch)

### 상세 아키텍처

```
Input Text (Concatenated)
    ↓
[Tokenizer] - MAX_LENGTH=256
    ↓
DistilBERT Encoder (6 layers, 768 hidden)
    ↓
[CLS] Token Representation (768-dim)
    ↓
Dropout (p=0.3)
    ↓
Linear Layer (768 → 3)
    ↓
Softmax
    ↓
Output Probabilities [P(A wins), P(B wins), P(tie)]
```

### 주요 하이퍼파라미터

| 파라미터 | 값 | 설명 | 코드 위치 |
|---------|-----|------|----------|
| MAX_LENGTH | 256 | 토큰 최대 길이 | [baseline.py:56](baseline.py#L56) |
| BATCH_SIZE | 16 | 배치 크기 | [baseline.py:57](baseline.py#L57) |
| EPOCHS | 1 | 에포크 수 (3-5로 증가 권장) | [baseline.py:58](baseline.py#L58) |
| LEARNING_RATE | 2e-5 | 학습률 | [baseline.py:59](baseline.py#L59) |
| DROPOUT | 0.3 | Dropout 비율 | [baseline.py:164](baseline.py#L164) |

## Baseline 코드 상세 설명

[baseline.py](baseline.py)는 총 338줄의 완전한 학습 및 추론 파이프라인입니다. 이 섹션에서는 코드의 각 부분을 단계별로 상세히 설명합니다.

### 코드 실행 흐름 (Execution Flow)

```
1. 라이브러리 임포트 & 시드 설정
   ↓
2. Config 클래스로 하이퍼파라미터 설정
   ↓
3. 데이터 로딩 (train.csv, test.csv)
   ↓
4. Train/Validation 분할 (90%/10%)
   ↓
5. Dataset & DataLoader 생성
   ↓
6. 모델 초기화 (DistilBERT + Classification Head)
   ↓
7. Optimizer & Scheduler 설정
   ↓
8. 학습 루프 (Epochs)
   │  ├─ Training
   │  ├─ Validation
   │  └─ Best Model 저장
   ↓
9. Best Model 로드
   ↓
10. 테스트 데이터 예측
   ↓
11. Submission 파일 생성
```

### 1. 설정 및 초기화 ([baseline.py:13-38](baseline.py#L13-L38))

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
```

**주요 라이브러리**:
- `transformers`: HuggingFace의 DistilBERT 모델 사용
- `torch`: PyTorch 딥러닝 프레임워크
- `sklearn`: 데이터 분할 및 평가 지표

**랜덤 시드 설정** ([baseline.py:32-37](baseline.py#L32-L37)):
```python
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```
재현 가능한 결과를 위해 모든 랜덤 시드를 고정합니다.

### 2. 설정 클래스 ([baseline.py:43-69](baseline.py#L43-L69))

```python
class Config:
    MODEL_NAME = "/kaggle/input/distilbert-base-uncased/..."
    MAX_LENGTH = 256
    BATCH_SIZE = 16
    EPOCHS = 1
    LEARNING_RATE = 2e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 경로 설정
    TRAIN_DATA_PATH = "/kaggle/input/.../train.csv"
    TEST_DATA_PATH = "/kaggle/input/.../test.csv"
    SUBMISSION_PATH = "/kaggle/working/submission.csv"
    MODEL_SAVE_PATH = "/kaggle/working/best_model.pt"
```

**역할**: 모든 하이퍼파라미터와 경로를 한 곳에서 관리하여 수정이 용이합니다.

**💡 초보자 팁**:
- `BATCH_SIZE`를 줄이면 메모리 사용량 감소 (GPU 메모리 부족 시)
- `EPOCHS`를 늘리면 성능 향상 가능 (단, 과적합 주의)
- `MAX_LENGTH`를 늘리면 더 긴 텍스트 처리 가능 (단, 메모리 사용 증가)

### 3. 데이터 로딩 및 전처리 ([baseline.py:92-107](baseline.py#L92-L107))

```python
train_df = pd.read_csv(config.TRAIN_DATA_PATH)
test_df = pd.read_csv(config.TEST_DATA_PATH)

# Train/Validation split (90%/10%)
train_data, val_data = train_test_split(
    train_df,
    test_size=0.1,
    random_state=42,
    stratify=train_df['winner_model_a'].astype(str) +
             train_df['winner_model_b'].astype(str)
)
```

**Stratified Split**: 클래스 분포를 유지하면서 데이터를 분할하여 검증 세트의 대표성을 보장합니다.

**🎯 왜 Stratified Split인가?**:
- 세 클래스(`winner_model_a`, `winner_model_b`, `winner_tie`)의 비율이 불균형할 수 있음
- Stratified split은 train/val 세트 모두에서 동일한 클래스 비율 유지
- 예: Train에 winner_tie가 20%라면, Val에도 약 20% 유지

**💡 초보자 팁**:
- `test_size=0.1`은 10% 검증, 90% 학습을 의미
- `random_state=42`는 재현성을 위한 시드 (다른 숫자도 가능)

### 4. Dataset 클래스 ([baseline.py:112-149](baseline.py#L112-L149))

```python
class LLMComparisonDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 입력 형식: prompt + [SEP] + response_a + [SEP] + response_b
        text = f"{row['prompt']} [SEP] {row['response_a']} [SEP] {row['response_b']}"

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # [CLS], [SEP] 토큰 추가
            max_length=self.max_length,
            padding='max_length',     # 최대 길이까지 패딩
            truncation=True,          # 최대 길이 초과 시 자르기
            return_attention_mask=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }        

        if not self.is_test:
            labels = torch.tensor([
                row['winner_model_a'],
                row['winner_model_b'],
                row['winner_tie']
            ], dtype=torch.float)
            item['labels'] = labels

        return item
```

**핵심 동작**:
1. **입력 결합**: Prompt와 두 응답을 `[SEP]` 토큰으로 구분하여 하나의 시퀀스로 만듭니다.
2. **토큰화**: DistilBERT 토크나이저로 텍스트를 토큰 ID로 변환합니다.
3. **패딩/자르기**: 모든 입력을 256 토큰으로 통일합니다.
4. **레이블**: 학습/검증 데이터의 경우 3개 클래스의 확률값을 반환합니다.

**🎯 왜 이런 입력 형식인가?**:
```
입력: "What is AI? [SEP] AI is artificial intelligence... [SEP] Artificial Intelligence refers to..."
       ↓
토큰화: [CLS] What is AI ? [SEP] AI is artificial ... [SEP] Artificial Intelligence ... [SEP] [PAD] [PAD] ...
       ↓
BERT가 세 부분의 관계를 학습: Prompt → Response A → Response B
```

**실제 예시**:
- `row['prompt']` = "Explain quantum computing"
- `row['response_a']` = "Quantum computing uses qubits..."
- `row['response_b']` = "A quantum computer leverages..."
- 결합된 텍스트 = "Explain quantum computing [SEP] Quantum computing uses qubits... [SEP] A quantum computer leverages..."

**💡 초보자 팁**:
- `max_length=256`이 텍스트보다 짧으면 뒷부분이 잘림 (`truncation=True`)
- `padding='max_length'`로 모든 입력을 동일한 길이로 만듦 (배치 처리를 위해 필수)
- `return_attention_mask=True`는 실제 토큰과 패딩을 구분하는 마스크 생성

### 5. Model 클래스 ([baseline.py:156-171](baseline.py#L156-L171))

```python
class LLMComparisonModel(nn.Module):
    def __init__(self, model_name, num_classes=3):
        super(LLMComparisonModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        probs = self.softmax(logits)
        return probs
```

**구조 설명**:
1. **DistilBERT Encoder**: Pre-trained 모델을 로드하여 텍스트 인코딩
2. **[CLS] 토큰 추출**: `last_hidden_state[:, 0, :]`로 첫 번째 토큰(문장 전체 표현)을 가져옴
3. **Dropout**: 과적합 방지를 위한 정규화 (30% 확률로 뉴런 비활성화)
4. **Classification Head**: 768차원 → 3차원으로 선형 변환
5. **Softmax**: 로짓을 확률값으로 변환 (합이 1)

**🎯 [CLS] 토큰과 슬라이싱 이해하기**:

`[CLS]`는 "Classification"의 약자로, BERT가 문장 전체 의미를 요약하도록 학습된 특수 토큰입니다.

```python
# DistilBERT 출력 형태
outputs.last_hidden_state: [배치_크기, 시퀀스_길이, 히든_차원]
                          [16,       256,        768]
                           ↓         ↓           ↓
                         샘플수   토큰수    벡터차원

# 각 토큰마다 768차원 벡터 생성:
# [CLS] → [0.23, 0.45, ..., 0.89]  (768개) ← 문장 전체 정보 압축!
# What  → [0.11, 0.33, ..., 0.77]  (768개)
# is    → [0.44, 0.21, ..., 0.66]  (768개)
# ...

# [:, 0, :] 슬라이싱 의미
pooled_output = outputs.last_hidden_state[:, 0, :]
#                                          ↑  ↑  ↑
#                                          |  |  └─ 모든 768개 차원
#                                          |  └──── 0번째 토큰 ([CLS])
#                                          └─────── 모든 배치 샘플

# 결과: [16, 768] - 각 샘플의 [CLS] 토큰 벡터만 추출
```

**데이터 흐름**:
```
[16, 256, 768]  →  [:, 0, :]  →  [16, 768]  →  Dropout  →  Linear  →  [16, 3]  →  Softmax  →  확률
모든 토큰 벡터      [CLS]만       중간 표현                              로짓              최종 확률
```

**💡 초보자 팁**:
- `[CLS]` 토큰은 중간 단계이며, 최종 결과는 3개 클래스의 확률값입니다
- `nn.Dropout(0.3)`: 학습 시 30%의 뉴런을 무작위로 끔 → 과적합 방지
- `nn.Linear(768, 3)`: 768차원 입력을 3차원 출력(3개 클래스)으로 변환
- `nn.Softmax(dim=1)`: 로짓을 확률로 변환, `dim=1`은 클래스 차원에 대해 적용

**모델 크기**:
- DistilBERT 파라미터: ~66M
- Classification Head 파라미터: 768 × 3 + 3 = 2,307
- 전체: ~66.4M 파라미터

### 6. 학습 함수 ([baseline.py:200-221](baseline.py#L200-L221))

```python
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        # Forward pass
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask)

        # Loss 계산 (Binary Cross Entropy)
        loss = nn.BCELoss()(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

**학습 프로세스**:
1. **Forward Pass**: 입력 → 모델 → 예측 확률
2. **Loss 계산**: BCE Loss로 예측과 실제 레이블 간 차이 측정
3. **Backward Pass**: 그래디언트 계산 및 파라미터 업데이트
4. **Scheduler**: Learning rate를 점진적으로 감소 (warmup 사용)

**BCE Loss 선택 이유**:
- Multi-label classification으로 간주 (여러 클래스가 동시에 부분적으로 참일 수 있음)
- 확률값이 soft label로 제공됨 (예: [0.5, 0.3, 0.2])

**💡 초보자 팁**:
- `optimizer.zero_grad()`: 이전 batch의 그래디언트를 초기화 (필수!)
- `loss.backward()`: 역전파로 그래디언트 계산
- `optimizer.step()`: 계산된 그래디언트로 파라미터 업데이트
- `scheduler.step()`: Learning rate 조정 (매 step마다 호출)

**학습 과정 시각화**:
```
Batch 1: loss=0.7182
Batch 2: loss=0.7036  ← 조금씩 감소
Batch 3: loss=0.7312
...
Batch 3234: loss=0.6161
Average Training Loss: 0.6308  ← 전체 평균
```

### 7. 검증 함수 ([baseline.py:223-242](baseline.py#L223-L242))

```python
def validate(model, dataloader, device):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():  # 그래디언트 계산 안 함
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            predictions.append(outputs.cpu().numpy())
            actuals.append(labels.cpu().numpy())

    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    loss = log_loss(actuals, predictions)

    return loss, predictions, actuals
```

**평가 지표**: Log Loss (Cross Entropy)
- Kaggle 대회의 공식 평가 지표
- 확률 예측의 정확도를 측정

### 8. Optimizer와 Scheduler ([baseline.py:249-251](baseline.py#L249-L251))

```python
optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
```

**AdamW**: Adam optimizer의 개선 버전으로 weight decay를 올바르게 처리합니다.

**Linear Schedule with Warmup**:
- Learning rate를 선형적으로 감소시켜 안정적인 수렴을 도움
- Warmup 없이 시작 (BERT fine-tuning에서 일반적)

### 9. 학습 루프 ([baseline.py:265-278](baseline.py#L265-L278))

```python
best_val_loss = float('inf')

for epoch in range(config.EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, config.DEVICE)
    val_loss, val_preds, val_actuals = validate(model, val_loader, config.DEVICE)

    # Best model 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
```

**Early Stopping 패턴**:
- 검증 손실이 개선될 때만 모델 저장
- 과적합 방지

### 10. 예측 및 제출 ([baseline.py:292-313](baseline.py#L292-L313))

```python
# Best model 로드
model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
model.eval()

# 테스트 데이터 예측
predictions = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        outputs = model(input_ids, attention_mask)
        predictions.append(outputs.cpu().numpy())

predictions = np.vstack(predictions)

# 제출 파일 생성
submission = sample_submission.copy()
submission['winner_model_a'] = predictions[:, 0]
submission['winner_model_b'] = predictions[:, 1]
submission['winner_tie'] = predictions[:, 2]

submission.to_csv(config.SUBMISSION_PATH, index=False)
```

**최종 단계**:
1. 저장된 best model 로드
2. 테스트 데이터에 대해 예측 수행
3. 확률값을 제출 형식에 맞게 변환
4. CSV 파일로 저장

### 전체 실행 예시 (로컬 실행 결과)

```
Libraries imported successfully!
PyTorch version: 2.9.0+cu128
CUDA available: True
Random seeds set to 42

Configuration:
  Model: ./models/distilbert-base-uncased
  Device: cuda
  Batch Size: 16
  Epochs: 1

Loading data...
Train data shape: (57477, 9)
Test data shape: (3, 4)
Train size: 51729, Validation size: 5748

Loading tokenizer and model...
✓ Model loaded on: cuda
✓ Model parameters: 66,365,187

============================================================
STARTING TRAINING
============================================================

Epoch 1/1
------------------------------------------------------------
Training: 100%|█████████| 3234/3234 [10:50<00:00, 4.97it/s, loss=0.6161]
Training loss: 0.6308
Validating: 100%|█████| 360/360 [00:21<00:00, 16.52it/s]
Validation loss: 1.0750
✓ Model saved with validation loss: 1.0750

============================================================
TRAINING COMPLETED! Best validation loss: 1.0750
============================================================

Making predictions on test data...
Predicting: 100%|█████████████████| 1/1 [00:00<00:00, 15.31it/s]
✓ Predictions shape: (3, 3)

============================================================
SUBMISSION FILE CREATED
============================================================
Saved to: outputs/submission.csv

First few predictions:
        id  winner_model_a  winner_model_b  winner_tie
0   136060        0.233993        0.215965    0.550042
1   211333        0.317617        0.405372    0.277011
2  1233961        0.355113        0.364986    0.279901

Probability sums check:
  Min: 1.000000
  Max: 1.000000
  Mean: 1.000000

✓ All probabilities sum to ~1.0. Submission is valid!
```

**실행 시간 분석**:
- 학습: 10분 50초 (3,234 batches, ~4.97 it/s)
- 검증: 21초 (360 batches, ~16.52 it/s)
- 예측: 1초 미만 (테스트 데이터 3개)
- **총 소요 시간**: 약 11분 (GPU T4 기준, 1 epoch)

**성능 분석**:
- Training Loss: 0.6308 (BCE Loss)
- Validation Loss: 1.0750 (Log Loss)
- Validation loss가 training loss보다 높음 → 약간의 과적합 또는 데이터 분포 차이

**개선 방향**:
- Epochs를 3-5로 증가시켜 성능 향상
- Validation loss 모니터링하며 early stopping 적용
- Learning rate 조정 또는 warmup 추가

## 중요 사항

### Code Competition 규정

이 대회는 **Code Competition**입니다. 모든 제출은 Kaggle Notebook을 통해 이루어져야 합니다.

#### 제출 요구사항
- ✅ **CPU Notebook**: 최대 9시간 실행 시간
- ✅ **GPU Notebook**: 최대 9시간 실행 시간
- ❌ **인터넷 비활성화**: 외부 데이터 다운로드 불가 (데이터셋으로 미리 업로드 필요)
- ✅ **외부 데이터 허용**: 공개적이고 합리적 비용의 데이터/모델 사용 가능 (자세한 내용은 하단 "외부 데이터 및 도구" 참조)
- ✅ **제출 파일명**: 반드시 `submission.csv`여야 함
- 📝 **실행 시간 난독화**: 동일한 제출도 최대 15분의 차이 발생

#### 제출 방식
1. ❌ **CSV 파일 직접 제출 불가**: submission.csv를 직접 업로드할 수 없음
2. ✅ **노트북 제출**: Kaggle 노트북 자체를 제출하면 자동으로 실행되어 평가됨
3. ❌ **로컬 학습 모델 업로드 불가**: 로컬에서 학습한 모델을 사용할 수 없음
4. ✅ **Pre-trained 모델 허용**: DistilBERT, BERT 등 공개 모델은 데이터셋으로 업로드 후 사용 가능

#### 제출 프로세스
```
1. 노트북 작성 → 2. 노트북 제출 → 3. Kaggle이 자동 실행
   → 4. submission.csv 생성 → 5. 자동 평가 → 6. 리더보드 업데이트
```

**중요**: `submission.csv`와 `best_model.pt`는 노트북 실행 중에 자동으로 생성되어야 하며, Kaggle이 이를 평가에 사용합니다. 우리가 직접 업로드하는 것이 아닙니다.

### 대회 규칙 (Competition Rules)

#### 참가 자격 (Eligibility)
- **계정**: Kaggle.com에 등록된 계정 보유자
- **연령**: 18세 이상 또는 거주 지역의 성년 나이
- **거주 지역 제한**: 다음 지역 거주자는 참가 불가
  - 크림반도, 도네츠크, 루한스크, 쿠바, 이란, 시리아, 북한
  - 미국 수출 통제 또는 제재 대상자
- **복수 계정 금지**: 한 명당 하나의 Kaggle 계정만 사용 가능

#### 팀 규정 (Team Rules)
- **최대 팀 크기**: 10명
- **팀 합병**: 허용됨 (단, 제출 횟수 제한 내에서)
- **팀 구성**: 각 팀원은 개별 Kaggle 계정 필요
- **팀 합병 조건**:
  - 합병 후 팀 크기가 최대 제한 이내
  - 합병 시점까지의 제출 횟수가 허용 범위 이내
  - 합병 마감일 이전에 완료

#### 제출 제한 (Submission Limits)
- **일일 제출**: 최대 10회
- **최종 제출**: 최대 2개 선택 가능 (최종 평가용)

#### 데이터 사용 규정 (Data Usage)
- **라이선스**: CC BY-NC 4.0
- **비상업적 용도만 허용**: 대회 참가, Kaggle 포럼, 학술 연구 및 교육 목적
- **금지 사항**:
  - Hand-labeling (수동 라벨링) 금지
  - 검증/테스트 데이터에 대한 수동 예측 금지
  - 대회 데이터를 대회 외 참가자에게 공유 금지
- **랭킹 포인트**: 공개 데이터 특성상 Kaggle 랭킹 포인트 미부여

#### 외부 데이터 및 도구 (External Data & Tools)
**허용되는 외부 데이터**:
- ✅ 모든 참가자에게 공개적이고 동등하게 접근 가능한 데이터
- ✅ 비용이 들지 않거나 최소 비용으로 이용 가능한 데이터
- ✅ "합리성 기준(Reasonableness Standard)" 충족 시 사용 가능

**예시**:
- ✅ **허용**: Gemini Advanced 같은 소액 구독료 (합리적 비용)
- ❌ **불허**: 대회 상금을 초과하는 독점 데이터셋 라이선스

**자동화된 ML 도구 (AMLT)**:
- Google AutoML, H2O Driverless AI 등 사용 가능
- 단, 적절한 라이선스 보유 및 대회 규칙 준수 필요

#### 코드 공유 규정 (Code Sharing)
**비공개 공유 (Private Sharing)**:
- ❌ **금지**: 팀 외부로 Competition Code 비공개 공유
- ❌ **금지**: 팀 간 코드 공유 (팀 합병 제외)
- 위반 시 실격 처리

**공개 공유 (Public Sharing)**:
- ✅ **허용**: Competition Code를 공개적으로 공유 가능
- ✅ **조건**: Kaggle.com의 해당 대회 Discussion 또는 Notebooks에 공유
- ✅ **라이선스**: Open Source Initiative 승인 라이선스 적용
- ✅ **상업적 사용**: 공유 코드의 상업적 사용 제한 불가

**오픈 소스 사용**:
- Open Source Initiative 승인 라이선스만 사용 가능
- 상업적 사용을 제한하지 않는 라이선스

#### 우승자 결정 (Winner Determination)
- **평가 기준**: Private Leaderboard 점수
- **Public Leaderboard**: 공개 테스트 세트 기반 (대회 중 공개)
- **Private Leaderboard**: 비공개 테스트 세트 기반 (최종 순위 결정)
- **동점 처리**: 먼저 제출한 팀이 우승

#### 실격 사유 (Disqualification)
다음 행위 시 실격 처리될 수 있습니다:
- 부정행위, 속임수, 불공정 플레이
- 다른 참가자, 주최자, Kaggle을 위협하거나 괴롭힘
- 규칙 위반 (복수 계정, 코드 비공개 공유 등)
- 대회의 합법적 운영을 방해하는 행위

#### 지적 재산권 (Intellectual Property)
- **제출물 소유권**: 참가자가 제출물의 독점적 소유자여야 함
- **금지 사항**:
  - 제3자의 지적재산권 침해
  - 저작권, 상표권, 특허권, 영업비밀, 개인정보 침해
  - 명예훼손
- **보상**: 침해 발생 시 참가자가 대회 주최자에게 배상 책임

#### 상금 및 세금 (Prizes & Taxes)
- **이 대회**: Getting Started Competition으로 상금 없음
- **일반 규정**: 상금이 있는 경우 모든 세금은 우승자 책임
- **팀 상금**: 균등 분배 (팀원 합의 시 다른 분배 가능)

#### 개인정보 보호 (Privacy)
- Kaggle과 대회 주최자가 개인정보 수집 및 사용
- Kaggle Privacy Policy 적용
- 대회 주최자에게 개인정보 전송 (국가 간 전송 포함)

#### 법률 및 관할권 (Governing Law)
- **준거법**: 캘리포니아 주법
- **관할 법원**: 캘리포니아 Santa Clara 카운티 연방 또는 주 법원

#### 대회 타임라인
- **시작일**: 2024년 10월 16일
- **종료일**: 없음 (무기한 운영)
- **롤링 리더보드**: 2개월 이상 된 제출은 자동 제거

#### 고용 관계 부재
- 대회 참가는 고용 제안 또는 고용 계약을 구성하지 않음
- 제출물은 자발적으로 제공되며 신뢰 관계 아님

### 로컬 vs Kaggle 코드 차이

두 파일은 **경로만 다르고 나머지 코드는 동일**합니다:

| 구분 | baseline_local.py | baseline.py |
|------|-------------------|-------------|
| 모델 경로 | `./models/distilbert-base-uncased` | `/kaggle/input/distilbert-base-uncased/...` |
| 데이터 경로 | `./data/train.csv` | `/kaggle/input/llm-classification-finetuning/train.csv` |
| 출력 경로 | `./outputs/` | `/kaggle/working/` |
| 용도 | 로컬 테스트 및 검증 | Kaggle 제출용 |
| 경로 검증 | 파일 존재 여부 확인 후 종료 | 존재 여부만 출력 |

## 성능 개선 아이디어

### 1. 모델 개선
- **더 큰 모델**: BERT-base (110M), RoBERTa-base (125M), DeBERTa-base (140M)
- **앙상블**: 여러 모델의 예측 평균 또는 가중 평균
- **Multi-task Learning**: 관련 태스크를 함께 학습

### 2. 하이퍼파라미터 튜닝
- **Epochs 늘리기**: 3-5 epochs (과적합 주의)
- **Learning rate 조정**: 1e-5 ~ 5e-5 범위에서 실험
- **Batch size 조정**: 8, 16, 32 (메모리 허용 범위 내)
- **Dropout 비율**: 0.1 ~ 0.5 사이에서 조정
- **Max Length**: 512로 늘려 더 긴 컨텍스트 활용
- **Warmup steps**: 전체 step의 10% 정도 사용

### 3. 입력 형식 개선
- **각 응답 별도 인코딩**: `[CLS] prompt [SEP] response_a [SEP]`와 `[CLS] prompt [SEP] response_b [SEP]`를 각각 인코딩 후 결합
- **Cross-attention**: 두 응답 간 상호작용 모델링
- **특수 토큰 추가**: 응답 시작/끝을 명시적으로 표시

### 4. 추가 특성 활용
- **응답 길이**: 길이가 품질과 상관관계가 있을 수 있음
- **모델 이름**: model_a, model_b 정보 활용
- **텍스트 통계**: 문장 수, 단어 다양성 등

### 5. 데이터 증강
- **Back-translation**: 다른 언어로 번역 후 다시 번역
- **Paraphrasing**: 동일한 의미의 다른 표현 생성
- **Mixup/Cutout**: 텍스트 일부를 마스킹하거나 섞기

### 6. 정규화 기법
- **Label Smoothing**: Hard label을 부드럽게 만들기
- **Weight Decay**: L2 정규화 강도 조정
- **Gradient Clipping**: 그래디언트 폭발 방지

### 7. 학습 전략
- **Gradual Unfreezing**: BERT 레이어를 점진적으로 학습
- **Discriminative Learning Rate**: 레이어별로 다른 learning rate 적용
- **K-Fold Cross Validation**: 여러 fold로 학습 후 앙상블

## 문제 해결

### Out of Memory 에러
[baseline.py:57](baseline.py#L57) 또는 [baseline_local.py:48](baseline_local.py#L48)에서:
```python
BATCH_SIZE = 8  # 16 → 8 또는 4로 줄이기
MAX_LENGTH = 128  # 256 → 128로 줄이기
```

### 모델 경로 에러 (Kaggle)
Kaggle에서 데이터셋이 제대로 추가되었는지 확인:
```python
import os
print(os.listdir('/kaggle/input'))

# 올바른 경로 찾기
for root, dirs, files in os.walk('/kaggle/input/distilbert-base-uncased'):
    if 'config.json' in files:
        print(f"Model path: {root}")
        break
```

### 로컬 실행 시 데이터/모델 경로 에러
- `data/` 폴더에 train.csv, test.csv, sample_submission.csv가 있는지 확인
- `models/distilbert-base-uncased/` 폴더에 모델 파일이 있는지 확인
- 없다면 위의 "빠른 시작" 섹션 참조

### 패키지 설치 에러
```bash
pip install --upgrade pip
pip install -r requirements.txt

# CUDA 관련 에러 시
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 학습이 너무 느린 경우
- GPU가 제대로 사용되고 있는지 확인: `torch.cuda.is_available()` → `True`
- DataLoader의 `num_workers` 조정 (2~4 추천)
- Mixed Precision Training 사용: `torch.cuda.amp` 활용

### Validation Loss가 개선되지 않는 경우
- Learning rate를 낮추기 (1e-5)
- Epochs를 늘리기 (3-5)
- 데이터를 다시 확인 (클래스 불균형 등)
- Dropout을 낮추기 (0.1-0.2)

## FAQ (자주 묻는 질문)

### Q1: Getting Started Competition이란?
**A**: Kaggle이 머신러닝 초보자를 위해 만든 비경쟁 대회입니다.
- **목적**: 머신러닝 기초 개념 학습 및 Kaggle 플랫폼 익히기
- **특징**: 상금 없음, 무기한 운영, 커뮤니티 교류 중심
- **대상**: 데이터 사이언스 입문자 또는 MOOC 수강 완료자

### Q2: 팀은 어떻게 만드나요?
**A**: 대회 규칙에 동의하면 자동으로 개인 팀이 생성됩니다.
- **팀 관리**: More > Team 페이지에서 관리
- **팀 초대**: 다른 사람을 초대하거나 팀 합병 가능
- **팀 찾기**: Team 탭에서 팀원을 찾는 글 게시 가능
- **팀의 장점**: 새로운 기술을 배우고 즐겁게 경쟁하는 최고의 방법

### Q3: Kaggle Notebooks란?
**A**: 재현 가능하고 협업 가능한 클라우드 컴퓨팅 환경입니다.
- **지원 언어**: Python, R
- **지원 형식**: Jupyter Notebooks, RMarkdown
- **무료 리소스**: GPU 사용 가능
- **공유 기능**: Code 탭에서 다른 참가자의 노트북 확인 가능
- **학습 자료**: [Kaggle Courses](https://www.kaggle.com/learn) 참고

### Q4: 내 팀이 리더보드에서 사라졌어요!
**A**: 2개월 롤링 윈도우 때문입니다.
- **규칙**: 2개월 이상 오래된 제출은 자동으로 무효화됨
- **목적**: 리더보드를 관리 가능한 크기로 유지하고 최신 상태 유지
- **재등장 방법**: 새로운 제출을 하면 다시 리더보드에 나타남
- **자세한 설명**: [Rolling Leaderboard 결정 이유](https://www.kaggle.com/discussions)

### Q5: 도움이 필요하면 어떻게 하나요?
**A**: Discussion Forum을 활용하세요.
- **빠른 답변**: 전담 지원팀이 없으므로 포럼이 가장 빠름
- **유용한 정보**: 데이터, 평가 지표, 접근 방법에 대한 정보 가득
- **지식 공유**: 질문하고 답변하면서 함께 성장
- **전체 참가자 문제**: 모든 참가자에게 영향을 주는 문제만 Support 팀에 문의

### Q6: 로컬에서 학습한 모델을 제출할 수 있나요?
**A**: 아니요, Code Competition이므로 불가능합니다.
- ❌ **불가**: 로컬에서 학습한 모델 업로드 후 inference만 실행
- ✅ **가능**: Pre-trained 모델(DistilBERT 등)은 데이터셋으로 업로드 가능
- **이유**: 공정한 경쟁과 재현성을 위해 모든 학습이 Kaggle에서 실행되어야 함

### Q7: 제출한 노트북의 실행 시간이 매번 달라요
**A**: 정상입니다. 제출 실행 시간은 약간 난독화되어 있습니다.
- **변동 범위**: 동일한 노트북도 최대 15분 차이 발생
- **목적**: 하드웨어 상세 정보 보호
- **영향**: 성능 측정에는 영향 없음

### Q8: 인터넷이 비활성화되는데 어떻게 모델을 다운로드하나요?
**A**: 데이터셋으로 미리 업로드해야 합니다.
1. **로컬 다운로드**: `download_model.py`로 모델 다운로드
2. **압축**: `zip -r model.zip models/`
3. **Kaggle 업로드**: [Kaggle Datasets](https://www.kaggle.com/datasets)에 업로드
4. **노트북에 추가**: Add Data에서 업로드한 데이터셋 선택

### Q9: submission.csv를 어디에 저장해야 하나요?
**A**: `/kaggle/working/submission.csv`에 저장해야 합니다.
- **필수 경로**: Kaggle이 이 경로에서 제출 파일을 찾음
- **파일명**: 반드시 `submission.csv`여야 함
- **자동 감지**: 노트북 실행 완료 후 자동으로 평가

### Q10: 같은 실수를 반복하고 있는데 디버깅 방법은?
**A**: [Code Debugging Guide](https://www.kaggle.com/docs/competitions#code-debugging) 참고하세요.
- **일반적인 문제**: 경로 오류, 메모리 부족, 시간 초과
- **디버깅 팁**: print 문 활용, 작은 데이터셋으로 테스트, 에러 메시지 확인

## 참고 자료

### 공식 문서
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Kaggle Code Competitions](https://www.kaggle.com/docs/competitions#kernels-only-FAQ)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

### 논문
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108) - A distilled version of BERT
- [BERT Paper](https://arxiv.org/abs/1810.04805) - Pre-training of Deep Bidirectional Transformers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper

### 튜토리얼
- [BERT Explained](https://jalammar.github.io/illustrated-bert/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [HuggingFace Course](https://huggingface.co/learn/nlp-course/chapter1/1)

## Citation

이 프로젝트는 Kaggle의 LLM Classification Finetuning 대회를 기반으로 합니다.

```bibtex
@misc{llm-classification-finetuning,
    author = {Wei-lin Chiang and Lianmin Zheng and Lisa Dunlap and Joseph E. Gonzalez and Ion Stoica and Paul Mooney and Sohier Dane and Addison Howard and Nate Keating},
    title = {LLM Classification Finetuning},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/llm-classification-finetuning}},
    note = {Kaggle}
}
```

## 라이선스

교육 및 대회 참가 목적으로 사용됩니다.

---

**마지막 업데이트**: 2024년 10월
**대회 링크**: https://www.kaggle.com/competitions/llm-classification-finetuning
**Discussion Forum**: https://www.kaggle.com/competitions/llm-classification-finetuning/discussion
