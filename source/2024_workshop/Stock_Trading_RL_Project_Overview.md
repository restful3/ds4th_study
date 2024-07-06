# 프로젝트의 전체 파일 구조

```markdown
project_root/
├── main.py
├── config.py
├── stocktrainer.py
├── stockevaluator.py
├── stock_trading_env.py
├── agents/
│   ├── __init__.py
│   ├── dqn_agent.py
│   ├── ppo_agent.py
│   └── a2c_agent.py
├── models/
│   └── (저장된 모델 파일들)
└── results/
    └── (평가 결과 그래프 파일들)
```

### 파일 및 디렉토리 설명:

1. `main.py`
   - 프로그램의 진입점
   - 명령줄 인자를 파싱하고 전체 실행 흐름을 관리

2. `config.py`
   - 프로젝트의 전역 설정을 관리
   - 학습 및 평가에 필요한 파라미터들을 정의

3. `stocktrainer.py`
   - `StockTrainer` 클래스 정의
   - 주식 데이터를 가져오고 강화학습 모델을 학습시키는 로직 포함

4. `stockevaluator.py`
   - `StockEvaluator` 클래스 정의
   - 학습된 모델을 평가하고 결과를 시각화하는 로직 포함

5. `stock_trading_env.py`
   - `StockTradingEnv` 클래스 정의
   - OpenAI Gym 스타일의 주식 거래 환경 구현

6. `agents/` 디렉토리
   - 다양한 강화학습 알고리즘 에이전트들을 포함
   - `dqn_agent.py`: DQN 알고리즘 구현
   - `ppo_agent.py`: PPO 알고리즘 구현 (가정)
   - `a2c_agent.py`: A2C 알고리즘 구현 (가정)

7. `models/` 디렉토리
   - 학습된 모델 파일들이 저장되는 위치

8. `results/` 디렉토리
   - 평가 결과 그래프 등이 저장되는 위치

이 구조는 모듈화된 설계를 따르고 있어, 각 컴포넌트가 명확한 역할을 가지고 있습니다. `main.py`를 통해 전체 시스템을 제어하고, `config.py`로 설정을 중앙에서 관리하며, 개별 에이전트들은 `agents/` 디렉토리에 분리되어 있어 새로운 알고리즘을 쉽게 추가할 수 있습니다. 학습된 모델과 평가 결과는 각각 `models/`와 `results/` 디렉토리에 저장되어 관리가 용이합니다.

# config.py

`config.py` 파일은 프로젝트의 모든 설정을 중앙에서 관리하는 역할을 합니다. 이 파일은 학습, 평가, 그리고 각 강화학습 알고리즘에 필요한 모든 매개변수를 포함하고 있습니다.

## 주요 구성 요소

### 1. config 딕셔너리

이 딕셔너리는 모든 설정을 포함합니다:

```python
config = {
    # 주식 관련 설정
    'ticker': "005930",  # 삼성전자
    'start_date': datetime(2020, 1, 1),
    'end_date': datetime(2023, 12, 31),
    'initial_balance': 10_000_000,  # 초기 자금 1000만원
    'commission_rate': 0.00015,  # 거래 수수료 0.015%

    # 학습 관련 설정
    'epochs': 10,
    'batch_size': 32,

    # 에이전트 관련 설정
    'agent_type': 'dqn',  # 'dqn', 'ppo', 또는 'a2c'
    'state_size': 6,
    'action_size': 4,

    # 공통 하이퍼파라미터
    'learning_rate': 0.001,
    'gamma': 0.95,

    # 알고리즘별 특정 설정
    # ... (DQN, PPO, A2C 각각의 특정 설정)

    # 평가 관련 설정
    'eval_episodes': 10,
    'eval_start_date': "20240101",
    'eval_end_date': date.today().strftime("%Y%m%d"),
    'results_dir': 'results',
}
```

### 2. get_agent_config 함수

이 함수는 특정 에이전트 유형에 필요한 설정만을 추출합니다:

```python
def get_agent_config(agent_type):
    agent_config = {
        'state_size': config['state_size'],
        'action_size': config['action_size'],
        'learning_rate': config['learning_rate'],
        'gamma': config['gamma'],
    }

    if agent_type == 'dqn':
        agent_config.update({
            'epsilon': config['epsilon'],
            'epsilon_decay': config['epsilon_decay'],
            'epsilon_min': config['epsilon_min'],
            'memory_size': config['memory_size'],
        })
    elif agent_type == 'ppo':
        # PPO 특정 설정
    elif agent_type == 'a2c':
        # A2C 특정 설정
    
    return agent_config
```

### 3. update_config 함수

이 함수는 실행 중에 설정을 동적으로 업데이트할 수 있게 해줍니다:

```python
def update_config(updates):
    config.update(updates)
```

## 사용법

다른 파일에서 `config.py`를 import하여 설정을 사용할 수 있습니다:

```python
from config import config, get_agent_config, update_config

# 설정 사용
ticker = config['ticker']

# 에이전트 특정 설정 가져오기
dqn_config = get_agent_config('dqn')

# 설정 업데이트
update_config({'epochs': 20})
```

이 파일은 프로젝트의 모든 설정을 중앙에서 관리함으로써 일관성을 유지하고 설정 변경을 쉽게 만듭니다. 또한, 각 강화학습 알고리즘에 필요한 특정 설정을 쉽게 추출할 수 있게 해줍니다.

# stock_trading_env.py

이 파일은 주식 거래 시뮬레이션을 위한 강화학습 환경을 구현합니다. OpenAI Gym 스타일의 인터페이스를 따르고 있어, 다양한 강화학습 알고리즘과 호환됩니다.

## 주요 구성 요소

### StockTradingEnv 클래스

이 클래스는 주식 거래 환경의 핵심입니다.

```python
class StockTradingEnv:
    def __init__(self, df, initial_balance=1000000, commission_rate=0.00015):
        # 초기화 코드
```

#### 주요 속성:
- `df`: 주식 데이터를 담고 있는 DataFrame
- `initial_balance`: 초기 현금 잔액
- `commission_rate`: 거래 수수료율
- `balance`: 현재 현금 잔액
- `shares_held`: 현재 보유 주식 수
- `current_step`: 현재 거래일의 인덱스

#### 주요 메서드:

1. `reset()`
   - 환경을 초기 상태로 리셋합니다.
   - 반환값: 초기 상태 (observation)

2. `_get_observation()`
   - 현재 상태(observation)를 반환합니다.
   - 상태는 [현재 잔액, 보유 주식 수, 현재 주가, 시가, 고가, 저가]를 포함합니다.

3. `step(action)`
   - 에이전트의 행동을 처리하고 다음 상태로 이동합니다.
   - 매개변수: `action` (행동 유형과 양을 포함하는 리스트)
   - 반환값: (새로운 상태, 보상, 종료 여부, 추가 정보)

4. `backtest(model)` : (사용되고 있지 않음, 수정 필요 할지도)
   - 학습된 모델을 사용하여 백테스트를 수행합니다.
   - 매개변수: `model` (학습된 에이전트 모델)
   - 반환값: 포트폴리오 가치의 시계열 데이터

5. `calculate_sharpe_ratio(portfolio_values, risk_free_rate=0.01)`
   - 백테스트 결과를 바탕으로 Sharpe Ratio를 계산합니다.
   - 매개변수: 
     - `portfolio_values`: 포트폴리오 가치의 시계열 데이터
     - `risk_free_rate`: 무위험 이자율 (연간)
   - 반환값: Sharpe Ratio

## 주요 기능 상세 설명

### 1. 상태 표현
환경은 6개의 값으로 구성된 상태를 제공합니다:
- 현재 현금 잔액
- 보유 주식 수
- 현재 주가 (종가)
- 시가
- 고가
- 저가

이 상태 표현은 에이전트가 현재 시장 상황과 자신의 포지션을 파악하는 데 필요한 핵심 정보를 제공합니다.

### 2. 행동 처리
`step()` 메서드는 에이전트의 행동을 처리합니다. 행동은 두 부분으로 구성됩니다:
- 행동 유형 (0: 홀딩, 1: 매수, 2: 매도)
- 행동 양 (0과 1 사이의 값, 현금이나 주식의 어느 비율을 사용할지 결정)

매수나 매도 시 거래 수수료가 적용되며, 가능한 범위 내에서만 거래가 실행됩니다.

### 3. 보상 계산
각 스텝의 보상은 포트폴리오 가치의 상대적 변화로 계산됩니다:
```python
reward = (self.balance + self.shares_held * current_price - self.initial_balance) / self.initial_balance
```
이는 초기 자본 대비 현재 포트폴리오 가치의 변화율을 나타냅니다.

### 4. 백테스트
`backtest()` 메서드는 학습된 모델을 사용하여 전체 데이터셋에 대해 시뮬레이션을 수행합니다. 이를 통해 모델의 성능을 평가할 수 있습니다.

### 5. Sharpe Ratio 계산
`calculate_sharpe_ratio()` 메서드는 백테스트 결과를 바탕으로 Sharpe Ratio를 계산합니다. 이는 리스크 조정 수익률을 평가하는 중요한 지표입니다.

## 사용 예시

```python
# 환경 생성
env = StockTradingEnv(stock_data, initial_balance=1000000)

# 에피소드 실행
state = env.reset()
done = False
while not done:
    action = agent.get_action(state)  # 에이전트로부터 행동 선택
    next_state, reward, done, _ = env.step(action)
    state = next_state

# 백테스트 수행
portfolio_values = env.backtest(trained_model)

# Sharpe Ratio 계산
sharpe_ratio = env.calculate_sharpe_ratio(portfolio_values)
```

이 환경은 실제 주식 시장의 복잡성을 단순화하면서도, 강화학습 에이전트가 의미 있는 거래 전략을 학습할 수 있는 충분한 정보와 상호작용을 제공합니다.

# stocktrainer.py

이 파일은 주식 거래를 위한 강화학습 에이전트를 학습시키는 프로세스를 관리합니다. 다양한 강화학습 알고리즘(DQN, PPO, A2C)을 지원하며, 설정 파일을 통해 유연하게 매개변수를 조정할 수 있습니다.

## 주요 구성 요소

### StockTrainer 클래스

이 클래스는 전체 학습 프로세스를 관리합니다.

#### 주요 속성:
- 모든 설정값들은 `config.py`에서 가져옵니다 (ticker, 날짜, 초기 잔액, 수수료율 등)
- `agent`: 선택된 강화학습 에이전트 인스턴스
- `env`: 주식 거래 환경 인스턴스

#### 주요 메서드:

1. `get_stock_data()`
   - pykrx 라이브러리를 사용하여 지정된 기간의 주식 데이터를 가져옵니다.

2. `create_agent()`
   - 선택된 에이전트 타입(DQN, PPO, A2C)에 따라 적절한 에이전트 인스턴스를 생성합니다.

3. `train()`
   - 주식 거래 에이전트를 학습시킵니다.
   - 각 에피소드마다 전체 데이터셋에 대해 시뮬레이션을 수행하고 에이전트를 업데이트합니다.

4. `get_company_name(ticker)`
   - 주식 종목 코드를 회사 이름으로 변환합니다.

5. `save_model()`
   - 학습된 모델을 파일로 저장합니다. 파일명에는 회사 이름과 사용된 에이전트 타입이 포함됩니다.

## 주요 기능 상세 설명

### 1. 에이전트 생성
`create_agent()` 메서드는 설정된 에이전트 타입에 따라 적절한 에이전트를 생성합니다:

```python
def create_agent(self):
    agent_config = get_agent_config(self.agent_type)
    if self.agent_type == 'dqn':
        return DQNAgent(**agent_config)
    elif self.agent_type == 'ppo':
        return PPOAgent(**agent_config)
    elif self.agent_type == 'a2c':
        return A2CAgent(**agent_config)
    else:
        raise ValueError(f"Unsupported agent type: {self.agent_type}")
```

### 2. 학습 프로세스
`train()` 메서드는 전체 학습 프로세스를 관리합니다. 주요 단계는 다음과 같습니다:

1. 주식 데이터 가져오기
2. 거래 환경 초기화
3. 에이전트 생성
4. 지정된 에포크 수만큼 반복:
   - 환경 리셋
   - 각 거래일마다:
     - 에이전트로부터 행동 선택
     - 환경에서 행동 실행 및 새로운 상태, 보상 받기
     - 에이전트 타입에 따라 다른 학습 방식 적용
   - 에피소드 종료 시 총 보상과 최종 잔액 출력

### 3. 모델 저장
`save_model()` 메서드는 학습된 모델을 파일로 저장합니다. 파일명에는 회사 이름과 사용된 에이전트 타입이 포함되어, 나중에 모델을 쉽게 식별할 수 있습니다.

```python
def save_model(self):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    company_name = self.get_company_name(self.ticker)
    company_name = ''.join(e for e in company_name if e.isalnum() or e.isspace())
    company_name = company_name.replace(" ", "_")
    
    model_path = os.path.join(models_dir, f'{company_name}_{self.agent_type}.pth')
    self.agent.save(model_path)
    print(f"Model saved to {model_path}")
```

## 사용 방법

스크립트 실행 시, `config.py`에서 정의된 설정을 사용하여 StockTrainer 인스턴스를 생성하고, 학습을 수행한 후 모델을 저장합니다.

```python
if __name__ == "__main__":
    # 필요한 경우 설정 업데이트
    # update_config({'epochs': 200, 'learning_rate': 0.0005})

    trainer = StockTrainer()
    trainer.train()
    trainer.save_model()
```

이 파일은 다양한 강화학습 알고리즘을 사용하여 주식 거래 전략을 학습할 수 있는 유연한 프레임워크를 제공합니다. 설정 파일을 통해 다양한 매개변수를 쉽게 조정할 수 있어, 다양한 실험을 수행하기에 적합합니다.

# stockevaluator.py

이 파일은 학습된 주식 거래 강화학습 모델을 평가하는 기능을 제공합니다. 학습된 모델을 로드하고, 새로운 데이터에 대해 성능을 평가한 후, 결과를 시각화합니다.

## 주요 구성 요소

### StockEvaluator 클래스

이 클래스는 모델 평가 프로세스를 관리합니다.

#### 주요 속성:
- 대부분의 설정값은 `config.py`에서 가져옵니다 (평가 기간, 초기 잔액, 상태 및 행동 크기 등)
- `agent`: 로드된 강화학습 에이전트 인스턴스
- `env`: 주식 거래 환경 인스턴스
- `test_data`: 평가에 사용될 주식 데이터
- `portfolio_values`: 평가 과정에서의 포트폴리오 가치 변화
- `actions`: 에이전트가 취한 행동들의 기록

#### 주요 메서드:

1. `get_stock_data()`
   - pykrx 라이브러리를 사용하여 평가 기간의 주식 데이터를 가져옵니다.

2. `get_company_name()`
   - 주식 종목 코드를 회사 이름으로 변환합니다.

3. `load_model()`
   - 학습된 모델을 파일에서 로드합니다.

4. `evaluate()`
   - 로드된 모델을 사용하여 테스트 데이터에 대해 평가를 수행합니다.

5. `plot_evaluation()`
   - 평가 결과를 그래프로 시각화하고 파일로 저장합니다.

6. `run_evaluation()`
   - 전체 평가 프로세스를 실행합니다.

## 주요 기능 상세 설명

### 1. 모델 로딩
`load_model()` 메서드는 설정된 에이전트 타입에 따라 적절한 에이전트를 생성하고, 저장된 모델 파일을 로드합니다:

```python
def load_model(self):
    agent_config = get_agent_config(self.agent_type)
    if self.agent_type == 'dqn':
        self.agent = DQNAgent(**agent_config)
    elif self.agent_type == 'ppo':
        self.agent = PPOAgent(**agent_config)
    elif self.agent_type == 'a2c':
        self.agent = A2CAgent(**agent_config)
    else:
        raise ValueError(f"Unsupported agent type: {self.agent_type}")

    # 모델 파일 경로 설정 및 로딩
    ...
```

### 2. 평가 프로세스
`evaluate()` 메서드는 로드된 모델을 사용하여 테스트 데이터에 대해 평가를 수행합니다:

1. 테스트 데이터 로드
2. 거래 환경 초기화
3. 각 거래일마다:
   - 에이전트로부터 행동 선택
   - 환경에서 행동 실행
   - 포트폴리오 가치 및 행동 기록

### 3. 결과 시각화
`plot_evaluation()` 메서드는 평가 결과를 그래프로 시각화합니다:

- 주가와 포트폴리오 가치를 동시에 표시
- 에이전트의 매수, 매도, 홀딩 행동을 그래프 상에 표시
- 결과를 PNG 파일로 저장

```python
def plot_evaluation(self):
    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax2 = ax1.twinx()
    
    # 주가와 포트폴리오 가치 플로팅
    ...
    
    # 매수, 매도, 홀딩 행동 표시
    ...
    
    # 그래프 설정 및 저장
    ...
```

## 사용 방법

스크립트 실행 시, `config.py`에서 정의된 설정을 사용하여 StockEvaluator 인스턴스를 생성하고, 평가를 수행합니다:

```python
if __name__ == "__main__":
    ticker = config['ticker']
    evaluator = StockEvaluator(ticker)
    evaluator.run_evaluation()
```

이 파일은 학습된 모델의 성능을 실제 데이터로 평가하고, 그 결과를 시각적으로 표현하는 중요한 역할을 합니다. 이를 통해 모델의 실제 성능을 분석하고, 필요한 경우 모델이나 학습 과정을 개선할 수 있습니다.

# dqn_agent.py

이 파일은 Deep Q-Network (DQN) 알고리즘을 구현한 에이전트를 정의합니다. DQN은 Q-learning을 심층 신경망과 결합한 강화학습 알고리즘입니다.

## 주요 구성 요소

### 1. DQN 클래스

신경망 모델을 정의합니다.

```python
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(0)
```

- 3층의 완전연결층으로 구성
- ReLU 활성화 함수 사용

### 2. DQNAgent 클래스

DQN 알고리즘을 구현한 에이전트를 정의합니다.

#### 주요 속성:
- `memory`: 경험 리플레이를 위한 메모리 버퍼
- `gamma`: 할인율
- `epsilon`, `epsilon_decay`, `epsilon_min`: 입실론-그리디 정책을 위한 파라미터
- `model`: DQN 신경망 모델
- `optimizer`: Adam 옵티마이저

#### 주요 메서드:

1. `remember(state, action, reward, next_state, done)`
   - 경험을 메모리에 저장합니다.

2. `act(state)`
   - 현재 상태에서 행동을 선택합니다.
   - 입실론-그리디 정책을 사용합니다.

3. `replay(batch_size)`
   - 경험 리플레이를 수행하여 모델을 학습시킵니다.

4. `save(path)` 와 `load(path)`
   - 모델을 저장하고 불러옵니다.

## 주요 기능 상세 설명

### 1. 행동 선택 (act 메서드)

```python
def act(self, state):
    if np.random.rand() <= self.epsilon:
        return [np.random.randint(0, self.action_size - 1), np.random.rand()]
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = self.model(state).squeeze(0)
    if q_values.shape[0] == 0:
        return [np.random.randint(0, self.action_size - 1), np.random.rand()]
    action_type = torch.argmax(q_values[:-1]).item()
    action_amount = torch.sigmoid(q_values[-1]).item()
    return [action_type, max(0.01, action_amount)]
```

- 입실론 확률로 무작위 행동을 선택합니다.
- 그렇지 않으면 신경망을 통해 Q-값을 계산하고 최적의 행동을 선택합니다.
- 행동은 [행동 유형, 행동 양]의 형태로 반환됩니다.

### 2. 경험 리플레이 (replay 메서드)

```python
def replay(self, batch_size):
    minibatch = random.sample(self.memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        # ... (중략)
        target = reward
        if not done.item():
            target = reward + self.gamma * torch.max(next_q_values)
        # ... (중략)
        loss = nn.MSELoss()(current_q_values, target_f)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    # 입실론 감소
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay
```

- 메모리에서 무작위로 배치를 샘플링합니다.
- 각 샘플에 대해 Q-learning 업데이트를 수행합니다.
- 손실 함수로 MSE를 사용하여 모델을 학습시킵니다.
- 학습 후 입실론 값을 감소시킵니다.

이 DQN 에이전트는 연속적인 행동 공간을 처리할 수 있도록 설계되어 있습니다. 행동은 유형(예: 매수, 매도, 홀딩)과 양(0~1 사이의 값)으로 구성되어 있어, 주식 거래와 같은 복잡한 환경에서 유연하게 대응할 수 있습니다.

이 구현은 기본적인 DQN 알고리즘을 따르고 있지만, 더블 DQN, 우선순위 경험 리플레이 등의 고급 기법을 추가하여 성능을 더욱 향상시킬 수 있습니다.

# main.py

이 파일은 프로그램의 진입점으로, 주식 거래 강화학습 시스템의 전체 실행 흐름을 관리합니다. 사용자 입력을 처리하고, 학습과 평가 과정을 조정합니다.

## 주요 구성 요소

### 1. 필요한 모듈 임포트
```python
import argparse
from config import config, update_config
from stocktrainer import StockTrainer
from stockevaluator import StockEvaluator
import logging
```

### 2. 로깅 설정 함수
```python
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
```
- 로그 레벨을 INFO로 설정하고, 시간, 로그 레벨, 메시지를 포함하는 포맷을 지정합니다.

### 3. 명령줄 인자 파싱 함수
```python
def parse_arguments():
    parser = argparse.ArgumentParser(description='Stock Trading RL')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'both'], default='both', help='Operation mode')
    parser.add_argument('--agent', choices=['dqn', 'ppo', 'a2c'], default='dqn', help='RL agent type')
    parser.add_argument('--ticker', type=str, default='005930', help='Stock ticker')
    return parser.parse_args()
```
- 사용자가 지정할 수 있는 인자:
  - `mode`: 학습, 평가, 또는 둘 다 수행
  - `agent`: 사용할 강화학습 에이전트 유형
  - `ticker`: 대상 주식의 종목 코드

### 4. 메인 함수
```python
def main():
    setup_logging()
    args = parse_arguments()
    
    update_config({'agent_type': args.agent, 'ticker': args.ticker})
    
    if args.mode in ['train', 'both']:
        trainer = StockTrainer()
        trainer.train()
        trainer.save_model()
        
    if args.mode in ['evaluate', 'both']:
        evaluator = StockEvaluator(args.ticker)
        evaluator.run_evaluation()
    
    logging.info("Process completed successfully.")
```
- 로깅을 설정하고 명령줄 인자를 파싱합니다.
- 설정을 업데이트합니다.
- 선택된 모드에 따라 학습 및/또는 평가를 수행합니다.
- 프로세스 완료 시 로그를 남깁니다.

## 실행 흐름

1. 프로그램이 시작되면 로깅이 설정됩니다.
2. 명령줄 인자가 파싱됩니다.
3. 파싱된 인자를 바탕으로 설정이 업데이트됩니다.
4. 선택된 모드에 따라:
   - 'train' 또는 'both' 모드: StockTrainer를 사용하여 모델을 학습하고 저장합니다.
   - 'evaluate' 또는 'both' 모드: StockEvaluator를 사용하여 모델을 평가합니다.
5. 모든 프로세스가 완료되면 성공 메시지를 로깅합니다.

## 사용 방법

커맨드 라인에서 다음과 같이 실행할 수 있습니다:

```
python main.py --mode train --agent dqn --ticker 005930
```

이 명령은 DQN 에이전트를 사용하여 삼성전자(005930) 주식에 대해 학습을 수행합니다.

이 `main.py` 파일은 전체 시스템의 진입점 역할을 하며, 사용자가 쉽게 다양한 설정으로 시스템을 실행할 수 있게 해줍니다. 또한, 로깅을 통해 프로세스의 진행 상황을 모니터링할 수 있게 합니다.


```python

```
