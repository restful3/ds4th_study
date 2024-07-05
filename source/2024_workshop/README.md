# 강화학습을 이용한 주식 거래 시스템

이 프로젝트는 강화학습 알고리즘을 사용하여 주식 거래 전략을 개발하고 평가하는 시스템입니다.

## 주요 기능

- 다양한 강화학습 알고리즘(DQN, PPO, A2C) 지원
- 주식 데이터 수집 및 전처리
- 모델 학습 및 저장
- 학습된 모델을 사용한 백테스팅 및 평가
- 결과 시각화

## 파일 구조

- `main.py`: 프로그램의 진입점. 학습 및 평가 프로세스를 제어합니다.
- `config.py`: 설정 파일. 모든 하이퍼파라미터와 설정을 관리합니다.
- `stocktrainer.py`: 주식 거래 모델 학습을 위한 클래스를 포함합니다.
- `stockevaluator.py`: 학습된 모델의 성능을 평가하는 클래스를 포함합니다.
- `stock_trading_env.py`: 주식 거래 환경을 시뮬레이션하는 클래스를 포함합니다.
- `agents/`: 다양한 강화학습 에이전트 구현을 포함하는 디렉토리
  - `dqn_agent.py`: DQN 알고리즘 구현
  - `ppo_agent.py`: PPO 알고리즘 구현
  - `a2c_agent.py`: A2C 알고리즘 구현

## 설치 방법

1. 필요한 패키지 설치:
pip install -r requirements.txt
## 사용 방법

프로그램은 명령줄 인자를 통해 다양한 옵션을 지정할 수 있습니다:
python main.py [--mode {train,evaluate,both}] [--agent {dqn,ppo,a2c}] [--ticker TICKER]
- `--mode`: 실행 모드 선택 (train, evaluate, both)
- `--agent`: 사용할 강화학습 알고리즘 선택 (dqn, ppo, a2c)
- `--ticker`: 분석할 주식의 종목 코드

예시:
python main.py --mode both --agent dqn --ticker 005930
## 결과

- 학습된 모델은 `models/` 디렉토리에 저장됩니다.
- 평가 결과 그래프는 `results/` 디렉토리에 저장됩니다.

## 주의사항

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다. 실제 투자에 사용할 경우 주의가 필요합니다.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.