import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    """
    DQN 클래스는 Q-네트워크 모델을 정의합니다.

    Attributes:
        fc1 (Linear): 첫 번째 완전 연결 레이어
        fc2 (Linear): 두 번째 완전 연결 레이어
        fc3 (Linear): 출력 레이어
    """

    def __init__(self, state_size, action_size):
        """
        DQN 초기화 메서드

        Args:
            state_size (int): 상태의 크기
            action_size (int): 행동의 크기
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        """
        순전파를 정의합니다.

        Args:
            x (Tensor): 입력 상태

        Returns:
            Tensor: Q-값
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    """
    DQNAgent 클래스는 강화학습을 통해 주식 거래를 학습하는 에이전트를 구현합니다.

    Attributes:
        state_size (int): 상태의 크기
        action_size (int): 행동의 크기
        memory (deque): 리플레이 메모리
        gamma (float): 할인 계수
        epsilon (float): 탐험 계수
        epsilon_decay (float): 탐험 계수 감소율
        epsilon_min (float): 최소 탐험 계수
        learning_rate (float): 학습률
        model (DQN): Q-네트워크 모델
        optimizer (Adam): 옵티마이저
    """

    def __init__(self, state_size, action_size):
        """
        DQNAgent 초기화 메서드

        Args:
            state_size (int): 상태의 크기
            action_size (int): 행동의 크기
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        """
        리플레이 메모리에 경험을 저장합니다.

        Args:
            state (numpy array): 현재 상태
            action (int): 행동
            reward (float): 보상
            next_state (numpy array): 다음 상태
            done (bool): 종료 여부
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        현재 상태에서 행동을 선택합니다.

        Args:
            state (numpy array): 현재 상태

        Returns:
            int: 선택한 행동
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        """
        리플레이 메모리에서 샘플을 무작위로 선택하여 학습합니다.

        Args:
            batch_size (int): 배치 크기
        """
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            reward = torch.FloatTensor([reward])
            done = torch.FloatTensor([done])

            # 현재 상태에서 Q-값 계산
            current_q_values = self.model(state)
            current_q_values = current_q_values.squeeze(0)  # 차원 축소

            # 다음 상태에서 Q-값 계산
            target_q_values = self.model(next_state).detach()
            target_q_values = target_q_values.squeeze(0)  # 차원 축소

            # 목표 Q-값 계산
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(target_q_values)

            # 현재 Q-값을 클론하고 업데이트
            target_f = current_q_values.clone()
            target_f[action] = target

            self.optimizer.zero_grad()
            loss = nn.MSELoss()(current_q_values, target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
