import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, 
                 memory_size=2000, batch_size=32, target_update_frequency=100):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_update_frequency = target_update_frequency
        
        self.qnet = DQN(state_size, action_size).to(device)
        self.qnet_target = DQN(state_size, action_size).to(device)
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.qnet_target.eval()
        
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(memory_size)
        self.update_counter = 0

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.qnet(state).squeeze(0)
        return torch.argmax(q_values).item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # 현재 상태의 Q 값 계산 (qnet 사용)
        current_q_values = self.qnet(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 다음 상태의 최대 Q 값 계산 (qnet_target 사용)
        with torch.no_grad():
            next_q_values = self.qnet_target(next_states).max(1)[0]

        # TD 타겟 계산
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # 손실 계산
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # 최적화 단계
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 엡실론 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 목표 신경망 업데이트
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.qnet_target.load_state_dict(self.qnet.state_dict())

    def save(self, path):
        torch.save({
            'qnet_state_dict': self.qnet.state_dict(),
            'qnet_target_state_dict': self.qnet_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.qnet.load_state_dict(checkpoint['qnet_state_dict'])
        self.qnet_target.load_state_dict(checkpoint['qnet_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.qnet.eval()
        self.qnet_target.eval()