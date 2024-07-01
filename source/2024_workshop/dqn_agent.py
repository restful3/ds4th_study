import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = learning_rate
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return [random.choice([0, 1, 2]), random.uniform(0, 1)]
        state = torch.FloatTensor(state)
        q_values = self.model(state)
        action_type = torch.argmax(q_values[0][:3]).item()
        action_amount = torch.sigmoid(q_values[0][3]).item()
        return [action_type, action_amount]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            reward = torch.FloatTensor([reward])
            done = torch.FloatTensor([done])

            current_q_values = self.model(state)
            current_q_values = current_q_values.squeeze(0)

            target_q_values = self.model(next_state).detach()
            target_q_values = target_q_values.squeeze(0)

            target = reward
            if not done:
                target = reward + self.gamma * torch.max(target_q_values)

            target_f = current_q_values.clone()
            target_f[action[0]] = target

            self.optimizer.zero_grad()
            loss = nn.MSELoss()(current_q_values, target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
