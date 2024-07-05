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
        return self.fc3(x).squeeze(0)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=2000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        action_type = int(action[0])
        action_amount = float(action[1])
        self.memory.append((state, (action_type, action_amount), reward, next_state, done))

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
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            reward = torch.FloatTensor([reward]).to(device)
            done = torch.FloatTensor([done]).to(device)

            current_q_values = self.model(state).squeeze(0)
            next_q_values = self.model(next_state).squeeze(0)

            target = reward
            if not done.item():
                target = reward + self.gamma * torch.max(next_q_values)

            target_f = current_q_values.clone()
            action_type = int(action[0])

            if action_type >= len(target_f):
                print(f"Error: action_type {action_type} is out of bounds for target_f with size {len(target_f)}")
                continue
            
            target_f[action_type] = target.item()

            if len(target_f) <= action_type + 1:
                print(f"Error: action amount index is out of bounds for target_f with size {len(target_f)}")
                continue
            
            target_f[action_type + 1] = action[1]

            loss = nn.MSELoss()(current_q_values, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()