import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class A2CNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(A2CNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 1)
        )

    def forward(self, state):
        shared = self.shared(state)
        return self.actor(shared), self.critic(shared)

class A2CAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, entropy_coef=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.network = A2CNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        self.states = []
        self.actions = []
        self.rewards = []

    def act(self, state):
        state = torch.FloatTensor(state).to(device)
        probs, _ = self.network(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item()

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def train(self):
        states = torch.FloatTensor(self.states).to(device)
        actions = torch.LongTensor(self.actions).to(device)
        rewards = torch.FloatTensor(self.rewards).to(device)

        # Compute returns
        returns = []
        R = 0
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Compute actor and critic losses
        probs, values = self.network(states)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        advantages = returns - values.squeeze()

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy = dist.entropy().mean()

        loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []

    def save(self, path):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])