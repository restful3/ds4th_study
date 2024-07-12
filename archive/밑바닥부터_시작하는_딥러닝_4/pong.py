import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.00025
BUFFER_SIZE = 10000
BATCH_SIZE = 32
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 1000000
TARGET_UPDATE = 1000

# Environment
env = gym.make('PongNoFrameskip-v4')
env = gym.wrappers.AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
env = gym.wrappers.FrameStack(env, 4)

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Experience Replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)

# Epsilon Greedy Policy
class EpsilonGreedyPolicy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        self.steps_done = 0
    
    def get_epsilon(self):
        epsilon = self.end + (self.start - self.end) * np.exp(-1. * self.steps_done / self.decay)
        self.steps_done += 1
        return epsilon
    
    def select_action(self, state, policy_net, device):
        epsilon = self.get_epsilon()
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                return policy_net(state).argmax(dim=1).item()
        else:
            return random.randrange(env.action_space.n)

# Initialize everything
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = QNetwork((4, 84, 84), env.action_space.n).to(device)
target_net = QNetwork((4, 84, 84), env.action_space.n).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
replay_buffer = ReplayBuffer(BUFFER_SIZE)
epsilon_policy = EpsilonGreedyPolicy(EPSILON_START, EPSILON_END, EPSILON_DECAY)

# Training loop
def optimize_model():
    if len(replay_buffer) < BATCH_SIZE:
        return
    state, action, reward, next_state, done = replay_buffer.sample(BATCH_SIZE)
    state = torch.tensor(state, dtype=torch.float32, device=device) / 255.0
    action = torch.tensor(action, dtype=torch.long, device=device).unsqueeze(1)
    reward = torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(1)
    next_state = torch.tensor(next_state, dtype=torch.float32, device=device) / 255.0
    done = torch.tensor(done, dtype=torch.float32, device=device).unsqueeze(1)
    
    q_values = policy_net(state).gather(1, action)
    next_q_values = target_net(next_state).max(1)[0].unsqueeze(1)
    target_q_values = reward + (GAMMA * next_q_values * (1 - done))
    
    loss = nn.MSELoss()(q_values, target_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    for t in range(10000):
        action = epsilon_policy.select_action(state, policy_net, device)
        next_state, reward, done, truncated, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done or truncated)
        state = next_state
        total_reward += reward
        optimize_model()
        if done or truncated:
            break
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    print(f"Episode: {episode}, Total Reward: {total_reward}")

# Save the trained model
torch.save(policy_net.state_dict(), "pong_dqn.pth")

# Load the trained model and watch the agent play
policy_net.load_state_dict(torch.load("pong_dqn.pth"))

def play_game(env, policy_net, device):
    state = env.reset()
    total_reward = 0
    while True:
        env.render()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
        action = policy_net(state).argmax(dim=1).item()
        next_state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
        if done or truncated:
            break
    print(f"Total Reward: {total_reward}")

# Play the game with the trained agent
play_game(env, policy_net, device)

env.close()
