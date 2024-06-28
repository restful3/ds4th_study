import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L

class Policy(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x
    
class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2
        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.pi)

    def get_action(self, state):
        state = state[np.newaxis, :]
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p = probs.data)
        return action, probs[action]
    
    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)

    def update(self):
        self.pi.cleargrads()

        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma*G
            loss += -F.log(prob) * G

        loss.backward()
        self.optimizer.update()
        self.memory = []

    

if __name__ == '__main__':

    episodes = 3000
    env = gym.make('CartPole-v0', render_mode='rgb_array')
    agent = Agent()
    reward_history = []

    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0

        while not done:
            action, prob = agent.get_action(state)  # 행동 선택
            next_state, reward, terminated, truncated, info = env.step(action)  # 행동 수행
            done = terminated | truncated

            agent.add(reward, prob)  # 보상과 행동의 확률을 에이전트에 추가
            state = next_state       # 상태 전이
            total_reward += reward   # 보상 총합 계산

        agent.update()  # 정책 갱신

        reward_history.append(total_reward)
        if episode % 100 == 0:
            print("episode :{}, total reward : {:.1f}".format(episode, total_reward))


    # [그림 9-2] 에피소드별 보상 합계 추이
    from common.utils import plot_total_reward
    plot_total_reward(reward_history)
