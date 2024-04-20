import numpy as np

# 기본 구현
np.random.seed(0)  # 시드 고정
rewards = []

for n in range(1, 11):  # 10번 플레이
    reward = np.random.rand()  # 보상(무작위수로 시뮬레이션)
    rewards.append(reward)
    Q = sum(rewards) / n
    print(Q)

print('---')

# 증분 구현
np.random.seed(0)
Q = 0

for n in range(1, 11):
    reward = np.random.rand()
    Q = Q + (reward - Q) / n  # [식 1.5]
    print(Q)
