import numpy as np
import gym


env = gym.make('CartPole-v0', render_mode='human')
state = env.reset()[0]
done = False

while not done:  # 에피소드가 끝날 때까지 반복
    env.render()  # 진행 과정 시각화
    action = np.random.choice([0, 1])  # 행동 선택(무작위)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated | truncated      # 둘 중 하나만 True면 에피소드 종료
env.close()
