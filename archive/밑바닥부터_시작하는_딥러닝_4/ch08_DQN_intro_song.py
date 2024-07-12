import gym

env = gym.make('CartPole-v0', render_mode='human')

state = env.reset()[0]
print(f"상태 : {state}")

action_space = env.action_space
print(f'행동의 차원 수 : {action_space}')

action = 0
next_state, reward, terminated, truncated, info = env.step(action)
print(next_state)