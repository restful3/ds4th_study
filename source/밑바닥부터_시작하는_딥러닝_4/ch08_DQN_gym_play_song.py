import numpy as np
import gym

env  = gym.make('CartPole-v0', render_mode='human')
state = env.reset()[0]
done = False

while not done:
    env.render()
    action = np.random.choice([0, 1])
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated | truncated 

env.close()    
