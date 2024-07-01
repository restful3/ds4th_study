import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

class StockTradingEnv:
    """
    주식 거래 환경
    """
    def __init__(self, df, initial_balance=1000000):
        self.df = df
        self.initial_balance = initial_balance
        self.action_space = spaces.Discrete(3)  # 매수, 매도, 유지
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32)
        self.current_step = 0

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_value = self.initial_balance
        self.current_step = 0
        return self._get_observation()

    def _get_observation(self):
        obs = np.array([
            self.df.loc[self.current_step, 'Open'],
            self.df.loc[self.current_step, 'High'],
            self.df.loc[self.current_step, 'Low'],
            self.df.loc[self.current_step, 'Close'],
            self.balance,
            self.shares_held
        ])
        return obs

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Close']
        if action == 0:  # 매수
            shares_bought = self.balance // current_price
            self.balance -= shares_bought * current_price
            self.shares_held += shares_bought
        elif action == 1:  # 매도
            self.balance += self.shares_held * current_price
            self.shares_held = 0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        next_state = self._get_observation()
        self.total_value = self.balance + self.shares_held * current_price
        reward = self.total_value - self.initial_balance if done else 0

        return next_state, reward, done, {}

    def render(self):
        profit = self.total_value - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        print(f'Total value: {self.total_value}')
        print(f'Profit: {profit}')
