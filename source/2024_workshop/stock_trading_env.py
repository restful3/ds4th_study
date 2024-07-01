import numpy as np
\
class StockTradingEnv:
    def __init__(self, df, initial_balance=1000000):
        self.df = df
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.total_steps = len(self.df) - 1
        return self._get_observation()

    def _get_observation(self):
        return np.array([
            self.balance, 
            self.shares_held, 
            self.df.loc[self.current_step, 'Close'], 
            self.df.loc[self.current_step, 'Open'], 
            self.df.loc[self.current_step, 'High'], 
            self.df.loc[self.current_step, 'Low']
        ])

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Close']
        self.current_step += 1
        done = self.current_step == self.total_steps

        action_type = action[0]  # 0: Hold, 1: Buy, 2: Sell
        action_amount = action[1]  # Fraction of balance or shares to use

        if action_type == 1:  # Buy
            amount_to_invest = self.balance * action_amount
            shares_to_buy = int(amount_to_invest // current_price)
            self.balance -= shares_to_buy * current_price
            self.shares_held += shares_to_buy

        elif action_type == 2:  # Sell
            shares_to_sell = int(self.shares_held * action_amount)
            self.balance += shares_to_sell * current_price
            self.shares_held -= shares_to_sell

        reward = self.balance + self.shares_held * current_price - self.initial_balance if done else 0

        return self._get_observation(), reward, done, {}

