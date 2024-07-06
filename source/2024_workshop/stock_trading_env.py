import numpy as np

HOLD = 0
BUY = 1
SELL = 2

class StockTradingEnv:
    def __init__(self, df, initial_balance=1000000, commission_rate=0.00015):
        self.df = df
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
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
            self.df.iloc[self.current_step]['Close'],
            self.df.iloc[self.current_step]['Open'],
            self.df.iloc[self.current_step]['High'],
            self.df.iloc[self.current_step]['Low']
        ])
    
    # 캘리 공식 
    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Close']
        self.current_step += 1
        done = self.current_step == self.total_steps

        # Kelly Criterion parameters (example values, these should be based on your model/strategy)
        b = 1  # Example: 100% expected return
        p = 0.6  # Example: 60% chance of winning (price going up)
        q = 1 - p

        # Calculate the optimal fraction to invest using Kelly Criterion
        f_star = (b * p - q) / b

        if action == BUY:
            # Calculate the amount of balance to be used according to Kelly Criterion
            balance_to_use = self.balance * f_star
            shares_to_buy = int(balance_to_use // current_price)
            cost = shares_to_buy * current_price
            commission = cost * self.commission_rate
            total_cost = cost + commission

            if total_cost <= self.balance:
                self.balance -= total_cost
                self.shares_held += shares_to_buy
            else:
                # Fallback to use the available balance
                shares_to_buy = int(self.balance // current_price)
                cost = shares_to_buy * current_price
                commission = cost * self.commission_rate
                total_cost = cost + commission
                self.balance -= total_cost
                self.shares_held += shares_to_buy

        elif action == SELL:
            if self.shares_held > 0:
                # Calculate the maximum shares to sell according to Kelly Criterion
                shares_to_sell = int(self.shares_held * f_star)
                sale_value = shares_to_sell * current_price
                commission = sale_value * self.commission_rate

                self.balance += sale_value - commission
                self.shares_held -= shares_to_sell
        # HOLD action does nothing
        reward = (self.balance + self.shares_held * current_price - self.initial_balance) / self.initial_balance
        return self._get_observation(), reward, done, {}
    

    # 최대 10% 만 매매 하게
    # def step(self, action):
    #     current_price = self.df.loc[self.current_step, 'Close']
    #     self.current_step += 1
    #     done = self.current_step == self.total_steps

    #     if action == BUY:
    #         # Calculate the amount of balance to be used (20% of available balance)
    #         balance_to_use = self.balance * 0.10
    #         shares_to_buy = balance_to_use // current_price
    #         cost = shares_to_buy * current_price
    #         commission = cost * self.commission_rate
    #         total_cost = cost + commission

    #         if total_cost <= self.balance:
    #             self.balance -= total_cost
    #             self.shares_held += shares_to_buy
    #         else:
    #             # Fallback to use the available balance
    #             shares_to_buy = self.balance // current_price
    #             cost = shares_to_buy * current_price
    #             commission = cost * self.commission_rate
    #             total_cost = cost + commission
    #             self.balance -= total_cost
    #             self.shares_held += shares_to_buy

    #     elif action == SELL:
    #         if self.shares_held > 0:
    #             # Calculate the maximum shares to sell (20% of shares held) and convert to integer
    #             shares_to_sell = int(self.shares_held * 0.10)
    #             sale_value = shares_to_sell * current_price
    #             commission = sale_value * self.commission_rate

    #             self.balance += sale_value - commission
    #             self.shares_held -= shares_to_sell

    #     # HOLD action does nothing
    #     reward = (self.balance + self.shares_held * current_price - self.initial_balance) / self.initial_balance
    #     return self._get_observation(), reward, done, {}

    def backtest(self, model):
        self.reset()
        done = False
        portfolio_values = [self.initial_balance]
        while not done:
            state = self._get_observation()
            state = np.reshape(state, [1, 6])
            action = model.act(state)
            _, reward, done, _ = self.step(action)
            portfolio_value = self.balance + self.shares_held * self.df.loc[self.current_step, 'Close']
            portfolio_values.append(portfolio_value)
        return portfolio_values
        
    def calculate_sharpe_ratio(self, portfolio_values, risk_free_rate=0.01):
        daily_returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] 
                        for i in range(1, len(portfolio_values))]
        excess_returns = [r - risk_free_rate/252 for r in daily_returns]  # Assuming 252 trading days
        if len(excess_returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
            return sharpe_ratio
        else:
            return 0  # 또는 다른 적절한 기본값