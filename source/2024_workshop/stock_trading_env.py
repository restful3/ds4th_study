import numpy as np

BUY = 1
SELL = 2
class StockTradingEnv:
    def __init__(self, df, initial_balance=1000000, commission_rate=0.00015):
        '''
        환경은 생성될 때 주식 데이터(DataFrame)를 받습니다.
        이 데이터는 self.df로 저장되어 환경 내부에서 사용됩니다.
        '''
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
        '''
        현재 상태를 numpy 배열로 반환합니다.
        상태는 다음 정보를 포함합니다:

        현재 잔액 (self.balance)
        보유 주식 수 (self.shares_held)
        현재 주가 (Close)
        시가 (Open)
        고가 (High)
        저가 (Low)
        
        '''
        return np.array([
            self.balance,
            self.shares_held,
            self.df.iloc[self.current_step]['Close'],
            self.df.iloc[self.current_step]['Open'],
            self.df.iloc[self.current_step]['High'],
            self.df.iloc[self.current_step]['Low']
        ])
        
    def step(self, action):
        '''
        에이전트가 action을 취하면, 환경은 이를 처리하고 새로운 상태를 반환합니다.
        주식 거래(매수/매도)를 시뮬레이션하고, 다음 거래일로 이동합니다.
        새로운 상태, 보상, 에피소드 종료 여부를 반환합니다.        
        '''        
        
        current_price = self.df.loc[self.current_step, 'Close']
        self.current_step += 1
        done = self.current_step == self.total_steps

        action_type = action[0]  # 0: Hold, 1: Buy, 2: Sell
        action_amount = action[1]  # Fraction of balance or shares to use

        if action_type == BUY:  # Buy
            amount_to_invest = self.balance * action_amount
            shares_to_buy = int(amount_to_invest // current_price)
            cost = shares_to_buy * current_price
            commission = cost * self.commission_rate
            total_cost = cost + commission
            if total_cost <= self.balance:
                self.balance -= total_cost
                self.shares_held += shares_to_buy

        elif action_type == SELL:  # Sell
            shares_to_sell = int(self.shares_held * action_amount)
            sale_value = shares_to_sell * current_price
            commission = sale_value * self.commission_rate
            self.balance += sale_value - commission
            self.shares_held -= shares_to_sell

        reward = (self.balance + self.shares_held * current_price - self.initial_balance) / self.initial_balance
        return self._get_observation(), reward, done, {}

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