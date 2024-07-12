import numpy as np
from pykrx import stock
from stock_trading_env import StockTradingEnv
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.a2c_agent import A2CAgent
import os
from config import config, get_agent_config, update_config

class StockTrainer:
    def __init__(self):
        self.ticker = config['ticker']
        self.start_date = config['start_date']
        self.end_date = config['end_date']
        self.initial_balance = config['initial_balance']
        self.commission_rate = config['commission_rate']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.agent_type = config['agent_type']
        self.state_size = config['state_size']
        self.action_size = config['action_size']  # 이제 3이 되어야 함 (Hold, Buy, Sell)
        self.agent = None
        self.env = None
        self.best_cumulative_reward = -np.inf
        self.best_model = None        

    def get_stock_data(self):
        df = stock.get_market_ohlcv_by_date(self.start_date, self.end_date, self.ticker)
        df.reset_index(inplace=True)
        df.rename(columns={'날짜': 'Date', '종가': 'Close', '시가': 'Open', '고가': 'High', '저가': 'Low', '거래량': 'Volume', '등락률':'Rate'}, inplace=True)
        # MA 20 (단순 이동평균)
        df['MA20_Simple'] = df['Close'].rolling(window=20).mean()
        # MA 20 (지수 가중 이동평균, EMA)
        df['MA20_EMA'] = df['Close'].ewm(span=20, min_periods=20, adjust=False).mean()
        # MA 60 (단순 이동평균)
        df['MA60_Simple'] = df['Close'].rolling(window=60).mean()
        # MA 60 (지수 가중 이동평균, EMA)
        df['MA60_EMA'] = df['Close'].ewm(span=20, min_periods=60, adjust=False).mean()
        # MA 20이 결측인 행 제거
        df.dropna(subset=['MA60_Simple'], inplace=True)
        return df

    def create_agent(self):
        agent_config = get_agent_config(self.agent_type)
        if self.agent_type == 'dqn':
            return DQNAgent(**agent_config)
        elif self.agent_type == 'ppo':
            return PPOAgent(**agent_config)
        elif self.agent_type == 'a2c':
            return A2CAgent(**agent_config)
        else:
            raise ValueError(f"Unsupported agent type: {self.agent_type}")

    def train(self):
        train_data = self.get_stock_data()
        train_data.reset_index(drop=True, inplace=True)

        self.env = StockTradingEnv(train_data, initial_balance=self.initial_balance, commission_rate=self.commission_rate)
        self.agent = self.create_agent()

        for e in range(self.epochs):
            state = self.env.reset()
            cumulative_reward = 0
            done = False

            while not done:
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                
                if self.agent_type == 'dqn':
                    self.agent.remember(state, action, reward, next_state, done)
                    self.agent.update()
                elif self.agent_type in ['ppo', 'a2c']:
                    self.agent.train(state, action, reward, next_state, done)

                state = next_state
                cumulative_reward += reward

            print(f"Episode: {e+1}/{self.epochs}, Cumulative Reward: {cumulative_reward:.2f}")

            if cumulative_reward > self.best_cumulative_reward:
                self.best_cumulative_reward = cumulative_reward
                self.best_model = self.agent
                print(f"New best model found with Cumulative Reward: {cumulative_reward:.2f}")

    def get_company_name(self, ticker):
        # 종목 코드를 회사 이름으로 변환
        name = stock.get_market_ticker_name(ticker)
        return name

    def save_model(self):
        if self.best_model is None:
            print("No model to save. Training might not have been performed.")
            return

        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        ticker_clean = ''.join(e for e in self.ticker if e.isalnum())
        
        model_path = os.path.join(models_dir, f'{ticker_clean}_{self.agent_type}_best.pth')
        self.best_model.save(model_path)
        print(f"Best model saved to {model_path} with Cumulative Reward: {self.best_cumulative_reward:.2f}")

if __name__ == "__main__":
    trainer = StockTrainer()
    trainer.train()
    trainer.save_model()