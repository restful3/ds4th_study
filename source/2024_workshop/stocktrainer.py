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

    def get_stock_data(self):
        df = stock.get_market_ohlcv_by_date(self.start_date.strftime('%Y%m%d'), 
                                            self.end_date.strftime('%Y%m%d'), 
                                            self.ticker)
        df.reset_index(inplace=True)
        df.rename(columns={'날짜': 'Date', '종가': 'Close', '시가': 'Open', '고가': 'High', '저가': 'Low', '거래량': 'Volume'}, inplace=True)
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
            episode_reward = 0
            done = False

            while not done:
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                
                if self.agent_type == 'dqn':
                    self.agent.remember(state, action, reward, next_state, done)
                    self.agent.update()  # 새로운 update 메서드 사용
                elif self.agent_type in ['ppo', 'a2c']:
                    self.agent.train(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward

            print(f"Episode: {e+1}/{self.epochs}, Reward: {episode_reward:.2f}, Final Balance: {self.env.balance:.2f}")

    def get_company_name(self, ticker):
        # 종목 코드를 회사 이름으로 변환
        name = stock.get_market_ticker_name(ticker)
        return name

    def save_model(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        ticker_clean = ''.join(e for e in self.ticker if e.isalnum())
        
        model_path = os.path.join(models_dir, f'{ticker_clean}_{self.agent_type}.pth')
        self.agent.save(model_path)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    trainer = StockTrainer()
    trainer.train()
    trainer.save_model()