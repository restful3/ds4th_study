import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pykrx import stock
from stock_trading_env import StockTradingEnv
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.a2c_agent import A2CAgent
from datetime import date
import os
from config import config, get_agent_config

class StockEvaluator:
    def __init__(self, ticker):
        self.ticker = ticker
        self.start_date = config.get('eval_start_date', "20240101")
        self.end_date = config.get('eval_end_date', date.today().strftime("%Y%m%d"))
        self.initial_balance = config.get('initial_balance', 10000000)
        self.state_size = config.get('state_size', 6)
        self.action_size = config.get('action_size', 4)
        self.agent_type = config.get('agent_type', 'dqn')
        self.agent = None
        self.env = None
        self.test_data = None
        self.portfolio_values = None
        self.actions = None

    def get_stock_data(self):
        df = stock.get_market_ohlcv_by_date(self.start_date, self.end_date, self.ticker)
        df.reset_index(inplace=True)
        df.rename(columns={'날짜': 'Date', '종가': 'Close', '시가': 'Open', '고가': 'High', '저가': 'Low', '거래량': 'Volume'}, inplace=True)
        return df
    
    def load_model(self):
        agent_config = get_agent_config(self.agent_type)
        if self.agent_type == 'dqn':
            self.agent = DQNAgent(**agent_config)
        elif self.agent_type == 'ppo':
            self.agent = PPOAgent(**agent_config)
        elif self.agent_type == 'a2c':
            self.agent = A2CAgent(**agent_config)
        else:
            raise ValueError(f"Unsupported agent type: {self.agent_type}")

        # 소스 코드가 있는 폴더의 경로를 직접 지정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, 'models')
        ticker_clean = ''.join(e for e in self.ticker if e.isalnum())
        model_path = os.path.join(models_dir, f'{ticker_clean}_{self.agent_type}.pth')

        print(f"Current directory: {current_dir}")
        print(f"Models directory: {models_dir}")
        print(f"Attempting to load model from: {model_path}")
        
        if os.path.exists(model_path):
            self.agent.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print(f"Files in models directory: {os.listdir(models_dir)}")
            raise FileNotFoundError(f"Model file not found: {model_path}")    

    # def load_model(self):
    #     agent_config = get_agent_config(self.agent_type)
    #     if self.agent_type == 'dqn':
    #         self.agent = DQNAgent(**agent_config)
    #     elif self.agent_type == 'ppo':
    #         self.agent = PPOAgent(**agent_config)
    #     elif self.agent_type == 'a2c':
    #         self.agent = A2CAgent(**agent_config)
    #     else:
    #         raise ValueError(f"Unsupported agent type: {self.agent_type}")

    #     current_dir = os.path.dirname(os.path.abspath(__file__))
    #     models_dir = os.path.join(current_dir, 'models')
    #     ticker_clean = ''.join(e for e in self.ticker if e.isalnum())
    #     model_path = os.path.join(models_dir, f'{ticker_clean}_{self.agent_type}.pth')
        
    #     if os.path.exists(model_path):
    #         self.agent.load(model_path)
    #         print(f"Model loaded from {model_path}")
    #     else:
    #         raise FileNotFoundError(f"Model file not found: {model_path}")

    def evaluate(self):
        self.test_data = self.get_stock_data()
        self.test_data.reset_index(drop=True, inplace=True)

        self.env = StockTradingEnv(self.test_data, initial_balance=self.initial_balance)
        state = self.env.reset()
        self.portfolio_values = [self.initial_balance]
        self.actions = []

        for _ in range(len(self.test_data) - 1):
            state = np.reshape(state, [1, self.state_size])
            action = self.agent.act(state)
            next_state, _, done, _ = self.env.step(action)
            state = next_state
            portfolio_value = self.env.balance + self.env.shares_held * self.env.df.iloc[self.env.current_step]['Close']
            self.portfolio_values.append(portfolio_value)
            self.actions.append(action)  # action is now a single integer, no need for indexing
            if done:
                break

    def plot_evaluation(self):
        fig, ax1 = plt.subplots(figsize=(15, 7))
        ax2 = ax1.twinx()
        
        dates = pd.to_datetime(self.test_data['Date'])
        
        ax1.plot(dates, self.test_data['Close'], label='Stock Price', color='blue')
        ax2.plot(dates, self.portfolio_values, label='Portfolio Value', color='red')
        
        hold_dates = [date for date, action in zip(dates, self.actions) if action == 0]
        buy_dates = [date for date, action in zip(dates, self.actions) if action == 1]
        sell_dates = [date for date, action in zip(dates, self.actions) if action == 2]
        
        ax1.scatter(hold_dates, [self.test_data.loc[self.test_data['Date'] == str(date)[:10], 'Close'].values[0] for date in hold_dates], 
                    color='yellow', marker='o', s=50, label='Hold')
        ax1.scatter(buy_dates, [self.test_data.loc[self.test_data['Date'] == str(date)[:10], 'Close'].values[0] for date in buy_dates], 
                    color='green', marker='^', s=100, label='Buy')
        ax1.scatter(sell_dates, [self.test_data.loc[self.test_data['Date'] == str(date)[:10], 'Close'].values[0] for date in sell_dates], 
                    color='red', marker='v', s=100, label='Sell')  
              
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Price', color='blue')
        ax2.set_ylabel('Portfolio Value', color='red')
        
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        plt.title(f'Evaluation Results for {self.ticker}')
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
        fig.tight_layout()
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(current_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        result_path = os.path.join(results_dir, f'{self.ticker}_{self.agent_type}.png')
        
        plt.savefig(result_path)
        plt.close()
        
        print(f"Evaluation results saved to {result_path}")


    def run_evaluation(self):
        self.load_model()
        self.evaluate()
        self.plot_evaluation()
        print(f"Evaluation complete. Results saved in result_{self.ticker}_{self.agent_type}.png")

if __name__ == "__main__":
    ticker = config['ticker']
    evaluator = StockEvaluator(ticker)
    evaluator.run_evaluation()