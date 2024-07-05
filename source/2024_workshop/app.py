import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import date
from pykrx import stock
from stock_trading_env import StockTradingEnv
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import random
import numpy as np

def calculate_max_drawdown(portfolio_values):
    peak = portfolio_values[0]
    max_drawdown = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown * 100  # 퍼센트로 반환

def calculate_buy_and_hold(stock_data, initial_balance):
    initial_price = stock_data['Close'].iloc[0]
    shares = initial_balance // initial_price
    remaining_balance = initial_balance % initial_price
    portfolio_values = stock_data['Close'] * shares + remaining_balance
    return portfolio_values.tolist()

def calculate_random_strategy(env):
    env.reset()
    portfolio_values = [env.initial_balance]
    done = False
    while not done:
        action = [random.choice([0, 1, 2]), random.random()]
        _, _, done, _ = env.step(action)
        portfolio_value = env.balance + env.shares_held * env.df.loc[env.current_step, 'Close']
        portfolio_values.append(portfolio_value)
    
    # 길이를 stock_data와 일치시킴
    return portfolio_values[:len(env.df)]

# Streamlit 애플리케이션 설정
st.set_page_config(layout="wide")

# Helper function to get stock data
def get_stock_data(ticker, start_date, end_date):
    """st
    주어진 티커와 날짜 범위에 대한 주식 데이터를 가져옵니다.

    Args:
        ticker (str): 주식 티커
        start_date (str): 시작 날짜 (YYYYMMDD)
        end_date (str): 종료 날짜 (YYYYMMDD)

    Returns:
        DataFrame: 주식 데이터
    """
    df = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
    df.reset_index(inplace=True)
    df.rename(columns={'날짜': 'Date', '종가': 'Close', '시가': 'Open', '고가': 'High', '저가': 'Low', '거래량': 'Volume'}, inplace=True)
    return df

# Helper function to get stock name
def get_stock_name(ticker):
    """
    주어진 티커에 해당하는 종목명을 가져옵니다.

    Args:
        ticker (str): 주식 티커

    Returns:
        str: 종목명
    """
    return stock.get_market_ticker_name(ticker)

# 사이드바 설정
st.sidebar.header("설정 기능")

# 설정 부분을 접고 펼 수 있게 함
with st.sidebar.expander("설정", expanded=True):
    ticker = st.sidebar.text_input("Ticker", "005930")

    st.sidebar.write("기간 설정")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date_train = st.date_input("Train 기간 시작", date(2023, 1, 1))
    with col2:
        end_date_train = st.date_input("Train 기간 종료", date(2023, 12, 31))

    col3, col4 = st.sidebar.columns(2)
    with col3:
        start_date_test = st.date_input("Test 기간 시작", date(2024, 1, 1))
    with col4:
        end_date_test = st.date_input("Test 기간 종료", date(2024, 6, 30))
    
    col5, col6 = st.sidebar.columns(2)
    with col5:
        epochs = st.number_input("Epochs", value=10, min_value=1, step=1)
    with col6:
        learning_rate = st.number_input("Learning rate", value=0.001, min_value=0.0001, step=0.0001, format="%.4f")
    
    col7, col8 = st.sidebar.columns(2)
    with col7:
        gamma = st.number_input("Gamma", value=0.95, min_value=0.01, step=0.01)
    with col8:
        batch_size = st.number_input("Batch size", value=32, min_value=1, step=1)
    
    col9, col10 = st.sidebar.columns(2)
    with col9:
        initial_balance = st.number_input("Initial balance (₩)", value=1000000, min_value=0, step=10000)
    with col10:
        commission_rate = st.number_input("Commission rate (%)", value=0.015, min_value=0.0, max_value=100.0, step=0.001, format="%.3f")

# 전역 변수로 state_size와 action_size 설정
state_size = 6
action_size = 4  # 3 actions (hold, buy, sell) + 1 for amount

# 전역 변수로 agent 초기화
if 'agent' not in st.session_state:
    st.session_state.agent = None

# 탭 설정
# tabs = st.tabs(["학습", "테스트"])
tabs = st.tabs(["학습", "모델 평가"])


with tabs[0]:
    st.header("학습")
    if st.button("학습 시작"):
        training_status_container = st.container()
        with training_status_container:
            st.subheader("학습 상태")
            training_status = st.empty()
        
        stock_data = get_stock_data(ticker, start_date_train.strftime('%Y%m%d'), end_date_train.strftime('%Y%m%d'))
        stock_name = get_stock_name(ticker)
        env = StockTradingEnv(stock_data, initial_balance=initial_balance, commission_rate=commission_rate/100)
        st.session_state.agent = DQNAgent(state_size, action_size, learning_rate=learning_rate, gamma=gamma)
        
        final_balance = 0  # Initialize final_balance
        for e in range(epochs):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            total_reward = 0
            for time in range(env.df.shape[0] - 1):
                action = st.session_state.agent.act(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                next_state = np.reshape(next_state, [1, state_size])
                st.session_state.agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    final_balance = env.balance + env.shares_held * env.df.loc[env.current_step, 'Close']
                    training_status.write(f"episode: {e+1}/{epochs}, final balance: {final_balance}, epsilon: {st.session_state.agent.epsilon:.2f}")
                    break
                if len(st.session_state.agent.memory) > batch_size:
                    st.session_state.agent.replay(batch_size)
        
        training_status.write("학습 완료!")

        # 액션 시그널 생성
        actions = [st.session_state.agent.act(np.reshape(env._get_observation(), [1, state_size])) for _ in range(env.df.shape[0] - 1)]
        buy_signals = [i for i, action in enumerate(actions) if action[0] == 1]
        sell_signals = [i for i, action in enumerate(actions) if action[0] == 2]
        hold_signals = [i for i, action in enumerate(actions) if action[0] == 0]

        # 학습 결과 플롯
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()

        # 주가 그래프
        ax1.plot(stock_data['Date'], stock_data['Close'], label='Close Price', color='blue')

        # 매수, 매도, 홀드 신호
        ax1.scatter(stock_data.iloc[buy_signals]['Date'], stock_data.iloc[buy_signals]['Close'], marker='^', color='g', label='Buy Signal', alpha=1)
        ax1.scatter(stock_data.iloc[sell_signals]['Date'], stock_data.iloc[sell_signals]['Close'], marker='v', color='r', label='Sell Signal', alpha=1)
        ax1.scatter(stock_data.iloc[hold_signals]['Date'], stock_data.iloc[hold_signals]['Close'], marker='o', color='b', label='Hold Signal', alpha=1)

        # Total Balance 계산 및 그래프 추가
        total_balance = [initial_balance]
        shares_held = 0
        cash_balance = initial_balance

        for i in range(1, len(stock_data)):
            action = actions[i-1]
            price = stock_data['Close'].iloc[i]
            
            if action[0] == 1:  # Buy
                max_buyable = cash_balance // price
                shares_to_buy = int(max_buyable * action[1])
                cost = shares_to_buy * price
                if cost <= cash_balance:
                    cash_balance -= cost
                    shares_held += shares_to_buy
            elif action[0] == 2:  # Sell
                shares_to_sell = int(shares_held * action[1])
                cash_balance += shares_to_sell * price
                shares_held -= shares_to_sell
            
            current_total_balance = cash_balance + (shares_held * price)
            total_balance.append(current_total_balance)
            
            # 로깅 (디버깅용)
            # print(f"Day {i}: Action={action[0]}, Price={price}, Shares={shares_held}, Cash={cash_balance}, Total={current_total_balance}")

        ax2.plot(stock_data['Date'], total_balance, label='Total Balance', color='orange')

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Close Price", color='blue')
        ax2.set_ylabel("Total Balance", color='orange')

        # 범례 설정
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax1.set_title(f"{ticker} - {stock_name}")
        fig.tight_layout()
        st.pyplot(fig)

        # 최종 성과 출력
        final_return = (total_balance[-1] - initial_balance) / initial_balance * 100
        st.write(f"Initial Balance: ₩{initial_balance:,.0f}")
        st.write(f"Final Balance: ₩{total_balance[-1]:,.0f}")
        st.write(f"Total Return: {final_return:.2f}%")

        # 학습 완료 후 모델 저장
        model_path = os.path.join('models', f'{ticker}.pth')
        os.makedirs('models', exist_ok=True)
        torch.save(st.session_state.agent.model.state_dict(), model_path)
        st.success(f"Model saved to {model_path}")      

with tabs[1]:
    st.header("모델 평가")
    
    # 비교 대상 선택
    compare_with = st.multiselect("비교 대상", ["Buy and Hold", "Random Strategy"])
    
    if st.button("평가 시작"):
        model_path = os.path.join('models', f'{ticker}.pth')
        if os.path.exists(model_path):
            stock_data_eval = get_stock_data(ticker, start_date_test.strftime('%Y%m%d'), end_date_test.strftime('%Y%m%d'))
            stock_name = get_stock_name(ticker)
            env_eval = StockTradingEnv(stock_data_eval, initial_balance=initial_balance, commission_rate=commission_rate/100)
            
            # 모델 로드
            agent_eval = DQNAgent(state_size, action_size, learning_rate=learning_rate, gamma=gamma)
            agent_eval.model.load_state_dict(torch.load(model_path))
            agent_eval.epsilon = 0  # 평가 시 탐험을 하지 않도록 설정
            
            # 평가 시작
            state = env_eval.reset()
            portfolio_values = [initial_balance]
            actions_taken = []
            
            for i in range(env_eval.df.shape[0] - 1):
                state = np.reshape(state, [1, state_size])
                action = agent_eval.act(state)
                actions_taken.append(action)
                next_state, reward, done, _ = env_eval.step(action)
                state = next_state
                
                current_price = env_eval.df.loc[env_eval.current_step, 'Close']
                
                if action[0] == 1:  # Buy
                    shares_to_buy = int((env_eval.balance * action[1]) // current_price)
                    cost = shares_to_buy * current_price
                    if cost <= env_eval.balance:
                        env_eval.balance -= cost
                        env_eval.shares_held += shares_to_buy
                elif action[0] == 2:  # Sell
                    shares_to_sell = int(env_eval.shares_held * action[1])
                    env_eval.balance += shares_to_sell * current_price
                    env_eval.shares_held -= shares_to_sell
                
                portfolio_value = env_eval.balance + env_eval.shares_held * current_price
                portfolio_values.append(portfolio_value)

                # 디버깅 출력
                st.write(f"Step {i}: Action={action[0]}, Amount={action[1]:.2f}, Price={current_price:.2f}, Shares={env_eval.shares_held}, Cash={env_eval.balance:.2f}, Portfolio={portfolio_value:.2f}")
            
            # 결과 시각화
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax2 = ax1.twinx()
            
            ax1.plot(stock_data_eval['Date'], stock_data_eval['Close'], label='Close Price', color='blue')
            ax2.plot(stock_data_eval['Date'], portfolio_values, label='AI Model Portfolio', color='orange')
            
            actions_taken = np.array(actions_taken)
            buy_signals = np.where(actions_taken[:, 0] == 1)[0]
            sell_signals = np.where(actions_taken[:, 0] == 2)[0]
            
            ax1.scatter(stock_data_eval.iloc[buy_signals]['Date'], stock_data_eval.iloc[buy_signals]['Close'], marker='^', color='g', label='Buy Signal', alpha=1)
            ax1.scatter(stock_data_eval.iloc[sell_signals]['Date'], stock_data_eval.iloc[sell_signals]['Close'], marker='v', color='r', label='Sell Signal', alpha=1)

            ax1.set_xlabel("Date")
            ax1.set_ylabel("Stock Price", color='blue')
            ax2.set_ylabel("Portfolio Value", color='orange')

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            ax1.set_title(f"{ticker} - {stock_name} ({start_date_test} to {end_date_test})")
            fig.tight_layout()
            st.pyplot(fig)  # 업데이트된 그래프 다시 표시

            # 성능 지표 계산 및 출력
            ai_total_return = (portfolio_values[-1] - initial_balance) / initial_balance * 100
            ai_max_drawdown = calculate_max_drawdown(portfolio_values)
            st.write(f"AI Model Total Return: {ai_total_return:.2f}%")
            st.write(f"AI Model Max Drawdown: {ai_max_drawdown:.2f}%")
            st.write(f"Final Balance: ₩{portfolio_values[-1]:,.0f}")
            
            # 액션 분포 확인
            action_counts = {'Buy': 0, 'Sell': 0, 'Hold': 0}
            for action in actions_taken:
                if action[0] == 1:
                    action_counts['Buy'] += 1
                elif action[0] == 2:
                    action_counts['Sell'] += 1
                else:
                    action_counts['Hold'] += 1
            st.write("Action distribution:", action_counts)
            
            # 비교 전략 계산 및 시각화
            if "Buy and Hold" in compare_with:
                buy_hold_values = calculate_buy_and_hold(stock_data_eval, initial_balance)
                ax2.plot(stock_data_eval['Date'], buy_hold_values, label='Buy and Hold', color='purple')
                bh_total_return = (buy_hold_values[-1] - initial_balance) / initial_balance * 100
                bh_max_drawdown = calculate_max_drawdown(buy_hold_values)
                st.write(f"Buy and Hold Total Return: {bh_total_return:.2f}%")
                st.write(f"Buy and Hold Max Drawdown: {bh_max_drawdown:.2f}%")

            if "Random Strategy" in compare_with:
                random_values = calculate_random_strategy(env_eval)
                ax2.plot(stock_data_eval['Date'], random_values, label='Random Strategy', color='red')
                random_total_return = (random_values[-1] - initial_balance) / initial_balance * 100
                random_max_drawdown = calculate_max_drawdown(random_values)
                st.write(f"Random Strategy Total Return: {random_total_return:.2f}%")
                st.write(f"Random Strategy Max Drawdown: {random_max_drawdown:.2f}%")

            # 범례 업데이트
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            st.error(f"{ticker} 모델이 존재하지 않습니다. 먼저 모델을 학습시켜 주세요.")
