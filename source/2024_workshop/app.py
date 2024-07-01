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

# Streamlit 애플리케이션 설정
st.set_page_config(layout="wide")

# Helper function to get stock data
def get_stock_data(ticker, start_date, end_date):
    """
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
        epochs = st.number_input("Epochs", value=1000, min_value=1, step=1)
    with col6:
        learning_rate = st.number_input("Learning rate", value=0.001, min_value=0.0001, step=0.0001, format="%.4f")
    
    col7, col8 = st.sidebar.columns(2)
    with col7:
        gamma = st.number_input("Gamma", value=0.95, min_value=0.01, step=0.01)
    with col8:
        batch_size = st.number_input("Batch size", value=32, min_value=1, step=1)
    
    initial_balance = st.number_input("Initial balance (₩)", value=1000000, min_value=0, step=10000)

# 전역 변수로 state_size와 action_size 설정
state_size = 6
action_size = 4  # 3 actions (hold, buy, sell) + 1 for amount

# 전역 변수로 agent 초기화
if 'agent' not in st.session_state:
    st.session_state.agent = None

# 탭 설정
tabs = st.tabs(["학습", "테스트"])

with tabs[0]:
    st.header("학습")
    if st.button("학습 시작"):
        training_status_container = st.container()
        with training_status_container:
            st.subheader("학습 상태")
            training_status = st.empty()
        
        stock_data = get_stock_data(ticker, start_date_train.strftime('%Y%m%d'), end_date_train.strftime('%Y%m%d'))
        stock_name = get_stock_name(ticker)
        env = StockTradingEnv(stock_data, initial_balance=initial_balance)
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

        # 모델 저장
        model_path = os.path.join('models', f'{ticker}.pth')
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(st.session_state.agent.model.state_dict(), model_path)
        st.success(f"모델이 {model_path}에 저장되었습니다.")

        # 학습 결과 플롯
        fig, ax = plt.subplots()
        ax.plot(stock_data['Date'], stock_data['Close'], label='Close Price')
        actions = [st.session_state.agent.act(np.reshape(env._get_observation(), [1, state_size])) for _ in range(env.df.shape[0] - 1)]
        buy_signals = np.where(np.array(actions)[:, 0] == 1)[0]
        sell_signals = np.where(np.array(actions)[:, 0] == 2)[0]
        hold_signals = np.where(np.array(actions)[:, 0] == 0)[0]
        
        ax.scatter(stock_data.iloc[buy_signals]['Date'], stock_data.iloc[buy_signals]['Close'], marker='^', color='g', label='Buy Signal', alpha=1)
        ax.scatter(stock_data.iloc[sell_signals]['Date'], stock_data.iloc[sell_signals]['Close'], marker='v', color='r', label='Sell Signal', alpha=1)
        ax.scatter(stock_data.iloc[hold_signals]['Date'], stock_data.iloc[hold_signals]['Close'], marker='o', color='b', label='Hold Signal', alpha=1)
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.set_title(f"{ticker} - {stock_name}")
        ax.legend()
        st.pyplot(fig)

        # 학습 결과 아래에 최종 총 자산 표시
        st.write(f"Final Balance after training: ₩{final_balance}")

with tabs[1]:
    st.header("테스트")
    if st.button("테스트 시작"):
        model_path = os.path.join('models', f'{ticker}.pth')
        if os.path.exists(model_path):
            stock_data_test = get_stock_data(ticker, start_date_test.strftime('%Y%m%d'), end_date_test.strftime('%Y%m%d'))
            stock_name = get_stock_name(ticker)
            env_test = StockTradingEnv(stock_data_test, initial_balance=initial_balance)
            st.session_state.agent = DQNAgent(state_size, action_size, learning_rate=learning_rate, gamma=gamma)
            st.session_state.agent.model.load_state_dict(torch.load(model_path))
            
            state = env_test.reset()
            state = np.reshape(state, [1, state_size])
            actions_test = []
            
            for _ in range(env_test.df.shape[0] - 1):
                action = st.session_state.agent.act(state)
                next_state, reward, done, _ = env_test.step(action)
                state = np.reshape(next_state, [1, state_size])
                actions_test.append(action)
            
            fig_test, ax_test = plt.subplots()
            ax_test.plot(stock_data_test['Date'], stock_data_test['Close'], label='Close Price')
            
            buy_signals_test = np.where(np.array(actions_test)[:, 0] == 1)[0]
            sell_signals_test = np.where(np.array(actions_test)[:, 0] == 2)[0]
            hold_signals_test = np.where(np.array(actions_test)[:, 0] == 0)[0]
            
            ax_test.scatter(stock_data_test.iloc[buy_signals_test]['Date'], stock_data_test.iloc[buy_signals_test]['Close'], marker='^', color='g', label='Buy Signal', alpha=1)
            ax_test.scatter(stock_data_test.iloc[sell_signals_test]['Date'], stock_data_test.iloc[sell_signals_test]['Close'], marker='v', color='r', label='Sell Signal', alpha=1)
            ax_test.scatter(stock_data_test.iloc[hold_signals_test]['Date'], stock_data_test.iloc[hold_signals_test]['Close'], marker='o', color='b', label='Hold Signal', alpha=1)
            
            ax_test.set_xlabel("Date")
            ax_test.set_ylabel("Close Price")
            ax_test.set_title(f"{ticker} - {stock_name}")
            ax_test.legend()
            st.pyplot(fig_test)

            # 테스트 결과 아래에 최종 총 자산 표시
            final_balance_test = env_test.balance + env_test.shares_held * env_test.df.loc[env_test.current_step, 'Close']
            st.write(f"Final Balance after testing: ₩{final_balance_test}")
        else:
            st.error(f"{ticker} 모델이 존재하지 않습니다.")
