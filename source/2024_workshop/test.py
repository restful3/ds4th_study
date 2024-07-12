import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter

# 데이터 로드
df = pd.read_csv('./stock_data.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)

# 호드릭-프레스콧 필터 적용 함수
def apply_hp_filter(data, lambda_param=1600):
    cycle, trend = hpfilter(data, lamb=lambda_param)
    return cycle

# 종가에 대해 HP 필터 적용
df['price_cycle'] = apply_hp_filter(df['Close'])

# 거래량에 대해 HP 필터 적용
df['volume_cycle'] = apply_hp_filter(df['Volume'])

# 이동평균 계산 및 HP 필터 적용
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA60'] = df['Close'].rolling(window=60).mean()
df['ma20_cycle'] = apply_hp_filter(df['MA20'])
df['ma60_cycle'] = apply_hp_filter(df['MA60'])

# 그래프 그리기
fig, axs = plt.subplots(5, 1, figsize=(15, 25), sharex=True)

# 원본 주가
axs[0].plot(df.index, df['Close'], label='Close Price')
axs[0].set_title('Original Close Price')
axs[0].legend()

# 주가 변동성
axs[1].plot(df.index, df['price_cycle'], label='Price Cycle')
axs[1].set_title('Price Volatility (HP Filter)')
axs[1].legend()

# 거래량 변동성
axs[2].plot(df.index, df['volume_cycle'], label='Volume Cycle')
axs[2].set_title('Volume Volatility (HP Filter)')
axs[2].legend()

# MA20 변동성
axs[3].plot(df.index, df['ma20_cycle'], label='MA20 Cycle')
axs[3].set_title('MA20 Volatility (HP Filter)')
axs[3].legend()

# MA60 변동성
axs[4].plot(df.index, df['ma60_cycle'], label='MA60 Cycle')
axs[4].set_title('MA60 Volatility (HP Filter)')
axs[4].legend()

plt.tight_layout()
plt.show()