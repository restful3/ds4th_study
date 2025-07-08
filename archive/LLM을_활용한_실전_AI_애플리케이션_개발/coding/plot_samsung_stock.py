# filename: plot_samsung_stock.py
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 삼성전자 티커 (한국 증시)
ticker = "005930.KS"

# 오늘 날짜와 3개월 전 날짜 계산
end_date = datetime.today()
start_date = end_date - timedelta(days=90)

# 데이터 다운로드
data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

# 종가 데이터
close_prices = data['Close']

# y축 최소값
y_min = close_prices.min()

# 그래프 그리기
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=close_prices.index,
    y=close_prices,
    mode='lines',
    line=dict(color='green'),
    fill='tozeroy',
    fillcolor='rgba(0,128,0,0.3)',  # 투명한 녹색
    name='Close Price'
))

# y축 범위 설정 (최소값부터 최대값까지)
fig.update_yaxes(range=[y_min, close_prices.max()])

# 레이아웃 설정 (적절한 비율)
fig.update_layout(
    title='삼성전자 지난 3개월 주식 가격',
    xaxis_title='날짜',
    yaxis_title='주가 (KRW)',
    width=900,
    height=500,
    template='plotly_white'
)

# 이미지로 저장 (png)
fig.write_image("samsung_stock_price.png")

print("samsung_stock_price.png 파일이 생성되었습니다.")