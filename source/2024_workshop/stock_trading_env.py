import numpy as np

class StockTradingEnv:
    """
    StockTradingEnv 클래스는 주식 거래 환경을 시뮬레이션합니다.

    Attributes:
        stock_data (DataFrame): 주식 시장 데이터
        current_step (int): 현재 스텝
        total_steps (int): 총 스텝 수
        cash (float): 현금 보유량
        stock_owned (int): 보유 주식 수
        stock_price (float): 현재 주식 가격
        total_value (float): 포트폴리오 총 가치
    """

    def __init__(self, stock_data):
        """
        StockTradingEnv 초기화 메서드

        Args:
            stock_data (DataFrame): 주식 시장 데이터
        """
        self.stock_data = stock_data
        self.current_step = 0
        self.total_steps = len(stock_data)
        self.cash = 10000
        self.stock_owned = 0
        self.stock_price = self.stock_data.iloc[self.current_step]['Close']
        self.total_value = self.cash

    def reset(self):
        """
        환경을 초기 상태로 리셋합니다.

        Returns:
            numpy array: 초기 상태
        """
        self.current_step = 0
        self.cash = 10000
        self.stock_owned = 0
        self.stock_price = self.stock_data.iloc[self.current_step]['Close']
        self.total_value = self.cash
        return self._get_observation()

    def _get_observation(self):
        """
        현재 상태를 반환합니다.

        Returns:
            numpy array: 현재 상태 (주식 가격, 보유 주식 수, 현금 보유량)
        """
        obs = np.array([self.stock_price, self.stock_owned, self.cash])
        return obs

    def step(self, action):
        """
        환경에서 한 스텝을 진행합니다.

        Args:
            action (int): 에이전트의 행동 (0: 매수, 1: 매도, 2: 보유)

        Returns:
            tuple: (다음 상태, 보상, 종료 여부, 추가 정보)
        """
        self.stock_price = self.stock_data.iloc[self.current_step]['Close']
        reward = 0
        done = False

        if action == 0:  # 매수
            if self.cash > self.stock_price:
                self.stock_owned += 1
                self.cash -= self.stock_price
        elif action == 1:  # 매도
            if self.stock_owned > 0:
                self.stock_owned -= 1
                self.cash += self.stock_price
        elif action == 2:  # 보유
            pass

        self.total_value = self.cash + self.stock_owned * self.stock_price
        reward = self.total_value - self.cash

        self.current_step += 1
        if self.current_step >= self.total_steps:
            done = True

        return self._get_observation(), reward, done, {}
