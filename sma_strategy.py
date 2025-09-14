from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA


class SMAStrategy(Strategy):
    # SMA 기간을 매개변수로 설정
    n_days = 20

    def init(self):
        # init()에서 사용할 기술적 지표를 계산합니다.
        # self.data.Close는 데이터프레임의 'Close' 컬럼을 의미합니다.
        self.sma = self.I(SMA, self.data.Close, self.n_days)

    def next(self):
        # next()는 각 캔들마다 호출되며, 매수/매도 로직을 작성합니다.
        # crossover() 함수는 SMA선과 종가선의 교차를 감지합니다.
        # +1은 상승 돌파(SMA가 종가를 위로 뚫음)를 의미
        if crossover(self.data.Close, self.sma):
            self.buy(size = 1) # 매수

        # -1은 하락 돌파(종가가 SMA를 아래로 뚫음)를 의미
        elif crossover(self.sma, self.data.Close):
            self.sell(size = 1) # 매도





            