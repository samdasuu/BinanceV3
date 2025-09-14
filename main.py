from sma_strategy import SMAStrategy
from backtesting import Backtest
from get_chart_df import GetChart

CASH = 100000000
COMMISSION = .02

class RunBacktestStrategy():
    def __init__(self, df, strategy):
        self.strategy = strategy
        self.df = df

    def run(self):
        bt = Backtest(data=self.df, strategy=self.strategy, cash=CASH, commission=COMMISSION)
        stats = bt.run()
        print(stats)
        bt.plot()


df = GetChart().btc_1min()
# 전략에 데이터를 넣는 형태
RunBacktestStrategy(df, SMAStrategy).run()


