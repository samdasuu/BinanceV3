from sma_strategy import SMAStrategy
from backtesting import Backtest
from utils.get_data import GetChart
import pandas as pd

CASH = 100000000
COMMISSION = .02

class RunBacktestStrategy():
    def __init__(self, df, strategy):
        self.strategy = strategy
        self.df = df.rename(
            columns={'open': 'Open', 
                     'high': 'High',
                     'low': 'Low',
                     'close': 'Close',
                     'volume': 'Volume'}
        )

    def run(self):
        bt = Backtest(data=self.df, strategy=self.strategy, cash=CASH, commission=COMMISSION)
        stats = bt.run()
        print(stats)
        bt.plot()


df = GetChart().btc_1min()
# 전략에 데이터를 넣는 형태
RunBacktestStrategy(df, SMAStrategy).run()


