
import matplotlib.pyplot as plt
import pandas as pd

class EventPractal():
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def is_volume_spike(self, row, vol_mult=2.5):
        """
        개별 시점(row)이 거래량 스파이크 조건을 만족하는지 확인
        """
        if pd.isna(row['vol_ma300']) or row['vol_ma300'] == 0:
            return False
        return (row['volume'] >= vol_mult * row['vol_ma300'])


    def has_up_momentum(self, i, lookback=15):
        """
        i번째 시점에서 상승 모멘텀 조건 확인
        - 최근 lookback 봉 누적 로그수익률 > 0
        - 또는 현재 종가 > EMA20
        """
        if i - lookback < 0:
            return False
        cum_logret = self.df['logret'].iloc[i - lookback + 1:i + 1].sum()
        cond1 = (cum_logret > 0)
        cond2 = (self.df['close'].iloc[i] > self.df['close_ema20'].iloc[i])
        return bool(cond1 or cond2)


    def detect_events(self, vol_mult=2.0):
        """
        거래량 스파이크 + 모멘텀 조건이 동시에 충족된 시점들의 인덱스 반환
        """
        events = []
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            # if self.is_volume_spike(row, vol_mult) and self.has_up_momentum(i, lookback_mom):
            if self.is_volume_spike(row, vol_mult):
                events.append(self.df.index[i])
        return events


    # def show_detect_events(self, events):
    #     plt.figure(figsize=(12,5))
    #     plt.plot(self.df['close'], label="close")
    #     plt.scatter(events, self.df.loc[events, 'close'], color='red', marker='^', label="event")
    #     plt.legend()
    #     plt.show()
    
    def run(self):
        events = self.detect_events()
        print('-------------이벤트 포착 완료-------------')
        # self.show_detect_events(events)
        return events