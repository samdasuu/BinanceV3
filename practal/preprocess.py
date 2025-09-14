import numpy as np
from utils.get_data import GetChart

DAY = 300

class PreprocessPractal():
    def preprocess_ohlcv(self, df):
        """
        Input:
            df with columns ['open','high','low','close','volume']
            index = datetime (1분봉)
        Output:
            df with 추가 컬럼:
            - logret: 종가 로그수익률
            - vol_roc: 거래량 변화율
            - close_ema20: EMA(20)
            - vol_ma300: 거래량 300봉 평균
            - vol_std300: 거래량 300봉 표준편차
            - vol_z: 거래량 z-score
        """
        df = df.copy()

        # 종가 로그수익률
        df['logret'] = np.log(df['close']).diff()

        # 거래량 변화율
        df['vol_roc'] = df['volume'].pct_change().replace([np.inf, -np.inf], np.nan)

        # 모멘텀 확인용 EMA
        df['close_ema20'] = df['close'].ewm(span=20, adjust=False).mean()

        # 거래량 평균 & 표준편차
        df['vol_ma300'] = df['volume'].rolling(300, min_periods=300).mean()
        df['vol_std300'] = df['volume'].rolling(300, min_periods=300).std()

        # 거래량 z-score
        df['vol_z'] = (df['volume'] - df['vol_ma300']) / df['vol_std300']

        return df
    

    
    def run(self):
        chart = GetChart().btc_1min(day = DAY)
        print('-------------차트 가져오기 완료-------------')
        return self.preprocess_ohlcv(chart)