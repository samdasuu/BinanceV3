import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
secret = os.getenv("BINANCE_SECRET_KEY")

import ccxt
import pandas as pd

class GetChart():
    def __init__(self):
    # Binance 선물 거래소 초기화
        self.exchange = ccxt.binance({
            'options': {
                'defaultType': 'future',  # 선물 거래소로 설정
            }
        })

    def ohlcv2df(self, ohlcv):
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # timestamp를 datetime 형식으로 변환하고 인덱스로 설정
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    #run
    def btc_1min(self, day = 7):
        # 차트 데이터 가져오기
        ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=1440 * day)
        return self.ohlcv2df(ohlcv)
    
    
    def btc_1day(self, month = 12):
        # 차트 데이터 가져오기
        ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe='1d', limit=30 * month)
        return self.ohlcv2df(ohlcv)

