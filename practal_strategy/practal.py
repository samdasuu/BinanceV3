import numpy as np
import pandas as pd

def preprocess_ohlcv(df: pd.DataFrame):
    """
    Input:
        df with columns ['Open','High','Low','Close','Volume']
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


def is_volume_spike(row, vol_mult=2.5):
    """
    개별 시점(row)이 거래량 스파이크 조건을 만족하는지 확인
    """
    if pd.isna(row['vol_ma300']) or row['vol_ma300'] == 0:
        return False
    return (row['volume'] >= vol_mult * row['vol_ma300'])


def has_up_momentum(df, i, lookback=15):
    """
    i번째 시점에서 상승 모멘텀 조건 확인
    - 최근 lookback 봉 누적 로그수익률 > 0
    - 또는 현재 종가 > EMA20
    """
    if i - lookback < 0:
        return False
    cum_logret = df['logret'].iloc[i - lookback + 1:i + 1].sum()
    cond1 = (cum_logret > 0)
    cond2 = (df['close'].iloc[i] > df['close_ema20'].iloc[i])
    return bool(cond1 or cond2)


def detect_events(df, vol_mult=2.0, lookback_mom=15):
    """
    거래량 스파이크 + 모멘텀 조건이 동시에 충족된 시점들의 인덱스 반환
    """
    events = []
    for i in range(len(df)):
        row = df.iloc[i]
        if is_volume_spike(row, vol_mult) and has_up_momentum(df, i, lookback_mom):
            events.append(df.index[i])
    return events


import matplotlib.pyplot as plt
def show_detect_events(df, events):
    plt.figure(figsize=(12,5))
    plt.plot(df['close'], label="close")
    plt.scatter(events, df.loc[events, 'close'], color='red', marker='^', label="event")
    plt.legend()
    plt.show()


from scipy.stats import pearsonr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np

# --- z-score 정규화 ---
def z_norm(series: np.ndarray):
    return (series - series.mean()) / (series.std() + 1e-8)

# --- 패턴 유사도 계산 ---
def calc_similarity(seq1: np.ndarray, seq2: np.ndarray, weight_corr=0.5, weight_dtw=0.5, scale=10.0):
    """
    seq1: 현재 패턴 (예: 최근 60봉)
    seq2: 과거 패턴 (예: 과거 60봉)
    weight_corr, weight_dtw: 가중치
    scale: DTW 점수를 0~1로 변환할 때 사용하는 스케일
    """
    # 정규화
    s1, s2 = z_norm(seq1), z_norm(seq2)

    # 피어슨 상관
    corr, _ = pearsonr(s1, s2)
    corr_score = (corr + 1) / 2   # [-1,1] → [0,1]

    # DTW 거리 → 유사도로 변환
    dist, _ = fastdtw(s1, s2, dist=euclidean)
    sim_dtw = np.exp(-dist / scale)  # 거리 작을수록 1에 가까움

    # 합성 점수
    final_score = weight_corr * corr_score + weight_dtw * sim_dtw

    return corr_score, sim_dtw, final_score


def check_entry_condition(df, now_idx, past_idx, 
                          window=60, future_window=20, 
                          vol_mult=2.0, rise_thr=0.05):
    """
    df: 전처리된 OHLCV 데이터프레임
    now_idx: 현재 이벤트 끝 인덱스
    past_idx: 매칭된 과거 구간 끝 인덱스
    window: 패턴 길이 (기본 60봉)
    future_window: 과거 상승 확인 기간 (기본 20봉)
    vol_mult: 거래량 스파이크 배수 (기본 2배)
    rise_thr: 과거 상승률 기준 (기본 5%)
    """

    # --- 조건1: 거래량 스파이크 ---
    row = df.iloc[now_idx]
    cond1 = (row['volume'] >= vol_mult * row['vol_ma300'])

    # --- 조건2: 패턴 유사도 ---
    now_seq = df['close'].iloc[now_idx-window+1 : now_idx+1].values
    past_seq = df['close'].iloc[past_idx-window+1 : past_idx+1].values
    corr, sim_dtw, final_score = calc_similarity(now_seq, past_seq)

    cond2 = (corr >= 0.8 and sim_dtw >= 0.8)

    # --- 조건3: 과거 상승 확인 ---
    if past_idx + future_window < len(df):
        past_price_end = df['close'].iloc[past_idx]
        past_price_future = df['close'].iloc[past_idx + future_window]
        future_ret = (past_price_future / past_price_end) - 1
        cond3 = (future_ret >= rise_thr)
    else:
        cond3 = False

    return cond1 and cond2 and cond3, (corr, sim_dtw, final_score)



def generate_signals(df, events, window=60, search_back=1000):
    """
    df: 전처리된 OHLCV 데이터
    events: detect_events 로 찾은 이벤트 시점 리스트
    window: 패턴 길이 (기본 60봉)
    search_back: 이벤트 직전 몇 봉까지 과거에서 탐색할지 (기본 1000봉)

    return: signals 리스트 (dict 형태)
    """
    signals = []

    for event_time in events:
        now_idx = df.index.get_loc(event_time)

        # 과거 후보 구간들
        past_candidates = range(now_idx - search_back, now_idx - window)
        best_score = -1
        best_past_idx = None
        best_corr, best_dtw = None, None

        for past_idx in past_candidates:
            if past_idx < window: 
                continue
            now_seq = df['close'].iloc[now_idx-window+1:now_idx+1].values
            past_seq = df['close'].iloc[past_idx-window+1:past_idx+1].values
            corr, sim_dtw, final_score = calc_similarity(now_seq, past_seq)

            if final_score > best_score:
                best_score = final_score
                best_past_idx = past_idx
                best_corr, best_dtw = corr, sim_dtw

        # 조건 체크
        if best_past_idx is not None:
            signal, scores = check_entry_condition(
                df, now_idx, best_past_idx, window=window
            )
            if signal:
                signals.append({
                    "time": event_time,
                    "now_idx": now_idx,
                    "past_idx": best_past_idx,
                    "corr": scores[0],
                    "dtw": scores[1],
                    "final_score": scores[2]
                })

    return signals

def simulate_exit(df, entry_idx, entry_price, window=60, 
                  max_horizon=60, take_profit=0.08, 
                  stop_corr=0.5):
    """
    df: 전처리된 OHLCV
    entry_idx: 진입 시점 index 위치 (int)
    entry_price: 진입 가격
    window: 패턴 길이
    max_horizon: 최대 보유 시간 (분)
    take_profit: 익절 기준 (예: 0.08 = +8%)
    stop_corr: 손절 기준 상관계수

    return: (exit_idx, exit_price, reason)
    """
    for i in range(entry_idx+1, min(entry_idx+max_horizon, len(df))):
        price_now = df['close'].iloc[i]
        ret = (price_now / entry_price) - 1

        # 1) 익절 조건
        if ret >= take_profit:
            return i, price_now, "TP"

        # 2) 손절 조건 (상관계수 재계산)
        now_seq = df['close'].iloc[i-window+1:i+1].values
        entry_seq = df['close'].iloc[entry_idx-window+1:entry_idx+1].values
        if len(now_seq) == window and len(entry_seq) == window:
            corr, _, _ = calc_similarity(now_seq, entry_seq)
            if corr < stop_corr:
                return i, price_now, "SL_corr"

    # 3) 시간 만료 청산
    exit_idx = min(entry_idx+max_horizon, len(df))-1
    exit_price = df['close'].iloc[exit_idx]
    return exit_idx, exit_price, "TIME"


def backtest(df, signals, window=60, max_horizon=60, 
             take_profit=0.08, stop_corr=0.5):
    results = []

    for sig in signals:
        entry_idx = sig["now_idx"]
        entry_price = df['close'].iloc[entry_idx]

        exit_idx, exit_price, reason = simulate_exit(
            df, entry_idx, entry_price, 
            window=window, max_horizon=max_horizon, 
            take_profit=take_profit, stop_corr=stop_corr
        )

        ret = (exit_price / entry_price) - 1
        results.append({
            "entry_time": df.index[entry_idx],
            "exit_time": df.index[exit_idx],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return": ret,
            "reason": reason,
            "corr": sig["corr"],
            "dtw": sig["dtw"],
            "final_score": sig["final_score"]
        })

    return pd.DataFrame(results)


