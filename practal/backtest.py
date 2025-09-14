import pandas as pd
from practal.condition import ConditionPractal
from practal.event import EventPractal
from practal.preprocess import PreprocessPractal

df = PreprocessPractal().run()
events = EventPractal(df).run()
condition = ConditionPractal(df, events)
siganls = condition.find()

def backtest(df, signals):
    results = []
    for i, sig in enumerate(signals):
        print(f'-------------{i}/{len(signals)}-------------')
        entry_idx = sig["now_idx"]
        entry_price = df['close'].iloc[entry_idx]

        exit_idx, exit_price, reason = condition.exit(entry_idx=entry_idx, entry_price=entry_price)

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