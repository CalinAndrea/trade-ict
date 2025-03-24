import pandas as pd
import numpy as np
from datetime import time

# === Load 5-minute data ===
df = pd.read_csv("CME_MINI_NQ1!, 5 (3).csv")
df['timestamp'] = pd.to_datetime(df['time'], utc=True)
df = df.sort_values('timestamp').reset_index(drop=True)
df['bar_index'] = np.arange(len(df))
df['timestamp'] = df['timestamp'].dt.tz_convert("America/New_York")

# === Zigzag detection ===
def zigzag(df, threshold=50.0):
    pivots = []
    direction = None
    pivot_idx = df.iloc[0]['bar_index']
    pivot_price = df.iloc[0]['close']
    pivot_time = df.iloc[0]['timestamp']
    for i in range(1, len(df)):
        curr = df.iloc[i]
        if direction is None:
            if curr.close > pivot_price + threshold:
                pivots.append({'bar_index': pivot_idx, 'type': 'LOW', 'price': pivot_price, 'timestamp': pivot_time})
                direction = 'UP'
            elif curr.close < pivot_price - threshold:
                pivots.append({'bar_index': pivot_idx, 'type': 'HIGH', 'price': pivot_price, 'timestamp': pivot_time})
                direction = 'DOWN'
        elif direction == 'UP':
            if curr.close < pivot_price - threshold:
                pivots.append({'bar_index': pivot_idx, 'type': 'HIGH', 'price': pivot_price, 'timestamp': pivot_time})
                direction = 'DOWN'
            elif curr.close > pivot_price:
                pivot_idx, pivot_price, pivot_time = curr.bar_index, curr.close, curr.timestamp
        else:
            if curr.close > pivot_price + threshold:
                pivots.append({'bar_index': pivot_idx, 'type': 'LOW', 'price': pivot_price, 'timestamp': pivot_time})
                direction = 'UP'
            elif curr.close < pivot_price:
                pivot_idx, pivot_price, pivot_time = curr.bar_index, curr.close, curr.timestamp
    if direction:
        pivots.append({'bar_index': pivot_idx, 'type': 'HIGH' if direction == 'UP' else 'LOW', 'price': pivot_price, 'timestamp': pivot_time})
    return pd.DataFrame(pivots)

# === Stop hunt ===
def detect_multibar_stop_hunt(df, pivots, tolerance=5.0, lookahead=3):
    hunts = []
    for _, pivot in pivots.iterrows():
        pivot_index = pivot['bar_index']
        pivot_price = pivot['price']
        pivot_type = pivot['type']
        df_future = df[(df['bar_index'] > pivot_index) & (df['bar_index'] <= pivot_index + lookahead)]
        if pivot_type == 'HIGH':
            exceed = df_future[df_future['high'] > pivot_price + tolerance]
            if not exceed.empty:
                post = df_future[df_future['bar_index'] >= exceed.iloc[0]['bar_index']]
                reversal = post[post['close'] < pivot_price]
                if not reversal.empty:
                    hunts.append({'bar_index': reversal.iloc[0]['bar_index'], 'price': pivot_price, 'type': 'SHORT'})
        else:
            exceed = df_future[df_future['low'] < pivot_price - tolerance]
            if not exceed.empty:
                post = df_future[df_future['bar_index'] >= exceed.iloc[0]['bar_index']]
                reversal = post[post['close'] > pivot_price]
                if not reversal.empty:
                    hunts.append({'bar_index': reversal.iloc[0]['bar_index'], 'price': pivot_price, 'type': 'LONG'})
    return pd.DataFrame(hunts)

# === FVGs ===
def detect_fvgs_3candle(df, min_gap=5.0):
    out = []
    for i in range(len(df) - 2):
        A, C = df.iloc[i], df.iloc[i + 2]
        if A.high < C.low and (C.low - A.high) >= min_gap:
            out.append({'start_time': A.timestamp, 'end_time': C.timestamp, 'level1': A.high, 'level2': C.low, 'type': 'UP'})
        elif A.low > C.high and (A.low - C.high) >= min_gap:
            out.append({'start_time': A.timestamp, 'end_time': C.timestamp, 'level1': A.low, 'level2': C.high, 'type': 'DOWN'})
    return pd.DataFrame(out)

# === Run ===
pivots = zigzag(df)
stop_hunts = detect_multibar_stop_hunt(df, pivots)
stop_hunts['timestamp'] = stop_hunts['bar_index'].apply(lambda i: df.loc[df['bar_index'] == i, 'timestamp'].values[0])
if stop_hunts['timestamp'].dt.tz is None:
    stop_hunts['timestamp'] = stop_hunts['timestamp'].dt.tz_localize("America/New_York")
else:
    stop_hunts['timestamp'] = stop_hunts['timestamp'].dt.tz_convert("America/New_York")

fvgs = detect_fvgs_3candle(df)
fvgs['midpoint'] = (fvgs['level1'] + fvgs['level2']) / 2
fvgs['fvg_start'] = pd.to_datetime(fvgs['start_time']).dt.tz_convert("America/New_York")
fvgs['fvg_end'] = pd.to_datetime(fvgs['end_time']).dt.tz_convert("America/New_York")

trades = []
for _, hunt in stop_hunts.iterrows():
    direction = hunt['type']
    hunt_time = hunt['timestamp']
    future_bars = df[df['timestamp'] > hunt_time]
    if future_bars.empty:
        continue
    for _, candle in future_bars.iterrows():
        candle_time = candle['timestamp']
        if not time(8, 0) <= candle_time.time() <= time(15, 0):
            continue
        expected_type = 'UP' if direction == 'LONG' else 'DOWN'
        valid_fvgs = fvgs[(fvgs['type'] == expected_type) & (fvgs['fvg_end'] < candle_time)]
        for _, fvg in valid_fvgs.iterrows():
            midpoint = fvg['midpoint']
            if candle['low'] <= midpoint <= candle['high']:
                pre_entry = df[(df['timestamp'] > fvg['fvg_end']) & (df['timestamp'] < candle_time)]
                if direction == 'LONG' and (pre_entry['close'] < midpoint).any():
                    continue
                if direction == 'SHORT' and (pre_entry['close'] > midpoint).any():
                    continue
                move = df[(df['timestamp'] >= hunt_time) & (df['timestamp'] <= candle_time)]
                if move.empty:
                    continue
                high = move['high'].max()
                low = move['low'].min()
                fib_50 = (high + low) / 2
                if direction == 'LONG' and midpoint > fib_50:
                    continue
                if direction == 'SHORT' and midpoint < fib_50:
                    continue
                entry = midpoint
                sl = entry - 20 if direction == 'LONG' else entry + 20
                tp = entry + 40 if direction == 'LONG' else entry - 40
                outcome = 0
                future_check = df[df['timestamp'] >= candle_time]
                for _, b in future_check.iterrows():
                    if direction == 'LONG':
                        if b['low'] <= sl:
                            outcome = -20
                            break
                        if b['high'] >= tp:
                            outcome = 40
                            break
                    else:
                        if b['high'] >= sl:
                            outcome = -20
                            break
                        if b['low'] <= tp:
                            outcome = 40
                            break
                trades.append({
                    "date": candle_time.date(),
                    "timestamp": candle_time,
                    "direction": direction,
                    "entry": round(entry, 2),
                    "sl": round(sl, 2),
                    "tp": round(tp, 2),
                    "PnL": outcome,
                    "fvg_start": fvg['fvg_start'],
                    "fvg_end": fvg['fvg_end']
                })
                break
        else:
            continue
        break

weekly_trades = pd.DataFrame(trades)
print("Top trades:")
print(weekly_trades.head(10))
print("\nStats:")
total = len(weekly_trades)
wins = (weekly_trades['PnL'] > 0).sum()
losses = (weekly_trades['PnL'] < 0).sum()
net_pnl = weekly_trades['PnL'].sum()
print(f"Trades: {total} | Wins: {wins} | Losses: {losses} | Win %: {wins/total*100:.1f}% | Net: {net_pnl} pts")
