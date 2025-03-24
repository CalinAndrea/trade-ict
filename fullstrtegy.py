import pandas as pd
import numpy as np
from datetime import time

# === Load 5-minute data ===
df_5m = pd.read_csv("CME_MINI_NQ1!, 5 (3).csv")
df_5m['timestamp'] = pd.to_datetime(df_5m['time'])
df_5m = df_5m.sort_values('timestamp').reset_index(drop=True)
df_5m['bar_index'] = np.arange(len(df_5m))

# === Helper Functions ===
def localize_if_needed(series, tz='America/New_York'):
    if series.dt.tz is None:
        return series.dt.tz_localize(tz)
    else:
        return series.dt.tz_convert(tz)

def detect_fvgs_3candle(df, min_gap=5.0):
    out = []
    for i in range(len(df) - 2):
        A, C = df.iloc[i], df.iloc[i + 2]
        if A.high < C.low and (C.low - A.high) >= min_gap:
            out.append({'start_time': A.timestamp, 'end_time': C.timestamp,
                        'level1': A.high, 'level2': C.low, 'type': 'UP'})
        elif A.low > C.high and (A.low - C.high) >= min_gap:
            out.append({'start_time': A.timestamp, 'end_time': C.timestamp,
                        'level1': A.low, 'level2': C.high, 'type': 'DOWN'})
    return pd.DataFrame(out)

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
        pivots.append({'bar_index': pivot_idx, 'type': 'HIGH' if direction == 'UP' else 'LOW',
                       'price': pivot_price, 'timestamp': pivot_time})
    return pd.DataFrame(pivots)

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

# === Weekly Simulation ===
weekly_trades = []
available_dates = df_5m['timestamp'].dt.date.unique()

for day in available_dates:
    daily_df = df_5m[df_5m['timestamp'].dt.date == day].copy()
    daily_df.reset_index(drop=True, inplace=True)
    daily_df['bar_index'] = np.arange(len(daily_df))
    daily_df['timestamp'] = localize_if_needed(pd.to_datetime(daily_df['timestamp']))

    fvgs = detect_fvgs_3candle(daily_df)
    fvgs['midpoint'] = (fvgs['level1'] + fvgs['level2']) / 2
    fvgs['fvg_start'] = localize_if_needed(pd.to_datetime(fvgs['start_time']))
    fvgs['fvg_end'] = localize_if_needed(pd.to_datetime(fvgs['end_time']))

    pivots = zigzag(daily_df)
    stop_hunts = detect_multibar_stop_hunt(daily_df, pivots)
    stop_hunts['timestamp'] = stop_hunts['bar_index'].apply(lambda i: daily_df.loc[daily_df['bar_index'] == i, 'timestamp'].values[0])
    stop_hunts['timestamp'] = localize_if_needed(pd.to_datetime(stop_hunts['timestamp']))

    for _, hunt in stop_hunts.iterrows():
        direction = hunt['type']
        hunt_time = hunt['timestamp']
        future_bars = daily_df[daily_df['timestamp'] > hunt_time]
        if future_bars.empty:
            continue

        for _, candle in future_bars.iterrows():
            candle_time = candle['timestamp']
            if not time(8,0) <= candle_time.time() <= time(15,0):
                continue

            expected_type = 'UP' if direction == 'LONG' else 'DOWN'
            valid_fvgs = fvgs[(fvgs['type'] == expected_type) & (fvgs['fvg_end'] < candle_time)]

            for _, fvg in valid_fvgs.iterrows():
                midpoint = fvg['midpoint']
                if candle['low'] <= midpoint <= candle['high']:
                    pre_entry = daily_df[(daily_df['timestamp'] > fvg['fvg_end']) & (daily_df['timestamp'] < candle_time)]
                    if direction == 'LONG' and (pre_entry['close'] < midpoint).any():
                        continue
                    if direction == 'SHORT' and (pre_entry['close'] > midpoint).any():
                        continue

                    move = daily_df[(daily_df['timestamp'] >= hunt_time) & (daily_df['timestamp'] <= candle_time)]
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
                    future_check = daily_df[daily_df['timestamp'] >= candle_time]
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

                    weekly_trades.append({
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

weekly_df = pd.DataFrame(weekly_trades)

# === Display top 10 trade details ===
print("Top 20 Trades:")
print(weekly_df.head(20).to_string(index=False))

# === Statistics ===
total_trades = len(weekly_df)
win_trades = (weekly_df['PnL'] > 0).sum()
loss_trades = (weekly_df['PnL'] < 0).sum()
win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0
net_pnl = weekly_df['PnL'].sum()

print(f"📊 Stats:\nTotal Trades: {total_trades}\nWins: {win_trades}\nLosses: {loss_trades}\nWin Rate: {win_rate:.2f}%\nNet PnL: {net_pnl} points")
