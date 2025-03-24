import pandas as pd
import numpy as np
from datetime import time

# === Load 5-minute data ===
df = pd.read_csv("CME_MINI_NQ1!, 5 (4).csv")
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

# === ML Feature Extraction ===
def extract_ml_features(trades):
    trades = trades.copy()
    trades['hour'] = trades['timestamp'].dt.hour + trades['timestamp'].dt.minute / 60.0
    trades['direction_code'] = trades['direction'].map({'LONG': 1, 'SHORT': -1})
    trades['duration'] = (trades['timestamp'] - trades['fvg_start']).dt.total_seconds() / 60.0
    trades['fvg_size'] = abs(trades['tp'] - trades['sl'])
    trades['target'] = (trades['PnL'] > 0).astype(int)
    return trades
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

# === Extract ML features and save ===
ml_data = extract_ml_features(weekly_trades)

# === Load trained model and predict ===
import joblib
model = joblib.load("trade_model.pkl")
ml_data['win_prob'] = model.predict_proba(ml_data[['hour', 'direction_code', 'duration', 'fvg_size']])[:, 1]

# === Filter trades with 70%+ confidence ===
high_confidence_trades = ml_data[ml_data['win_prob'] >= 0.70].copy()
high_confidence_trades = high_confidence_trades.sort_values(by='win_prob', ascending=False)

# Save
high_confidence_trades.to_csv("filtered_trades_conf_70.csv", index=False)

# === Generate Pine Script ===
def export_pine_script(trades, filename="pine_trades_output.txt"):
    with open(filename, "w") as f:
        f.write("//@version=5\n")
        f.write("indicator(\"AI High-Confidence Trades\", overlay=true)\n\n")
        for _, row in trades.iterrows():
            ts = pd.to_datetime(row['timestamp']).tz_convert("America/New_York")
            ts_str = ts.strftime('%Y-%m-%dT%H:%M:%S-04:00')
            entry = row['entry']
            sl = row['sl']
            tp = row['tp']
            direction = row['direction']
            marker = 'labelup' if direction == 'LONG' else 'labeldown'
            location = 'belowbar' if direction == 'LONG' else 'abovebar'
            color = 'green' if direction == 'LONG' else 'red'

            # Entry label
            f.write(f"plotshape(time == timestamp(\"{ts_str}\"), location=location.{location}, style=shape.{marker}, ")
            f.write(f"text=\"{direction} @{entry}\", color=color.{color}, textcolor=color.white)\n")

            # SL/TP dashed lines
            f.write(f"line.new(x1=timestamp(\"{ts_str}\"), y1={entry}, x2=timestamp(\"{ts_str}\"), y2={sl}, xloc=xloc.bar_time, color=color.red, style=line.style_dashed)\n")
            f.write(f"line.new(x1=timestamp(\"{ts_str}\"), y1={entry}, x2=timestamp(\"{ts_str}\"), y2={tp}, xloc=xloc.bar_time, color=color.green, style=line.style_dashed)\n")


export_pine_script(high_confidence_trades)

# === Plot high-confidence trades ===
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(20, 10))

print("Top high-confidence trades:")
print(high_confidence_trades[['timestamp', 'direction', 'entry', 'win_prob', 'target']].head(10))
print("\n Stats for high-confidence trades:")
total = len(high_confidence_trades)
wins = (high_confidence_trades['PnL'] > 0).sum()
losses = (high_confidence_trades['PnL'] < 0).sum()
net_pnl = high_confidence_trades['PnL'].sum()
print(f"Trades: {total} | Wins: {wins} | Losses: {losses} | Win %: {wins/total*100:.1f}% | Net: {net_pnl} pts")

# Plot candles
for _, row in df.iterrows():
    t = mdates.date2num(row['timestamp'])
    color = 'green' if row['close'] >= row['open'] else 'red'
    ax.add_line(mlines.Line2D((t, t), (row['low'], row['high']), color=color))
    rect = Rectangle((t - 0.0015, min(row['open'], row['close'])), 0.003,
                     abs(row['open'] - row['close']), facecolor=color, edgecolor=color)
    ax.add_patch(rect)

# Plot trades
for _, trade in high_confidence_trades.iterrows():
    x = mdates.date2num(pd.to_datetime(trade['timestamp']))
    color = 'blue' if trade['direction'] == 'LONG' else 'purple'
    marker = '^' if trade['direction'] == 'LONG' else 'v'
    ax.plot(x, trade['entry'], marker=marker, color=color, markersize=10)
    ax.text(x, trade['entry'], f"{trade['PnL']} pts \n {trade['win_prob']:.0%}", fontsize=8, color=color, ha='center', va='bottom')

    # SL and TP lines
    ax.plot([x, x], [trade['entry'], trade['sl']], color='red', linestyle='dashed', linewidth=1)
    ax.plot([x, x], [trade['entry'], trade['tp']], color='green', linestyle='dashed', linewidth=1)

# Format chart
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.set_title("High Confidence Trades (>= 70%)")
plt.grid(True)
plt.tight_layout()
plt.show()
print("✅ Saved high-confidence trades to filtered_trades_conf_70.csv")
