import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle

# === LOAD 5-MIN CSV FILE ===
df_5m = pd.read_csv("CME_MINI_NQ1!, 5 (3).csv")  # Update path if needed
df_5m['timestamp'] = pd.to_datetime(df_5m['time'])
df_5m = df_5m.sort_values('timestamp').reset_index(drop=True)
df_5m['bar_index'] = np.arange(len(df_5m))

# === ZIGZAG DETECTION ===
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

# === STOP HUNT DETECTION ===
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

# === FVG DETECTION ===
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

def mark_mitigated_fvgs(fvg_df, df):
    mitigated = []
    for _, row in fvg_df.iterrows():
        post = df[df['timestamp'] > row['end_time']]
        if row['type'] == 'UP':
            mitigated.append((post['close'] < row['level2']).any())
        else:
            mitigated.append((post['close'] > row['level2']).any())
    fvg_df['mitigated'] = mitigated
    return fvg_df

# === RUN DETECTIONS ===
pivots = zigzag(df_5m, threshold=50.0)
stop_hunts = detect_multibar_stop_hunt(df_5m, pivots)
fvgs = detect_fvgs_3candle(df_5m, min_gap=5.0)
fvgs = mark_mitigated_fvgs(fvgs, df_5m)

# === PLOT ===
def draw_candles(ax, df, width=0.002):
    for _, row in df.iterrows():
        t = mdates.date2num(row['timestamp'])
        color = 'green' if row['close'] >= row['open'] else 'red'
        ax.add_line(mlines.Line2D((t, t), (row['low'], row['high']), color=color))
        rect = Rectangle((t - width / 2, min(row['open'], row['close'])), width, abs(row['open'] - row['close']),
                         facecolor=color, edgecolor=color)
        ax.add_patch(rect)

fig, ax = plt.subplots(figsize=(20, 10))
draw_candles(ax, df_5m)

# Plot pivots
for _, row in pivots.iterrows():
    x = mdates.date2num(row['timestamp'])
    marker = '^' if row['type'] == 'HIGH' else 'v'
    color = 'green' if row['type'] == 'HIGH' else 'red'
    ax.plot(x, row['price'], marker=marker, color=color, markersize=10)

# Plot stop hunts
for _, row in stop_hunts.iterrows():
    x = mdates.date2num(df_5m.loc[df_5m['bar_index'] == row['bar_index'], 'timestamp'].values[0])
    marker = 'x' if row['type'] == 'SHORT' else '+'
    color = 'purple' if row['type'] == 'SHORT' else 'blue'
    ax.plot(x, row['price'], marker=marker, color=color, markersize=10)

# Plot FVGs
for _, row in fvgs.iterrows():
    x_start = mdates.date2num(pd.to_datetime(row['start_time']))
    x_end = mdates.date2num(pd.to_datetime(row['end_time']))
    width = (x_end - x_start) * 5
    y1, y2 = sorted([row['level1'], row['level2']])
    color = 'blue' if row['mitigated'] else ('green' if row['type'] == 'UP' else 'red')
    rect = Rectangle((x_start, y1), width, y2 - y1, color=color, alpha=0.3)
    ax.add_patch(rect)

ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)        # Optional: angle for clarity

ax.set_title("All-in-One: 5-Min Candles + ZigZag + Stop Hunts + FVGs")
plt.grid()
plt.tight_layout()
plt.show()
