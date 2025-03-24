import pandas as pd
import numpy as np
from datetime import time

# === Load 5-minute data ===
df = pd.read_csv("CME_MINI_NQ1!, 5 (6).csv")
df['timestamp'] = pd.to_datetime(df['time'], utc=True)
df = df.sort_values('timestamp').reset_index(drop=True)
df['bar_index'] = np.arange(len(df))
df['timestamp'] = df['timestamp'].dt.tz_convert("America/New_York")

# === Zigzag detection (refined) ===
def zigzag(df, threshold=25.0):
    if df.empty:
        return pd.DataFrame(columns=['bar_index','type','price','timestamp'])

    pivots = []
    direction = None
    pivot_idx = df.iloc[0]['bar_index']
    pivot_price = df.iloc[0]['close']
    pivot_time  = df.iloc[0]['timestamp']

    for i in range(1, len(df)):
        curr_close = df.iloc[i]['close']
        curr_time  = df.iloc[i]['timestamp']
        curr_bi    = df.iloc[i]['bar_index']

        if direction is None:
            if curr_close > pivot_price + threshold:
                pivots.append({'bar_index': pivot_idx, 'type': 'LOW', 'price': pivot_price, 'timestamp': pivot_time})
                direction = 'UP'
                pivot_idx, pivot_price, pivot_time = curr_bi, curr_close, curr_time
            elif curr_close < pivot_price - threshold:
                pivots.append({'bar_index': pivot_idx, 'type': 'HIGH', 'price': pivot_price, 'timestamp': pivot_time})
                direction = 'DOWN'
                pivot_idx, pivot_price, pivot_time = curr_bi, curr_close, curr_time
        elif direction == 'UP':
            if curr_close < pivot_price - threshold:
                pivots.append({'bar_index': pivot_idx, 'type': 'HIGH', 'price': pivot_price, 'timestamp': pivot_time})
                direction = 'DOWN'
                pivot_idx, pivot_price, pivot_time = curr_bi, curr_close, curr_time
            elif curr_close > pivot_price:
                pivot_idx, pivot_price, pivot_time = curr_bi, curr_close, curr_time
        else:
            if curr_close > pivot_price + threshold:
                pivots.append({'bar_index': pivot_idx, 'type': 'LOW', 'price': pivot_price, 'timestamp': pivot_time})
                direction = 'UP'
                pivot_idx, pivot_price, pivot_time = curr_bi, curr_close, curr_time
            elif curr_close < pivot_price:
                pivot_idx, pivot_price, pivot_time = curr_bi, curr_close, curr_time

    if direction:
        pivots.append({'bar_index': pivot_idx, 'type': 'HIGH' if direction == 'UP' else 'LOW', 'price': pivot_price, 'timestamp': pivot_time})

    return pd.DataFrame(pivots)

# === Stop Hunt Detection ===
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

# === FVG Detection ===
def detect_fvgs_3candle(df, min_gap=5.0):
    out = []
    for i in range(len(df) - 2):
        A, C = df.iloc[i], df.iloc[i + 2]
        if A.high < C.low and (C.low - A.high) >= min_gap:
            out.append({'start_time': A.timestamp, 'end_index': i + 2, 'level1': A.high, 'level2': C.low, 'type': 'UP'})
        elif A.low > C.high and (A.low - C.high) >= min_gap:
            out.append({'start_time': A.timestamp, 'end_index': i + 2, 'level1': A.low, 'level2': C.high, 'type': 'DOWN'})

    for fvg in out:
        mitigated_ts = None
        midpoint = (fvg['level1'] + fvg['level2']) / 2
        fvg['midpoint'] = midpoint
        fvg['mitigated'] = False
        for j in range(fvg['end_index'] + 1, len(df)):
            bar = df.iloc[j]
            ts = bar['timestamp']
            if fvg['type'] == 'UP' and bar['low'] < midpoint:
                mitigated_ts = ts
                fvg['mitigated'] = True
                break
            elif fvg['type'] == 'DOWN' and bar['high'] > midpoint:
                mitigated_ts = ts
                fvg['mitigated'] = True
                break
        fvg['end_time'] = mitigated_ts if mitigated_ts else df.iloc[fvg['end_index']]['timestamp'].replace(hour=16, minute=0)
    return pd.DataFrame(out)

# === Run detections ===
pivots = zigzag(df)
stop_hunts = detect_multibar_stop_hunt(df, pivots)
stop_hunts['timestamp'] = stop_hunts['bar_index'].apply(lambda i: df.loc[df['bar_index'] == i, 'timestamp'].values[0])
stop_hunts['timestamp'] = pd.to_datetime(stop_hunts['timestamp'], utc=True).dt.tz_convert("America/New_York")

fvgs = detect_fvgs_3candle(df)

# === Export Pine Script ===
def export_overlay_pine(pivots, stop_hunts, fvgs, filename="pine_overlay.txt"):
    with open(filename, "w") as f:
        f.write("//@version=6\n")
        f.write("indicator(\"ICT Pivots, Stop Hunts & FVGs\", overlay=true)\n\n")

        for _, row in pivots.iterrows():
            ts = pd.to_datetime(row['timestamp'])
            if not time(8, 30) <= ts.time() <= time(16, 0):
                continue
            label = 'Pivot High' if row['type'] == 'HIGH' else 'Pivot Low'
            shape = 'labeldown' if row['type'] == 'HIGH' else 'labelup'
            location = 'abovebar' if row['type'] == 'HIGH' else 'belowbar'
            color = 'lime' if row['type'] == 'HIGH' else 'orange'
            ts_str = ts.strftime('%Y-%m-%dT%H:%M:%S-04:00')
            f.write(f"plotshape(time == timestamp(\"{ts_str}\"), location=location.{location}, style=shape.{shape}, text=\"{label}\", color=color.{color}, textcolor=color.white)\n")

        for _, row in stop_hunts.iterrows():
            ts = pd.to_datetime(row['timestamp'])
            if not time(8, 30) <= ts.time() <= time(16, 0):
                continue
            label = 'Stop Hunt'
            shape = 'triangleup' if row['type'] == 'LONG' else 'triangledown'
            location = 'belowbar' if row['type'] == 'LONG' else 'abovebar'
            color = 'blue' if row['type'] == 'LONG' else 'purple'
            ts_str = ts.strftime('%Y-%m-%dT%H:%M:%S-04:00')
            f.write(f"plotshape(time == timestamp(\"{ts_str}\"), location=location.{location}, style=shape.{shape}, text=\"{label}\", color=color.{color}, textcolor=color.white)\n")

        for _, row in fvgs.iterrows():
            ts_start = pd.to_datetime(row['start_time']).strftime('%Y-%m-%dT%H:%M:%S-04:00')
            ts_end = pd.to_datetime(row['end_time']).strftime('%Y-%m-%dT%H:%M:%S-04:00')
            y1, y2 = sorted([row['level1'], row['level2']])
            if row['mitigated']:
                color = 'gray'
            else:
                color = 'green' if row['type'] == 'UP' else 'red'
            f.write(f"box.new(left=timestamp(\"{ts_start}\"), right=timestamp(\"{ts_end}\"), top={y2}, bottom={y1}, xloc=xloc.bar_time, border_color=color.{color}, bgcolor=color.new(color.{color}, 85))\n")

export_overlay_pine(pivots, stop_hunts, fvgs)
print("✅ ZigZag, Stop Hunts & FVGs exported to Pine script.")
