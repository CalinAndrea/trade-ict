import pandas as pd
import numpy as np
from datetime import time
from collections import deque

# === Load 5-minute data ===
df = pd.read_csv("CME_MINI_NQ1!, 5 (7).csv")
df['timestamp'] = pd.to_datetime(df['time'], utc=True)
df = df.sort_values('timestamp').reset_index(drop=True)
df['bar_index'] = np.arange(len(df))
df['timestamp'] = df['timestamp'].dt.tz_convert("America/New_York")

# === Zigzag detection ===
def zigzag(df, threshold=20.0):
    if df.empty:
        return pd.DataFrame(columns=['bar_index','type','price','timestamp'])

    pivots = []
    direction = None
    last_pivot = df.iloc[0]
    swing_high = last_pivot['high']
    swing_low = last_pivot['low']
    swing_idx = last_pivot['bar_index']
    swing_time = last_pivot['timestamp']

    for i in range(1, len(df)):
        curr = df.iloc[i]

        if direction is None:
            if curr['close'] > last_pivot['close'] + threshold:
                direction = 'UP'
                swing_low = last_pivot['low']
                swing_idx = last_pivot['bar_index']
                swing_time = last_pivot['timestamp']
            elif curr['close'] < last_pivot['close'] - threshold:
                direction = 'DOWN'
                swing_high = last_pivot['high']
                swing_idx = last_pivot['bar_index']
                swing_time = last_pivot['timestamp']
        elif direction == 'UP':
            if curr['high'] > swing_high:
                swing_high = curr['high']
                swing_idx = curr['bar_index']
                swing_time = curr['timestamp']
            elif curr['close'] < swing_high - threshold:
                pivots.append({'bar_index': swing_idx, 'type': 'HIGH', 'price': swing_high, 'timestamp': swing_time})
                direction = 'DOWN'
                swing_low = curr['low']
                swing_idx = curr['bar_index']
                swing_time = curr['timestamp']
        elif direction == 'DOWN':
            if curr['low'] < swing_low:
                swing_low = curr['low']
                swing_idx = curr['bar_index']
                swing_time = curr['timestamp']
            elif curr['close'] > swing_low + threshold:
                pivots.append({'bar_index': swing_idx, 'type': 'LOW', 'price': swing_low, 'timestamp': swing_time})
                direction = 'UP'
                swing_high = curr['high']
                swing_idx = curr['bar_index']
                swing_time = curr['timestamp']

    if direction:
        final_type = 'HIGH' if direction == 'UP' else 'LOW'
        final_price = swing_high if direction == 'UP' else swing_low
        pivots.append({'bar_index': swing_idx, 'type': final_type, 'price': final_price, 'timestamp': swing_time})

    return pd.DataFrame(pivots)

# === Stop Hunt Detection ===
def detect_multibar_stop_hunt(df, pivots, tolerance=15.0, lookahead=3):
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
    stop_hunts = pd.DataFrame(hunts)
    if not stop_hunts.empty:
        stop_hunts['timestamp'] = stop_hunts['bar_index'].apply(lambda i: df.loc[df['bar_index'] == i, 'timestamp'].values[0])
        stop_hunts['timestamp'] = pd.to_datetime(stop_hunts['timestamp'], utc=True).dt.tz_convert("America/New_York")
    return stop_hunts

# === FVG Detection with Mitigation ===
def detect_fvgs(df, min_gap=5.0):
    out = []
    for i in range(len(df) - 2):
        A, C = df.iloc[i], df.iloc[i + 2]
        if A.high < C.low and (C.low - A.high) >= min_gap:
            out.append({'start_time': A.timestamp, 'end_index': i + 2, 'level1': A.high, 'level2': C.low, 'type': 'UP'})
        elif A.low > C.high and (A.low - C.high) >= min_gap:
            out.append({'start_time': A.timestamp, 'end_index': i + 2, 'level1': A.low, 'level2': C.high, 'type': 'DOWN'})

    for fvg in out:
        midpoint = (fvg['level1'] + fvg['level2']) / 2
        fvg['midpoint'] = midpoint
        fvg['mitigated'] = False
        mitigated_ts = None
        for j in range(fvg['end_index'] + 1, len(df)):
            bar = df.iloc[j]
            if fvg['type'] == 'UP' and bar['low'] <= midpoint:
                mitigated_ts = bar['timestamp']
                fvg['mitigated'] = True
                break
            elif fvg['type'] == 'DOWN' and bar['high'] >= midpoint:
                mitigated_ts = bar['timestamp']
                fvg['mitigated'] = True
                break
        fvg['end_time'] = mitigated_ts if mitigated_ts else df.iloc[-1]['timestamp']
    return pd.DataFrame(out)

# === Liquidity Pool Detection and Mitigation Check ===
def detect_liquidity_pools(pivots, df, tolerance=8.0, min_count=2):
    pools_raw = []
    for i in range(len(pivots)):
        base = pivots.iloc[i]
        cluster = [base]
        for j in range(i + 1, len(pivots)):
            compare = pivots.iloc[j]
            if compare['type'] == base['type'] and abs(compare['price'] - base['price']) <= tolerance:
                cluster.append(compare)

        if len(cluster) < min_count:
            continue

        cluster_df = pd.DataFrame(cluster)
        start_time = cluster_df['timestamp'].min()
        if base['type'] == 'HIGH':
            pool_price = cluster_df['price'].max()
        else:
            pool_price = cluster_df['price'].min()

        pools_raw.append({
            'start_time': start_time,
            'price': pool_price,
            'type': base['type']
        })

    # Mitigation check
    mitigated_pools = []
    last_time = df[df['timestamp'].dt.time <= time(16, 0)].iloc[-1]['timestamp']

    for pool in pools_raw:
        mitigated = False
        end_time = last_time
        reversed = False

        for _, bar in df[df['timestamp'] > pool['start_time']].iterrows():
            if pool['type'] == 'HIGH' and bar['high'] > pool['price']:
                mitigated = True
                end_time = bar['timestamp']
                if bar['close'] < pool['price']:
                    reversed = True
                break
            elif pool['type'] == 'LOW' and bar['low'] < pool['price']:
                mitigated = True
                end_time = bar['timestamp']
                if bar['close'] > pool['price']:
                    reversed = True
                break

        mitigated_pools.append({
            'start_time': pool['start_time'],
            'end_time': end_time,
            'price': pool['price'],
            'type': pool['type'],
            'mitigated': mitigated,
            'confluence': reversed
        })

    return pd.DataFrame(mitigated_pools)

# === Run detections ===
pivots = zigzag(df)
stop_hunts = detect_multibar_stop_hunt(df, pivots)
fvgs = detect_fvgs(df)
liq_pools = detect_liquidity_pools(pivots, df)

# === Export Pine Script ===
def export_overlay_pine(pivots, stop_hunts, fvgs, liq_pools, filename="pine_overlay.txt"):
    with open(filename, "w") as f:
        f.write("//@version=6\n")
        f.write("indicator(\"ICT Pivots, Stop Hunts, FVGs, Liquidity Pools\", overlay=true)\n\n")

        for _, row in pivots.iterrows():
            ts = pd.to_datetime(row['timestamp'])
            label = 'Pivot High' if row['type'] == 'HIGH' else 'Pivot Low'
            shape = 'labeldown' if row['type'] == 'HIGH' else 'labelup'
            location = 'abovebar' if row['type'] == 'HIGH' else 'belowbar'
            color = 'lime' if row['type'] == 'HIGH' else 'orange'
            ts_str = ts.strftime('%Y-%m-%dT%H:%M:%S-04:00')
            f.write(f"plotshape(time == timestamp(\"{ts_str}\"), location=location.{location}, style=shape.{shape}, text=\"{label}\", color=color.{color}, textcolor=color.white)\n")

        for _, row in stop_hunts.iterrows():
            ts = pd.to_datetime(row['timestamp'])
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
            color = 'gray' if row['mitigated'] else ('green' if row['type'] == 'UP' else 'red')
            f.write(f"box.new(left=timestamp(\"{ts_start}\"), right=timestamp(\"{ts_end}\"), top={y2}, bottom={y1}, xloc=xloc.bar_time, border_color=color.{color}, bgcolor=color.new(color.{color}, 85))\n")

        for _, row in liq_pools.iterrows():
            ts_start = row['start_time'].strftime('%Y-%m-%dT%H:%M:%S-04:00')
            ts_end = row['end_time'].strftime('%Y-%m-%dT%H:%M:%S-04:00')
            y = row['price']
            color = 'green' if row['type'] == 'HIGH' else 'red'
            thickness = '1' if row['mitigated'] else '2'
            f.write(f"line.new(x1=timestamp(\"{ts_start}\"), y1={y}, x2=timestamp(\"{ts_end}\"), y2={y}, xloc=xloc.bar_time, extend=extend.none, color=color.{color}, style=line.style_solid, width={thickness})\n")

# Export
export_overlay_pine(pivots, stop_hunts, fvgs, liq_pools)
print("✅ Pine script saved.")
