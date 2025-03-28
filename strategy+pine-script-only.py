import pandas as pd
import numpy as np
from datetime import time
from collections import deque
from scipy.signal import find_peaks

# === Load 5-minute data ===
df = pd.read_csv("CME_MINI_NQ1!, 5 (8).csv")
df['timestamp'] = pd.to_datetime(df['time'], utc=True)
df = df.sort_values('timestamp').reset_index(drop=True)
df['bar_index'] = np.arange(len(df))
df['timestamp'] = df['timestamp'].dt.tz_convert("America/New_York")
df = df.set_index("timestamp")  # Ensure timestamp is the index


def detect_swing_highs_lows(df, distance=2, prominence=2):
    highs = df['high'].values
    lows = df['low'].values

    high_peaks, _ = find_peaks(highs, distance=distance, prominence=prominence)
    low_peaks, _ = find_peaks(-lows, distance=distance, prominence=prominence)

    swing = pd.Series(data=np.nan, index=df.index, name="HighLow")
    level = pd.Series(data=np.nan, index=df.index, name="Level")

    swing.iloc[high_peaks] = 1
    swing.iloc[low_peaks] = -1
    level.iloc[high_peaks] = df['high'].iloc[high_peaks]
    level.iloc[low_peaks] = df['low'].iloc[low_peaks]

    result = pd.concat([swing, level], axis=1)
    return result

# === FVG Detection with Mitigation ===
def detect_fvgs(df, min_gap=5.0):
    out = []
    for i in range(len(df) - 2):
        A, C = df.iloc[i], df.iloc[i + 2]
        if A.high < C.low and (C.low - A.high) >= min_gap:
            out.append({'start_time': A.name, 'end_index': i + 2, 'level1': A.high, 'level2': C.low, 'type': 'UP'})
        elif A.low > C.high and (A.low - C.high) >= min_gap:
            out.append({'start_time': A.name, 'end_index': i + 2, 'level1': A.low, 'level2': C.high, 'type': 'DOWN'})

    for fvg in out:
        midpoint = (fvg['level1'] + fvg['level2']) / 2
        fvg['midpoint'] = midpoint
        fvg['mitigated'] = False
        mitigated_ts = None
        for j in range(fvg['end_index'] + 1, len(df)):
            bar = df.iloc[j]
            if fvg['type'] == 'UP' and bar['low'] <= midpoint:
                mitigated_ts = bar.name
                fvg['mitigated'] = True
                break
            elif fvg['type'] == 'DOWN' and bar['high'] >= midpoint:
                mitigated_ts = bar.name
                fvg['mitigated'] = True
                break
        fvg['end_time'] = mitigated_ts if mitigated_ts else df.iloc[-1].name
    return pd.DataFrame(out)

# === Liquidity Pool Detection and Mitigation Check ===
def detect_liquidity_pools(pivots, df, fvgs, tolerance=8.0, min_count=2):
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

    mitigated_pools = []
    last_time = df[df.index.time <= time(16, 0)].iloc[-1].name

    for pool in pools_raw:
        mitigated = False
        end_time = last_time
        reversed_sweep = False
        confluence = False

        for _, bar in df[df.index > pool['start_time']].iterrows():
            if pool['type'] == 'HIGH' and bar['high'] > pool['price']:
                mitigated = True
                end_time = bar.name
                if bar['close'] < pool['price']:
                    reversed_sweep = True
                break
            elif pool['type'] == 'LOW' and bar['low'] < pool['price']:
                mitigated = True
                end_time = bar.name
                if bar['close'] > pool['price']:
                    reversed_sweep = True
                break

        relevant_fvgs = fvgs[(fvgs['start_time'] <= end_time) & (~fvgs['mitigated'])]
        for _, fvg in relevant_fvgs.iterrows():
            if pool['type'] == 'HIGH' and fvg['type'] == 'DOWN' and abs(fvg['midpoint'] - pool['price']) <= tolerance:
                confluence = True
                break
            elif pool['type'] == 'LOW' and fvg['type'] == 'UP' and abs(fvg['midpoint'] - pool['price']) <= tolerance:
                confluence = True
                break

        mitigated_pools.append({
            'start_time': pool['start_time'],
            'end_time': end_time,
            'price': pool['price'],
            'type': pool['type'],
            'mitigated': mitigated,
            'reversed': reversed_sweep,
            'confluence': confluence
        })

    return pd.DataFrame(mitigated_pools)

# === Fib Zone Detection with Golden Pocket ===
def detect_fib_zones(pivots, df, min_range=30.0, lookahead=4):
    fib_zones = []
    if len(pivots) < 2:
        return pd.DataFrame(columns=['start_time', 'end_time', 'golden_high', 'golden_low', 'direction'])

    for i in range(len(pivots) - 1):
        p1 = pivots.iloc[i]
        for j in range(i + 1, min(i + lookahead + 1, len(pivots))):
            p2 = pivots.iloc[j]

            if p1['type'] == 'LOW' and p2['type'] == 'HIGH':
                low, high = p1['price'], p2['price']
                direction = 'UP'
            elif p1['type'] == 'HIGH' and p2['type'] == 'LOW':
                high, low = p1['price'], p2['price']
                direction = 'DOWN'
            else:
                continue

            if abs(high - low) < min_range:
                continue

            golden_low = high - (high - low) * 0.79
            golden_high = high - (high - low) * 0.62
            if direction == 'DOWN':
                golden_low = low + (high - low) * 0.62
                golden_high = low + (high - low) * 0.79

            midpoint = (golden_high + golden_low) / 2
            mitigated = False
            end_time = df.iloc[-1].name
            mask = df.index > (p2['timestamp'] if 'timestamp' in p2 else p2.name)
            for _, bar in df[mask].iterrows():
                if direction == 'UP' and bar['low'] <= midpoint:
                    mitigated = True
                    end_time = bar.name
                    break
                elif direction == 'DOWN' and bar['high'] >= midpoint:
                    mitigated = True
                    end_time = bar.name
                    break

            fib_zones.append({
                'start_time': p1['timestamp'] if 'timestamp' in p1 else p1.name,
                'end_time': end_time,
                'golden_high': golden_high,
                'golden_low': golden_low,
                'direction': direction,
                'mitigated': mitigated
            })

    return pd.DataFrame(fib_zones)

# === Clean Trend-Based Pivot Reduction ===
def reduce_to_major_swings(pivots):
    reduced = []
    direction = None
    current_extreme = None

    for i, row in pivots.iterrows():
        hl = row['HighLow']
        price = row['Level']

        if pd.isna(hl):
            continue

        if direction is None:
            direction = hl
            current_extreme = (i, price, hl)
            continue

        if hl == direction:
            if (hl == 1 and price > current_extreme[1]) or (hl == -1 and price < current_extreme[1]):
                current_extreme = (i, price, hl)
        else:
            reduced.append({
                'timestamp': current_extreme[0],
                'price': current_extreme[1],
                'type': 'HIGH' if current_extreme[2] == 1 else 'LOW'
            })
            direction = hl
            current_extreme = (i, price, hl)

    if current_extreme:
        reduced.append({
            'timestamp': current_extreme[0],
            'price': current_extreme[1],
            'type': 'HIGH' if current_extreme[2] == 1 else 'LOW'
        })

    return pd.DataFrame(reduced)

def detect_order_blocks(df, pivots, min_range=30):
    blocks = []
    for i in range(1, len(pivots)):
        prev = pivots.iloc[i - 1]
        curr = pivots.iloc[i]
        range_size = abs(curr['price'] - prev['price'])
        if range_size < min_range:
            continue

        range_df = df[(df.index >= prev['timestamp']) & (df.index <= curr['timestamp'])]

        if prev['type'] == 'LOW' and curr['type'] == 'HIGH':
            ob_type = 'BULLISH'
            ob_candle = range_df.loc[range_df['low'].idxmin()]
        elif prev['type'] == 'HIGH' and curr['type'] == 'LOW':
            ob_type = 'BEARISH'
            ob_candle = range_df.loc[range_df['high'].idxmax()]
        else:
            continue

        price = (ob_candle['open'] + ob_candle['close']) / 2
        ob_time = ob_candle.name

        # Look for the next opposite swing to mark the end of the move
        future_pivots = pivots[pivots['timestamp'] > curr['timestamp']]
        opposite_type = 'LOW' if curr['type'] == 'HIGH' else 'HIGH'
        next_swing = future_pivots[future_pivots['type'] == opposite_type]

        if next_swing.empty:
            continue

        swing_confirm_time = next_swing.iloc[0]['timestamp']
        mitigation_df = df[df.index > swing_confirm_time]

        mitigated = False
        end_time = df.iloc[-1].name
        for _, bar in mitigation_df.iterrows():
            if ob_type == "BULLISH" and bar['low'] <= price:
                mitigated = True
                end_time = bar.name
                break
            elif ob_type == "BEARISH" and bar['high'] >= price:
                mitigated = True
                end_time = bar.name
                break

        blocks.append({
            'start_time': ob_time,
            'end_time': end_time,
            'price': price,
            'type': ob_type,
            'mitigated': mitigated
        })

    return pd.DataFrame(blocks)

# === Market Structure Detection with Line to Break Candle Body ===
def detect_structure(pivots, df):
    structure = []
    i = 2
    while i < len(pivots):
        p1, p2, p3 = pivots.iloc[i - 2], pivots.iloc[i - 1], pivots.iloc[i]

        if p1['type'] == 'HIGH' and p2['type'] == 'LOW' and p3['type'] == 'HIGH' and p3['price'] > p1['price']:
            range_df = df.loc[p2['timestamp']:p3['timestamp']]
            break_candle = range_df[(range_df[['open', 'close']].max(axis=1) > p1['price'])].head(1)
            if not break_candle.empty:
                candle_time = break_candle.index[0]
                structure.append({
                    'timestamp': p1['timestamp'],
                    'bar_index': i - 2,
                    'confirm_time': candle_time,
                    'price': p1['price'],
                    'type': 'BOS' if candle_time != p3['timestamp'] else 'CHoCH'
                })

        elif p1['type'] == 'LOW' and p2['type'] == 'HIGH' and p3['type'] == 'LOW' and p3['price'] < p1['price']:
            range_df = df.loc[p2['timestamp']:p3['timestamp']]
            break_candle = range_df[(range_df[['open', 'close']].min(axis=1) < p1['price'])].head(1)
            if not break_candle.empty:
                candle_time = break_candle.index[0]
                structure.append({
                    'timestamp': p1['timestamp'],
                    'bar_index': i - 2,
                    'confirm_time': candle_time,
                    'price': p1['price'],
                    'type': 'BOS' if candle_time != p3['timestamp'] else 'CHoCH'
                })

        i += 1
    return pd.DataFrame(structure)

# === Breaker Block Detection (ICT-accurate, Pine-style) ===
# def detect_breaker_blocks(pivots, df):
#     breakers = []
#     for i in range(3, len(pivots)):
#         pA, pB, pC = pivots.iloc[i - 3], pivots.iloc[i - 2], pivots.iloc[i - 1]

#         if pA['type'] == 'LOW' and pB['type'] == 'HIGH' and pC['type'] == 'LOW':
#             if pC['price'] < pA['price']:  # MSS
#                 sub_df = df.loc[pA['timestamp']:pB['timestamp']]
#                 ob = sub_df[(sub_df['open'] > sub_df['close'])]  # Bearish candle
#                 if ob.empty:
#                     continue
#                 ob_candle = ob.iloc[-1]
#                 breaker = {
#                     'start_time': ob_candle.name,
#                     'end_time': df.index[-1],
#                     'top': ob_candle['high'],
#                     'bottom': ob_candle['low'],
#                     'type': 'BULLISH',
#                     'mitigated': False
#                 }
#                 for j in range(df.index.get_loc(ob_candle.name) + 1, len(df)):
#                     candle = df.iloc[j]
#                     if candle['close'] < breaker['bottom']:
#                         breaker['mitigated'] = True
#                         breaker['end_time'] = candle.name
#                         break
#                 breakers.append(breaker)

#         elif pA['type'] == 'HIGH' and pB['type'] == 'LOW' and pC['type'] == 'HIGH':
#             if pC['price'] > pA['price']:
#                 sub_df = df.loc[pA['timestamp']:pB['timestamp']]
#                 ob = sub_df[(sub_df['open'] < sub_df['close'])]  # Bullish candle
#                 if ob.empty:
#                     continue
#                 ob_candle = ob.iloc[-1]
#                 breaker = {
#                     'start_time': ob_candle.name,
#                     'end_time': df.index[-1],
#                     'top': ob_candle['high'],
#                     'bottom': ob_candle['low'],
#                     'type': 'BEARISH',
#                     'mitigated': False
#                 }
#                 for j in range(df.index.get_loc(ob_candle.name) + 1, len(df)):
#                     candle = df.iloc[j]
#                     if candle['close'] > breaker['top']:
#                         breaker['mitigated'] = True
#                         breaker['end_time'] = candle.name
#                         break
#                 breakers.append(breaker)

#     return pd.DataFrame(breakers)



# === Run detections ===
swings = detect_swing_highs_lows(df, distance=5, prominence=5)
swings["type"] = swings["HighLow"].map({1: "HIGH", -1: "LOW"})
swings["price"] = swings["Level"]
swings["timestamp"] = swings.index
pivots = reduce_to_major_swings(swings)

fvgs = detect_fvgs(df)
fib_zones = detect_fib_zones(pivots, df)
liq_pools = detect_liquidity_pools(pivots, df, fvgs)
order_blocks = detect_order_blocks(df, pivots)
structure = detect_structure(pivots, df)
# breakers = detect_breaker_blocks(pivots, df)

# === Export Pine Script ===
def export_overlay_pine(pivots, fvgs, liq_pools, fib_zones, structure, order_blocks, filename="pine_overlay.txt"):
    with open(filename, "w") as f:
        f.write("//@version=6\n")
        f.write("indicator(\"ICT Pivots, Stop Hunts, FVGs, Liquidity Pools\", overlay=true)\n\n")

        for _, row in pivots.iterrows():
            ts = pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%dT%H:%M:%S-04:00")
            is_low = row["type"] == "LOW"
            swing_type = "labelup" if is_low else "labeldown"
            location = "belowbar" if is_low else "abovebar"
            color = "orange" if is_low else "lime"
            label = "Low" if is_low else "High"
            f.write(f"plotshape(time == timestamp(\"{ts}\"), location=location.{location}, style=shape.{swing_type}, text=\"{label}\", color=color.{color}, textcolor=color.white)\n")

        for _, row in fvgs.iterrows():
            ts_start = pd.to_datetime(row['start_time']).strftime('%Y-%m-%dT%H:%M:%S-04:00')
            ts_end = pd.to_datetime(row['end_time']).strftime('%Y-%m-%dT%H:%M:%S-04:00')
            y1, y2 = sorted([row['level1'], row['level2']])
            color = 'gray' if row['mitigated'] else ('green' if row['type'] == 'UP' else 'red')
            f.write(f"box.new(left=timestamp(\"{ts_start}\"), right=timestamp(\"{ts_end}\"), top={y2}, bottom={y1}, xloc=xloc.bar_time, border_color=color.{color}, bgcolor=color.new(color.{color}, 85))\n")

        for _, row in liq_pools.iterrows():
            ts_start = pd.to_datetime(row['start_time']).strftime('%Y-%m-%dT%H:%M:%S-04:00')
            ts_end = pd.to_datetime(row['end_time']).strftime('%Y-%m-%dT%H:%M:%S-04:00')
            y = row['price']
            color = 'green' if row['type'] == 'HIGH' else 'red'
            if row['confluence']:
                color = 'blue'
            thickness = '1' if row['mitigated'] else '2'
            f.write(f"line.new(x1=timestamp(\"{ts_start}\"), y1={y}, x2=timestamp(\"{ts_end}\"), y2={y}, xloc=xloc.bar_time, extend=extend.none, color=color.{color}, style=line.style_solid, width={thickness})\n")

        for _, row in fib_zones.iterrows():
            ts_start = pd.to_datetime(row['start_time']).strftime('%Y-%m-%dT%H:%M:%S-04:00')
            ts_end = pd.to_datetime(row['end_time']).strftime('%Y-%m-%dT%H:%M:%S-04:00')
            color = 'green' if row['direction'] == 'UP' else 'red'
            if row['mitigated']:
                color = 'gray'
            f.write(f"box.new(left=timestamp(\"{ts_start}\"), right=timestamp(\"{ts_end}\"), top={row['golden_high']}, bottom={row['golden_low']}, xloc=xloc.bar_time, border_color=color.{color}, border_style=line.style_dashed, bgcolor=color.new(color.{color}, 80))\n")

        for _, row in structure.iterrows():
            ts = row['timestamp'].strftime("%Y-%m-%dT%H:%M:%S-04:00")
            ts_confirm = row['confirm_time'].strftime("%Y-%m-%dT%H:%M:%S-04:00")
            color = "blue" if row['type'] == "BOS" else "orange"
            y = row['price']
            mid_ts = pd.to_datetime(row['timestamp']) + (pd.to_datetime(row['confirm_time']) - pd.to_datetime(row['timestamp'])) / 2
            mid_ts_str = mid_ts.strftime("%Y-%m-%dT%H:%M:%S-04:00")
            f.write(f"line.new(x1=timestamp(\"{ts}\"), x2=timestamp(\"{ts_confirm}\"), y1={y}, y2={y}, xloc=xloc.bar_time, extend=extend.none, color=color.{color})\n")
            f.write(f"label.new(x=timestamp(\"{mid_ts_str}\"), y={y}, text=\"{row['type']}\", style=label.style_none, color=color.new(color.{color}, 0), textcolor=color.{color}, xloc=xloc.bar_time)\n")

        for _, row in order_blocks.iterrows():
            ts_start = row['start_time'].strftime("%Y-%m-%dT%H:%M:%S-04:00")
            ts_end = row['end_time'].strftime("%Y-%m-%dT%H:%M:%S-04:00")
            color = "gray" if row['mitigated'] else ("green" if row['type'] == "BULLISH" else "red")
            top = row['price'] + 1
            bottom = row['price'] - 1
            f.write(f"box.new(left=timestamp(\"{ts_start}\"), right=timestamp(\"{ts_end}\"), top={top}, bottom={bottom}, xloc=xloc.bar_time, extend=extend.none, border_color=color.{color}, bgcolor=color.new(color.{color}, 85))\n")

        # for _, row in breakers.iterrows():
        #     ts1 = row['start_time'].strftime("%Y-%m-%dT%H:%M:%S-04:00")
        #     ts2 = row['end_time'].strftime("%Y-%m-%dT%H:%M:%S-04:00")
        #     color = "green" if row['type'] == 'BULLISH' else "red"
        #     f.write(f"box.new(left=timestamp(\"{ts1}\"), right=timestamp(\"{ts2}\"), top={row['top']}, bottom={row['bottom']}, xloc=xloc.bar_time, extend=extend.none, border_color=color.{color}, bgcolor=color.new(color.{color}, 85))\n")

# Export
export_overlay_pine(pivots, fvgs, liq_pools, fib_zones, structure, order_blocks)
print("✅ Pine script saved.")
