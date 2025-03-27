import pandas as pd
import numpy as np
from datetime import time
from collections import deque

from scipy.signal import find_peaks


# === Load 5-minute data ===
df = pd.read_csv("CME_MINI_NQ1!, 5 (7).csv")
df['timestamp'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert("America/New_York")
df = df.sort_values('timestamp')
df = df.set_index("timestamp")
df['bar_index'] = np.arange(len(df))



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
    result.index.name = "timestamp"  # Optional: for clarity
    return result

# # === Zigzag detection ===
# def zigzag(df, threshold=20.0):
#     if df.empty:
#         return pd.DataFrame(columns=['bar_index','type','price','timestamp'])

#     pivots = []
#     direction = None
#     last_pivot = df.iloc[0]
#     swing_high = last_pivot['high']
#     swing_low = last_pivot['low']
#     swing_idx = last_pivot['bar_index']
#     swing_time = last_pivot['timestamp']

#     for i in range(1, len(df)):
#         curr = df.iloc[i]

#         if direction is None:
#             if curr['close'] > last_pivot['close'] + threshold:
#                 direction = 'UP'
#                 swing_low = last_pivot['low']
#                 swing_idx = last_pivot['bar_index']
#                 swing_time = last_pivot['timestamp']
#             elif curr['close'] < last_pivot['close'] - threshold:
#                 direction = 'DOWN'
#                 swing_high = last_pivot['high']
#                 swing_idx = last_pivot['bar_index']
#                 swing_time = last_pivot['timestamp']
#         elif direction == 'UP':
#             if curr['high'] > swing_high:
#                 swing_high = curr['high']
#                 swing_idx = curr['bar_index']
#                 swing_time = curr['timestamp']
#             elif curr['close'] < swing_high - threshold:
#                 pivots.append({'bar_index': swing_idx, 'type': 'HIGH', 'price': swing_high, 'timestamp': swing_time})
#                 direction = 'DOWN'
#                 swing_low = curr['low']
#                 swing_idx = curr['bar_index']
#                 swing_time = curr['timestamp']
#         elif direction == 'DOWN':
#             if curr['low'] < swing_low:
#                 swing_low = curr['low']
#                 swing_idx = curr['bar_index']
#                 swing_time = curr['timestamp']
#             elif curr['close'] > swing_low + threshold:
#                 pivots.append({'bar_index': swing_idx, 'type': 'LOW', 'price': swing_low, 'timestamp': swing_time})
#                 direction = 'UP'
#                 swing_high = curr['high']
#                 swing_idx = curr['bar_index']
#                 swing_time = curr['timestamp']

#     if direction:
#         final_type = 'HIGH' if direction == 'UP' else 'LOW'
#         final_price = swing_high if direction == 'UP' else swing_low
#         pivots.append({'bar_index': swing_idx, 'type': final_type, 'price': final_price, 'timestamp': swing_time})

#     return pd.DataFrame(pivots)

# # === FVG Detection with Mitigation ===
# def detect_fvgs(df, min_gap=5.0):
#     out = []
#     for i in range(len(df) - 2):
#         A, C = df.iloc[i], df.iloc[i + 2]
#         if A.high < C.low and (C.low - A.high) >= min_gap:
#             out.append({'start_time': A.timestamp, 'end_index': i + 2, 'level1': A.high, 'level2': C.low, 'type': 'UP'})
#         elif A.low > C.high and (A.low - C.high) >= min_gap:
#             out.append({'start_time': A.timestamp, 'end_index': i + 2, 'level1': A.low, 'level2': C.high, 'type': 'DOWN'})

#     for fvg in out:
#         midpoint = (fvg['level1'] + fvg['level2']) / 2
#         fvg['midpoint'] = midpoint
#         fvg['mitigated'] = False
#         mitigated_ts = None
#         for j in range(fvg['end_index'] + 1, len(df)):
#             bar = df.iloc[j]
#             if fvg['type'] == 'UP' and bar['low'] <= midpoint:
#                 mitigated_ts = bar['timestamp']
#                 fvg['mitigated'] = True
#                 break
#             elif fvg['type'] == 'DOWN' and bar['high'] >= midpoint:
#                 mitigated_ts = bar['timestamp']
#                 fvg['mitigated'] = True
#                 break
#         fvg['end_time'] = mitigated_ts if mitigated_ts else df.iloc[-1]['timestamp']
#     return pd.DataFrame(out)

# # === Liquidity Pool Detection and Mitigation Check ===
# def detect_liquidity_pools(pivots, df, fvgs, tolerance=8.0, min_count=2):
#     pools_raw = []
#     for i in range(len(pivots)):
#         base = pivots.iloc[i]
#         cluster = [base]
#         for j in range(i + 1, len(pivots)):
#             compare = pivots.iloc[j]
#             if compare['type'] == base['type'] and abs(compare['price'] - base['price']) <= tolerance:
#                 cluster.append(compare)

#         if len(cluster) < min_count:
#             continue

#         cluster_df = pd.DataFrame(cluster)
#         start_time = cluster_df['timestamp'].min()
#         if base['type'] == 'HIGH':
#             pool_price = cluster_df['price'].max()
#         else:
#             pool_price = cluster_df['price'].min()

#         pools_raw.append({
#             'start_time': start_time,
#             'price': pool_price,
#             'type': base['type']
#         })

#     mitigated_pools = []
#     last_time = df[df['timestamp'].dt.time <= time(16, 0)].iloc[-1]['timestamp']

#     for pool in pools_raw:
#         mitigated = False
#         end_time = last_time
#         reversed_sweep = False
#         confluence = False

#         for _, bar in df[df['timestamp'] > pool['start_time']].iterrows():
#             if pool['type'] == 'HIGH' and bar['high'] > pool['price']:
#                 mitigated = True
#                 end_time = bar['timestamp']
#                 if bar['close'] < pool['price']:
#                     reversed_sweep = True
#                 break
#             elif pool['type'] == 'LOW' and bar['low'] < pool['price']:
#                 mitigated = True
#                 end_time = bar['timestamp']
#                 if bar['close'] > pool['price']:
#                     reversed_sweep = True
#                 break

#         # === Check for confluence with unmitigated FVGs ===
#         relevant_fvgs = fvgs[(fvgs['start_time'] <= end_time) & (~fvgs['mitigated'])]
#         for _, fvg in relevant_fvgs.iterrows():
#             if pool['type'] == 'HIGH' and fvg['type'] == 'DOWN' and abs(fvg['midpoint'] - pool['price']) <= tolerance:
#                 confluence = True
#                 break
#             elif pool['type'] == 'LOW' and fvg['type'] == 'UP' and abs(fvg['midpoint'] - pool['price']) <= tolerance:
#                 confluence = True
#                 break

#         mitigated_pools.append({
#             'start_time': pool['start_time'],
#             'end_time': end_time,
#             'price': pool['price'],
#             'type': pool['type'],
#             'mitigated': mitigated,
#             'reversed': reversed_sweep,
#             'confluence': confluence
#         })

#     return pd.DataFrame(mitigated_pools)

# # === Fib Zone Detection with Golden Pocket ===
# def detect_fib_zones(pivots, df, min_range=40.0):
#     fib_zones = []
#     if len(pivots) < 2:
#         return pd.DataFrame(columns=['start_time', 'end_time', 'golden_high', 'golden_low', 'direction'])

#     for i in range(len(pivots) - 1):
#         p1 = pivots.iloc[i]
#         p2 = pivots.iloc[i+1]
#         if p1['type'] == 'LOW' and p2['type'] == 'HIGH':
#             low, high = p1['price'], p2['price']
#             direction = 'UP'
#         elif p1['type'] == 'HIGH' and p2['type'] == 'LOW':
#             high, low = p1['price'], p2['price']
#             direction = 'DOWN'
#         else:
#             continue

#         if abs(high - low) < min_range:
#             continue

#         golden_low = high - (high - low) * 0.79
#         golden_high = high - (high - low) * 0.62
#         if direction == 'DOWN':
#             golden_low = low + (high - low) * 0.62
#             golden_high = low + (high - low) * 0.79

#         # Mitigation logic
#         midpoint = (golden_high + golden_low) / 2
#         mitigated = False
#         end_time = df.iloc[-1]['timestamp']
#         for _, bar in df[df['timestamp'] > p2['timestamp']].iterrows():
#             if direction == 'UP' and bar['low'] <= midpoint:
#                 mitigated = True
#                 end_time = bar['timestamp']
#                 break
#             elif direction == 'DOWN' and bar['high'] >= midpoint:
#                 mitigated = True
#                 end_time = bar['timestamp']
#                 break

#         fib_zones.append({
#             'start_time': p1['timestamp'],
#             'end_time': end_time,
#             'golden_high': golden_high,
#             'golden_low': golden_low,
#             'direction': direction,
#             'mitigated': mitigated
#         })

#     return pd.DataFrame(fib_zones)

# # === Order Block Detection ===
# def detect_order_blocks(df, pivots, min_range=30):
#     blocks = []
#     for i in range(1, len(pivots)):
#         prev = pivots.iloc[i - 1]
#         curr = pivots.iloc[i]
#         range_size = abs(curr['price'] - prev['price'])
#         if range_size < min_range:
#             continue

#         if prev['type'] == 'LOW' and curr['type'] == 'HIGH':
#             ob_type = 'BULLISH'
#             range_df = df[(df['bar_index'] >= prev['bar_index']) & (df['bar_index'] <= curr['bar_index'])]
#             ob_candle = range_df.iloc[range_df['low'].idxmin() - range_df.index[0]]
#         elif prev['type'] == 'HIGH' and curr['type'] == 'LOW':
#             ob_type = 'BEARISH'
#             range_df = df[(df['bar_index'] >= prev['bar_index']) & (df['bar_index'] <= curr['bar_index'])]
#             ob_candle = range_df.iloc[range_df['high'].idxmax() - range_df.index[0]]
#         else:
#             continue

#         blocks.append({
#             'start_time': ob_candle['timestamp'],
#             'price': ob_candle['close'],
#             'type': ob_type
#         })
#     return pd.DataFrame(blocks)

# # === BOS and CHoCH Detection ===
# # === Improved BOS and CHoCH Detection ===
# def detect_structure(pivots, bos_buffer=2.0, choch_min_move=20):
#     structure = []
#     trend = None  # 'UP' or 'DOWN'
#     last_high = None
#     last_low = None

#     for i in range(len(pivots)):
#         pivot = pivots.iloc[i]

#         if pivot['type'] == 'HIGH':
#             if trend == 'DOWN' and last_high is not None and pivot['price'] > last_high + choch_min_move:
#                 structure.append({'timestamp': pivot['timestamp'], 'bar_index': pivot['bar_index'], 'type': 'CHoCH'})
#                 trend = 'UP'
#             elif trend == 'UP' and last_high is not None and pivot['price'] > last_high + bos_buffer:
#                 structure.append({'timestamp': pivot['timestamp'], 'bar_index': pivot['bar_index'], 'type': 'BOS'})
#             last_high = pivot['price']

#         elif pivot['type'] == 'LOW':
#             if trend == 'UP' and last_low is not None and pivot['price'] < last_low - choch_min_move:
#                 structure.append({'timestamp': pivot['timestamp'], 'bar_index': pivot['bar_index'], 'type': 'CHoCH'})
#                 trend = 'DOWN'
#             elif trend == 'DOWN' and last_low is not None and pivot['price'] < last_low - bos_buffer:
#                 structure.append({'timestamp': pivot['timestamp'], 'bar_index': pivot['bar_index'], 'type': 'BOS'})
#             last_low = pivot['price']

#     return pd.DataFrame(structure)


# # === Run detections ===
# pivots = zigzag(df)
# fvgs = detect_fvgs(df)
# fib_zones = detect_fib_zones(pivots, df)
# liq_pools = detect_liquidity_pools(pivots, df, fvgs)
# order_blocks = detect_order_blocks(df, pivots)
# structure = detect_structure(pivots)

# === Export Pine Script ===
def export_overlay_pine(swings, filename="pine_overlay.txt"):
    with open(filename, "w") as f:
        f.write("//@version=6\n")
        f.write("indicator(\"ICT Pivots, Stop Hunts, FVGs, Liquidity Pools\", overlay=true)\n\n")

        # === Swings ===
        # Ensure swings index is datetime and properly aligned
        for i in swings[~swings["HighLow"].isna()].index:
            ts = pd.to_datetime(i).strftime("%Y-%m-%dT%H:%M:%S-04:00")
            swing_type = "labelup" if swings.loc[i, "HighLow"] == -1 else "labeldown"
            location = "belowbar" if swings.loc[i, "HighLow"] == -1 else "abovebar"
            color = "orange" if swings.loc[i, "HighLow"] == -1 else "lime"
            label = "Low" if swings.loc[i, "HighLow"] == -1 else "High"
            f.write(f"plotshape(time == timestamp(\"{ts}\"), location=location.{location}, style=shape.{swing_type}, text=\"{label}\", color=color.{color}, textcolor=color.white)\n")


swings = detect_swing_highs_lows(df, distance=5, prominence=5)

# Export
export_overlay_pine(swings)
print("✅ Pine script saved.")
