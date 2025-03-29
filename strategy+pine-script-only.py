import pandas as pd
import numpy as np
from datetime import time
from collections import deque
from scipy.signal import find_peaks

# === Load 5-minute data ===
df = pd.read_csv("NQM5_5m_30d_nytime.csv")
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
def detect_fvgs(df, min_gap=10):
    out = []
    for i in range(len(df) - 3):  # need i+3 to be valid
        A, C = df.iloc[i], df.iloc[i + 2]

        if A.high < C.low and (C.low - A.high) >= min_gap:
            fvg = {
                'start_time': A.name,
                'end_index': i + 2,
                'level1': A.high,
                'level2': C.low,
                'type': 'UP'
            }
        elif A.low > C.high and (A.low - C.high) >= min_gap:
            fvg = {
                'start_time': A.name,
                'end_index': i + 2,
                'level1': A.low,
                'level2': C.high,
                'type': 'DOWN'
            }
        else:
            continue  # no valid FVG

        # 🔁 Activation happens at candle AFTER C (i+3)
        fvg['activation_time'] = df.iloc[i + 3].name

        # Midpoint and default mitigation state
        midpoint = (fvg['level1'] + fvg['level2']) / 2
        fvg['midpoint'] = midpoint
        fvg['mitigated'] = False

        # Search for mitigation after FVG is formed
        mitigated_ts = None
        for j in range(fvg['end_index'] + 1, len(df)):
            bar = df.iloc[j]
            if fvg['type'] == 'UP' and bar['low'] <= midpoint:
                fvg['mitigated'] = True
                mitigated_ts = bar.name
                break
            elif fvg['type'] == 'DOWN' and bar['high'] >= midpoint:
                fvg['mitigated'] = True
                mitigated_ts = bar.name
                break

        fvg['end_time'] = mitigated_ts if mitigated_ts else df.iloc[-1].name
        out.append(fvg)

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

    # ✅ Now convert to DataFrame and compute direction
    df_struct = pd.DataFrame(structure)

    if not df_struct.empty:
        df_struct['direction'] = df_struct.apply(
            lambda row: 'BULLISH' if row['price'] < df.loc[row['confirm_time']]['close'] else 'BEARISH',
            axis=1
        )

    return df_struct


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

def detect_entry_signals(df, fvgs, pivots, structure):
    entries = []

    for i in range(10, len(df)):
        candle = df.iloc[i]
        ts = candle.name

        # Only allow entries during NY session
        if ts.time() < time(8, 35) or ts.time() > time(15, 0):
            continue

        # Get most recent structure event
        recent_struct = structure[structure['confirm_time'] < ts].tail(1)
        if recent_struct.empty:
            continue

        valid_fvgs = fvgs[
            (fvgs['activation_time'].notna()) &
            (ts >= fvgs['activation_time']) &
            (ts <= fvgs['end_time']) &
            (candle['low'] <= fvgs['midpoint']) &
            (candle['high'] >= fvgs['midpoint'])
        ]

        for _, fvg in valid_fvgs.iterrows():
            midpoint = fvg['midpoint']
            is_bullish = fvg['type'] == 'UP'

            # Step 1: Ensure pivot before this FVG
            pivot_before_fvg = pivots[pivots['timestamp'] < fvg['start_time']]
            if pivot_before_fvg.empty:
                continue

            # Step 2: Ensure FVG not mitigated by the time of entry
            price_range = df[(df.index > fvg['activation_time']) & (df.index < ts)]
            if is_bullish and any(price_range['low'] <= midpoint):
                continue
            if not is_bullish and any(price_range['high'] >= midpoint):
                continue

            # Step 3: Structure must align with FVG direction
            struct = recent_struct.iloc[0]

            if is_bullish and struct['direction'] != 'BULLISH':
                continue
            if not is_bullish and struct['direction'] != 'BEARISH':
                continue


            # ✅ Step 4: Create entry
            entry = midpoint
            stop = entry - 20 if is_bullish else entry + 20
            target = entry + 40 if is_bullish else entry - 40

            entries.append({
                'timestamp': ts,
                'type': 'LONG' if is_bullish else 'SHORT',
                'entry': entry,
                'stop': stop,
                'target': target,
                'reason': 'Unmitigated FVG + Structure',
                'fvg_start_time': fvg['start_time'],
                'fvg_level1': fvg['level1'],
                'fvg_level2': fvg['level2'],
                'structure_type': struct['type'],
                'structure_price': struct['price'],
                'structure_confirm_time': struct['confirm_time'],
            })

    return pd.DataFrame(entries)

def simulate_trade_exits(entries, df):
    exit_times = []
    results = []

    for _, entry in entries.iterrows():
        ts = entry['timestamp']
        is_long = entry['type'] == 'LONG'
        stop = entry['stop']
        target = entry['target']
        entry_idx = df.index.get_loc(ts)

        exit_time = df.iloc[-1].name  # default: end of dataset
        result = "OPEN"

        for j in range(entry_idx + 1, len(df)):
            bar = df.iloc[j]

            if is_long:
                if bar['low'] <= stop:
                    exit_time = bar.name
                    result = "LOSS"
                    break
                elif bar['high'] >= target:
                    exit_time = bar.name
                    result = "WIN"
                    break
            else:
                if bar['high'] >= stop:
                    exit_time = bar.name
                    result = "LOSS"
                    break
                elif bar['low'] <= target:
                    exit_time = bar.name
                    result = "WIN"
                    break

        exit_times.append(exit_time)
        results.append(result)

    entries['exit_time'] = exit_times
    entries['result'] = results
    return entries


def export_entry_signals_to_pine(entries, df, filename="entry_positions_overlay.txt"):

    with open(filename, "w") as f:
        f.write("//@version=6\n")
        f.write("indicator(\"Unmitigated FVG Entries\", overlay=true)\n\n")

        for idx, row in entries.iterrows():
            ts = pd.to_datetime(row['timestamp'])
            
            ts_exit = pd.to_datetime(row['exit_time']) if pd.notna(row.get('exit_time')) else df.index[-1]

            ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S-04:00")
            ts_exit_str = ts_exit.strftime("%Y-%m-%dT%H:%M:%S-04:00")

            entry = round(row['entry'], 2)
            stop = round(row['stop'], 2)
            target = round(row['target'], 2)
            direction = row['type']
            is_long = direction == 'LONG'
            result = row['result'] if pd.notna(row.get('result')) else 'OPEN'

            rr = round(abs((target - entry) / (entry - stop)), 2) if (entry - stop) != 0 else 0.0
            tp_color = "green"
            sl_color = "red"
            fill_tp = "color.new(color.green, 80)"
            fill_sl = "color.new(color.red, 80)"

            # === Entry marker
            f.write(f"plotshape(time == timestamp(\"{ts_str}\"), location=location.belowbar, style=shape.triangleup, color=color.{tp_color}, size=size.small)\n" if is_long else
                    f"plotshape(time == timestamp(\"{ts_str}\"), location=location.abovebar, style=shape.triangledown, color=color.{sl_color}, size=size.small)\n")
            f.write(f"line.new(x1=timestamp(\"{ts_str}\"), x2=timestamp(\"{(ts + pd.Timedelta(minutes=20)).strftime('%Y-%m-%dT%H:%M:%S-04:00')}\"), y1={entry}, y2={entry}, xloc=xloc.bar_time, extend=extend.none, color=color.blue, width=1)\n")

            # === TP/SL zones as boxes (always draw)
            tp_top = max(entry, target)
            tp_bottom = min(entry, target)
            sl_top = max(entry, stop)
            sl_bottom = min(entry, stop)

            f.write(f"box.new(left=timestamp(\"{ts_str}\"), right=timestamp(\"{ts_exit_str}\"), top={tp_top}, bottom={tp_bottom}, xloc=xloc.bar_time, bgcolor={fill_tp}, border_color=color.{tp_color})\n")
            f.write(f"box.new(left=timestamp(\"{ts_str}\"), right=timestamp(\"{ts_exit_str}\"), top={sl_top}, bottom={sl_bottom}, xloc=xloc.bar_time, bgcolor={fill_sl}, border_color=color.{sl_color})\n")

            # FVG context box
            fvg_top = max(row['fvg_level1'], row['fvg_level2'])
            fvg_bottom = min(row['fvg_level1'], row['fvg_level2'])
            fvg_start_str = pd.to_datetime(row['fvg_start_time']).strftime("%Y-%m-%dT%H:%M:%S-04:00")

            fvg_color = "green" if row['fvg_level1'] < row['fvg_level2'] else "red"
            f.write(f"box.new(left=timestamp(\"{fvg_start_str}\"), right=timestamp(\"{ts_str}\"), top={fvg_top}, bottom={fvg_bottom}, xloc=xloc.bar_time, border_color=color.{fvg_color}, bgcolor=color.new(color.{fvg_color}, 85))\n")


def is_unmitigated_at_time(fvg_or_gp_row, ts):
    return ts < fvg_or_gp_row['end_time']

entries = detect_entry_signals(df, fvgs, pivots, structure)
entries = simulate_trade_exits(entries, df)

entries.to_csv("entry_signals_fvg_gp.csv", index=False)
print(f"✅ {len(entries)} entries detected based on FVG + GP only.")

export_entry_signals_to_pine(entries, df)
print("✅ Entries exported to Pine script.")

def print_trade_stats(entries):
    if entries.empty:
        print("⚠️ No trades to analyze.")
        return

    # Sanity check
    required_cols = {'entry', 'target', 'stop', 'result'}
    if not required_cols.issubset(entries.columns):
        print(f"❌ Missing required columns: {required_cols - set(entries.columns)}")
        return

    # Win/loss separation
    wins = entries[entries['result'] == 'WIN']
    losses = entries[entries['result'] == 'LOSS']

    # Calculate points
    wins['points'] = abs(wins['target'] - wins['entry'])
    losses['points'] = -abs(losses['entry'] - losses['stop'])

    all_trades = pd.concat([wins, losses])
    total = len(all_trades)
    win_rate = round(len(wins) / total * 100, 2) if total > 0 else 0
    total_points = round(all_trades['points'].sum(), 2)
    avg_points = round(all_trades['points'].mean(), 2)

    print("📊 Trade Performance Summary:")
    print(f"🔢 Total Trades        : {total}")
    print(f"✅ Wins                : {len(wins)}")
    print(f"❌ Losses              : {len(losses)}")
    print(f"📈 Win Rate            : {win_rate}%")
    print(f"💰 Total Points Gained : {total_points}")
    print(f"⚖️ Avg Points/Trade    : {avg_points}")


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

        # for _, row in fvgs.iterrows():
        #     ts_start = pd.to_datetime(row['start_time']).strftime('%Y-%m-%dT%H:%M:%S-04:00')
        #     ts_end = pd.to_datetime(row['end_time']).strftime('%Y-%m-%dT%H:%M:%S-04:00')
        #     y1, y2 = sorted([row['level1'], row['level2']])
        #     color = 'gray' if row['mitigated'] else ('green' if row['type'] == 'UP' else 'red')
        #     f.write(f"box.new(left=timestamp(\"{ts_start}\"), right=timestamp(\"{ts_end}\"), top={y2}, bottom={y1}, xloc=xloc.bar_time, border_color=color.{color}, bgcolor=color.new(color.{color}, 85))\n")

        # for _, row in liq_pools.iterrows():
        #     ts_start = pd.to_datetime(row['start_time']).strftime('%Y-%m-%dT%H:%M:%S-04:00')
        #     ts_end = pd.to_datetime(row['end_time']).strftime('%Y-%m-%dT%H:%M:%S-04:00')
        #     y = row['price']
        #     color = 'green' if row['type'] == 'HIGH' else 'red'
        #     if row['confluence']:
        #         color = 'blue'
        #     thickness = '1' if row['mitigated'] else '2'
        #     f.write(f"line.new(x1=timestamp(\"{ts_start}\"), y1={y}, x2=timestamp(\"{ts_end}\"), y2={y}, xloc=xloc.bar_time, extend=extend.none, color=color.{color}, style=line.style_solid, width={thickness})\n")

        # for _, row in fib_zones.iterrows():
        #     ts_start = pd.to_datetime(row['start_time']).strftime('%Y-%m-%dT%H:%M:%S-04:00')
        #     ts_end = pd.to_datetime(row['end_time']).strftime('%Y-%m-%dT%H:%M:%S-04:00')
        #     color = 'green' if row['direction'] == 'UP' else 'red'
        #     if row['mitigated']:
        #         color = 'gray'
        #     f.write(f"box.new(left=timestamp(\"{ts_start}\"), right=timestamp(\"{ts_end}\"), top={row['golden_high']}, bottom={row['golden_low']}, xloc=xloc.bar_time, border_color=color.{color}, border_style=line.style_dashed, bgcolor=color.new(color.{color}, 80))\n")

        for _, row in structure.iterrows():
            ts = row['timestamp'].strftime("%Y-%m-%dT%H:%M:%S-04:00")
            ts_confirm = row['confirm_time'].strftime("%Y-%m-%dT%H:%M:%S-04:00")
            color = "blue" if row['type'] == "BOS" else "orange"
            y = row['price']
            mid_ts = pd.to_datetime(row['timestamp']) + (pd.to_datetime(row['confirm_time']) - pd.to_datetime(row['timestamp'])) / 2
            mid_ts_str = mid_ts.strftime("%Y-%m-%dT%H:%M:%S-04:00")
            f.write(f"line.new(x1=timestamp(\"{ts}\"), x2=timestamp(\"{ts_confirm}\"), y1={y}, y2={y}, xloc=xloc.bar_time, extend=extend.none, color=color.{color})\n")
            f.write(f"label.new(x=timestamp(\"{mid_ts_str}\"), y={y}, text=\"{row['type']}\", style=label.style_none, color=color.new(color.{color}, 0), textcolor=color.{color}, xloc=xloc.bar_time)\n")

        # for _, row in order_blocks.iterrows():
        #     ts_start = row['start_time'].strftime("%Y-%m-%dT%H:%M:%S-04:00")
        #     ts_end = row['end_time'].strftime("%Y-%m-%dT%H:%M:%S-04:00")
        #     color = "gray" if row['mitigated'] else ("green" if row['type'] == "BULLISH" else "red")
        #     top = row['price'] + 1
        #     bottom = row['price'] - 1
        #     f.write(f"box.new(left=timestamp(\"{ts_start}\"), right=timestamp(\"{ts_end}\"), top={top}, bottom={bottom}, xloc=xloc.bar_time, extend=extend.none, border_color=color.{color}, bgcolor=color.new(color.{color}, 85))\n")

        # for _, row in breakers.iterrows():
        #     ts1 = row['start_time'].strftime("%Y-%m-%dT%H:%M:%S-04:00")
        #     ts2 = row['end_time'].strftime("%Y-%m-%dT%H:%M:%S-04:00")
        #     color = "green" if row['type'] == 'BULLISH' else "red"
        #     f.write(f"box.new(left=timestamp(\"{ts1}\"), right=timestamp(\"{ts2}\"), top={row['top']}, bottom={row['bottom']}, xloc=xloc.bar_time, extend=extend.none, border_color=color.{color}, bgcolor=color.new(color.{color}, 85))\n")

# Export
export_overlay_pine(pivots, fvgs, liq_pools, fib_zones, structure, order_blocks)
print("✅ Pine script saved.")

print("📊 Trade Statistics:")
print_trade_stats(entries)

