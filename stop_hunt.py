import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, time

# ===== STEP 1: LOAD & RESAMPLE ENTIRE 1-MIN DATA TO 5-MIN =====
csv_path = "CME_MINI_NQ1!, 1 (1).csv"  # Replace with your CSV
df = pd.read_csv(csv_path)

df['timestamp'] = pd.to_datetime(df['time'])
df.sort_values('timestamp', inplace=True)
df.set_index('timestamp', inplace=True)

# (Optional) localize/convert to ET if needed:
# if df.index.tz is None:
#     df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')

df_5m = df.resample("5min").agg({
    'open':'first',
    'high':'max',
    'low':'min',
    'close':'last',
    'Volume':'sum'
}).dropna().reset_index()

df_5m.rename(columns={'Volume':'volume'}, inplace=True)
df_5m['bar_index'] = np.arange(len(df_5m))

# ===== STEP 2: FILTER 8:00–16:00 FOR DETECTION =====
def in_session_8to16(dt):
    t = dt.time()
    return (t >= time(8,0)) and (t < time(16,0))

df_session = df_5m[df_5m['timestamp'].apply(in_session_8to16)].copy()
# We keep bar_index from df_5m so we can map back to the full dataset.

# ===== STEP 3: ZIGZAG DETECTION =====
def zigzag(df, threshold=50.0):
    """
    Basic ZigZag approach:
    'threshold' points needed to reverse direction => new pivot.
    Returns pivot points: [bar_index, type, price, timestamp].
    """
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
                # Mark prior pivot as LOW
                pivots.append({
                    'bar_index': pivot_idx,
                    'type': 'LOW',
                    'price': pivot_price,
                    'timestamp': pivot_time
                })
                direction = 'UP'
                pivot_idx = curr_bi
                pivot_price = curr_close
                pivot_time  = curr_time
            elif curr_close < pivot_price - threshold:
                # Mark prior pivot as HIGH
                pivots.append({
                    'bar_index': pivot_idx,
                    'type': 'HIGH',
                    'price': pivot_price,
                    'timestamp': pivot_time
                })
                direction = 'DOWN'
                pivot_idx = curr_bi
                pivot_price = curr_close
                pivot_time  = curr_time
        else:
            if direction == 'UP':
                if curr_close < pivot_price - threshold:
                    # Mark prior pivot as HIGH
                    pivots.append({
                        'bar_index': pivot_idx,
                        'type': 'HIGH',
                        'price': pivot_price,
                        'timestamp': pivot_time
                    })
                    direction = 'DOWN'
                    pivot_idx = curr_bi
                    pivot_price = curr_close
                    pivot_time  = curr_time
                else:
                    if curr_close > pivot_price:
                        pivot_idx = curr_bi
                        pivot_price = curr_close
                        pivot_time  = curr_time
            else:  # direction == 'DOWN'
                if curr_close > pivot_price + threshold:
                    # Mark prior pivot as LOW
                    pivots.append({
                        'bar_index': pivot_idx,
                        'type': 'LOW',
                        'price': pivot_price,
                        'timestamp': pivot_time
                    })
                    direction = 'UP'
                    pivot_idx = curr_bi
                    pivot_price = curr_close
                    pivot_time  = curr_time
                else:
                    if curr_close < pivot_price:
                        pivot_idx = curr_bi
                        pivot_price = curr_close
                        pivot_time  = curr_time
    
    # Final pivot
    if direction is not None:
        if direction == 'UP':
            pivots.append({
                'bar_index': pivot_idx,
                'type': 'HIGH',
                'price': pivot_price,
                'timestamp': pivot_time
            })
        else:
            pivots.append({
                'bar_index': pivot_idx,
                'type': 'LOW',
                'price': pivot_price,
                'timestamp': pivot_time
            })
    
    return pd.DataFrame(pivots)

# ===== STEP 4: MERGE CLOSE PIVOTS =====
def merge_close_pivots(pivots_df, min_price_diff=50.0):
    """
    If consecutive pivots are within 'min_price_diff' points, merge them 
    into the later pivot => reduces minor zigzags.
    """
    pivots_df = pivots_df.sort_values('timestamp').reset_index(drop=True)
    merged = []
    last_pivot = None
    for _, row in pivots_df.iterrows():
        if last_pivot is None:
            last_pivot = row
        else:
            if abs(row['price'] - last_pivot['price']) < min_price_diff:
                # Merge => pick the later pivot
                last_pivot = row
            else:
                merged.append(last_pivot)
                last_pivot = row
    if last_pivot is not None:
        merged.append(last_pivot)
    return pd.DataFrame(merged)

# Example parameters
zigzag_threshold = 50.0
min_price_diff   = 50.0

raw_zz = zigzag(df_session, threshold=zigzag_threshold)
merged_zz = merge_close_pivots(raw_zz, min_price_diff=min_price_diff).sort_values('timestamp').reset_index(drop=True)

print("ZigZag raw in session [8:00–16:00]:")
print(raw_zz.head(10))
print("\nMerged pivots in session [8:00–16:00]:")
print(merged_zz.head(10))

# ===== STEP 5: DETECT STOP HUNTS (MULTI-BAR REVERSAL) =====
def detect_multibar_stop_hunt_for_swing(df, pivot, tolerance=5.0, lookahead=3):
    """
    Multi-bar approach:
      If pivot_type=HIGH:
        1) find the first bar (within 'lookahead') whose high > pivot_price + tolerance
        2) from that bar onward, look for any bar's close < pivot_price => SHORT
      If pivot_type=LOW:
        1) find the first bar whose low < pivot_price - tolerance
        2) from that bar onward, look for any bar's close > pivot_price => LONG
    """
    pivot_index = pivot['bar_index']
    pivot_price = pivot['price']
    pivot_type  = pivot['type']
    
    # We look up to pivot_index + lookahead for the entire check
    df_future = df[(df['bar_index'] > pivot_index) & (df['bar_index'] <= pivot_index + lookahead)]
    if df_future.empty:
        return None
    
    if pivot_type == 'HIGH':
        # Step 1: find the bar that sweeps above pivot_price + tolerance
        exceed = df_future[df_future['high'] > pivot_price + tolerance]
        if not exceed.empty:
            first_exceed_idx = exceed.iloc[0]['bar_index']
            # Step 2: from that bar forward, see if any bar's close < pivot_price
            df_after_exceed = df_future[df_future['bar_index'] >= first_exceed_idx]
            reversal = df_after_exceed[df_after_exceed['close'] < pivot_price]
            if not reversal.empty:
                # The first bar that closes < pivot_price triggers the short
                rev_bar = reversal.iloc[0]
                return {
                    'pivot_index': pivot_index,
                    'pivot_type': 'HIGH',
                    'pivot_price': pivot_price,
                    'pivot_time': pivot['timestamp'],
                    'hunt_bar_index': rev_bar['bar_index'],
                    'hunt_timestamp': rev_bar['timestamp'],
                    'direction': 'SHORT'
                }
    else:  # pivot_type == 'LOW'
        # Step 1: find the bar that dips below pivot_price - tolerance
        exceed = df_future[df_future['low'] < pivot_price - tolerance]
        if not exceed.empty:
            first_exceed_idx = exceed.iloc[0]['bar_index']
            # Step 2: from that bar forward, see if any bar's close > pivot_price
            df_after_exceed = df_future[df_future['bar_index'] >= first_exceed_idx]
            reversal = df_after_exceed[df_after_exceed['close'] > pivot_price]
            if not reversal.empty:
                rev_bar = reversal.iloc[0]
                return {
                    'pivot_index': pivot_index,
                    'pivot_type': 'LOW',
                    'pivot_price': pivot_price,
                    'pivot_time': pivot['timestamp'],
                    'hunt_bar_index': rev_bar['bar_index'],
                    'hunt_timestamp': rev_bar['timestamp'],
                    'direction': 'LONG'
                }
    return None

def detect_stop_hunts_multibar(df, pivots, tolerance=5.0, lookahead=3):
    hunts = []
    for _, pivot in pivots.iterrows():
        hunt = detect_multibar_stop_hunt_for_swing(df, pivot, tolerance, lookahead)
        if hunt is not None:
            hunts.append(hunt)
    return pd.DataFrame(hunts)

stop_hunts_df = detect_stop_hunts_multibar(df_session, merged_zz, tolerance=5.0, lookahead=3)
print("\nDetected Multi-bar Stop Hunts in 8:00–16:00:")
print(stop_hunts_df)

# ===== STEP 6: MAP PIVOTS & HUNTS BACK TO FULL df_5m & PLOT =====
df_5m['zz_high'] = np.nan
df_5m['zz_low']  = np.nan
for _, row in merged_zz.iterrows():
    match_idx = df_5m.index[df_5m['bar_index'] == row['bar_index']]
    if len(match_idx)==1:
        i = match_idx[0]
        if row['type'] == 'HIGH':
            df_5m.loc[i,'zz_high'] = row['price']
        else:
            df_5m.loc[i,'zz_low']  = row['price']

df_5m['hunt_short'] = np.nan
df_5m['hunt_long']  = np.nan
for _, hunt in stop_hunts_df.iterrows():
    match_idx = df_5m.index[df_5m['bar_index'] == hunt['hunt_bar_index']]
    if len(match_idx)==1:
        i = match_idx[0]
        pivot_price = hunt['pivot_price']  # marker at pivot level
        if hunt['direction'] == 'SHORT':
            df_5m.loc[i,'hunt_short'] = pivot_price
        else:
            df_5m.loc[i,'hunt_long']  = pivot_price

ap_zz_high = mpf.make_addplot(df_5m['zz_high'],   type='scatter', marker='^', color='green', markersize=200)
ap_zz_low  = mpf.make_addplot(df_5m['zz_low'],    type='scatter', marker='v', color='red',   markersize=200)
ap_hunt_s  = mpf.make_addplot(df_5m['hunt_short'],type='scatter', marker='x', color='purple',markersize=200)
ap_hunt_l  = mpf.make_addplot(df_5m['hunt_long'], type='scatter', marker='+', color='blue',  markersize=200)

mpf.plot(
    df_5m.set_index('timestamp'),
    type='candle', volume=True,
    addplot=[ap_zz_high, ap_zz_low, ap_hunt_s, ap_hunt_l],
    title="ZigZag + Merge + Multi-bar Stop Hunts (8:00–16:00, hunts at pivot price)",
    style='yahoo',
    warn_too_much_data=99999
)
