import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.patches as patches

def detect_fvgs_3candle(df_5m, min_gap=20.0):
    """
    Detects Fair Value Gaps (FVGs) using a 3‑candle method:
      - UP FVG if Candle A.high < Candle C.low and (C.low - A.high) >= min_gap
      - DOWN FVG if Candle A.low > Candle C.high and (A.low - C.high) >= min_gap
    Returns a DataFrame with columns:
      [start_index, end_index, type, start_time, end_time, level1, level2, gap, midpoint]
    """
    out = []
    df_reset = df_5m.reset_index(drop=True)
    for i in range(len(df_reset) - 2):
        A = df_reset.iloc[i]
        B = df_reset.iloc[i+1]
        C = df_reset.iloc[i+2]
        # Upward FVG
        if A["high"] < C["low"]:
            gap = C["low"] - A["high"]
            if gap >= min_gap:
                out.append({
                    "start_index": i,
                    "end_index": i+2,
                    "type": "UP",
                    "start_time": A["timestamp"],
                    "end_time":   C["timestamp"],
                    "level1": A["high"],
                    "level2": C["low"],
                    "gap": gap,
                    "midpoint": (A["high"] + C["low"]) / 2
                })
        # Downward FVG
        if A["low"] > C["high"]:
            gap = A["low"] - C["high"]
            if gap >= min_gap:
                out.append({
                    "start_index": i,
                    "end_index": i+2,
                    "type": "DOWN",
                    "start_time": A["timestamp"],
                    "end_time":   C["timestamp"],
                    "level1": A["low"],
                    "level2": C["high"],
                    "gap": gap,
                    "midpoint": (A["low"] + C["high"]) / 2
                })
    return pd.DataFrame(out)

def mark_mitigated_fvgs(fvg_df, df_5m):
    """
    For each detected FVG, mark it as mitigated if any subsequent 5‑minute candle's close 
    (after the FVG's end) crosses the gap's boundary.
      - For an UP FVG: if any subsequent close < level2 (C.low), then the gap is filled.
      - For a DOWN FVG: if any subsequent close > level2 (C.high), then the gap is filled.
    Adds a boolean column "mitigated" to fvg_df.
    """
    mitigated = []
    for i, fvg in fvg_df.iterrows():
        if fvg["type"] == "UP":
            boundary = fvg["level2"]  # Candle C.low
        else:
            boundary = fvg["level2"]  # Candle C.high

        start_idx = fvg["end_index"] + 1
        if start_idx >= len(df_5m):
            mitigated.append(False)
            continue
        
        subsequent = df_5m.iloc[start_idx:]
        if fvg["type"] == "UP":
            # Mitigated if any subsequent close is below the gap's upper boundary
            mitigated.append((subsequent["close"] < boundary).any())
        else:  # DOWN FVG
            mitigated.append((subsequent["close"] > boundary).any())
    
    fvg_df["mitigated"] = mitigated
    return fvg_df

def main():
    # ===== STEP 1: LOAD CSV =====
    csv_path = "CME_MINI_NQ1!, 1 (1).csv"  # Update with your CSV filename
    df = pd.read_csv(csv_path)
    
    # Convert 'time' column to datetime and sort
    df['timestamp'] = pd.to_datetime(df['time'])
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    
    # ===== STEP 2: FILL MISSING 1-MINUTE BARS =====
    # Create a full 1-min date range from earliest to latest
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1T')
    df = df.reindex(full_range)
    
    # ===== STEP 3: RESAMPLE TO 5-MINUTE BARS =====
    df_5m = df.resample("5min").agg({
        'open':  'first',
        'high':  'max',
        'low':   'min',
        'close': 'last',
        'Volume':'sum'
    }).dropna().reset_index()
    
    df_5m.rename(columns={'index': 'timestamp', 'Volume': 'volume'}, inplace=True)
    df_5m['bar_index'] = np.arange(len(df_5m))
    
    # ===== STEP 4: DETECT FVGs =====
    min_gap = 5  # Adjust minimum gap threshold
    fvg_df = detect_fvgs_3candle(df_5m, min_gap=min_gap)
    print("Detected FVGs:")
    print(fvg_df.head(10))
    
    if fvg_df.empty:
        print("No FVGs detected.")
        return
    
    # ===== STEP 5: MARK MITIGATED FVGs =====
    fvg_df = mark_mitigated_fvgs(fvg_df, df_5m)
    print("\nFVGs with mitigation status:")
    print(fvg_df.head(10))
    
    # ===== STEP 6: PREPARE OHLC DATA FOR candlestick_ohlc =====
    ohlc_data = []
    for idx, row in df_5m.iterrows():
        t = mdates.date2num(row['timestamp'])
        ohlc_data.append([t, row['open'], row['high'], row['low'], row['close']])
    
    # ===== STEP 7: PLOT THE FULL 5-MINUTE CHART =====
    fig, ax = plt.subplots(figsize=(20,10))
    # Plot the candlestick chart
    candlestick_ohlc(ax, ohlc_data, width=0.0008, colorup='g', colordown='r')
    ax.xaxis_date()
    ax.set_title(f"Full 5-Min Chart (All 1-min bars filled) with FVG Rectangles (min_gap={min_gap})")
    ax.set_ylabel("Price")
    
    # ===== STEP 8: DRAW FVG RECTANGLES =====
    extension_factor = 5
    zorder = 1000
    
    # - UP FVG non-mitigated: green
    # - DOWN FVG non-mitigated: red
    # - Mitigated FVGs: blue
    for idx, row in fvg_df.iterrows():
        x_start = mdates.date2num(row['start_time'])
        x_end   = mdates.date2num(row['end_time'])
        original_width = x_end - x_start
        new_width = original_width * extension_factor
        
        y_bottom = min(row['level1'], row['level2'])
        y_top    = max(row['level1'], row['level2'])
        height = y_top - y_bottom
        
        if row["mitigated"]:
            patch_color = "blue"
        else:
            patch_color = "green" if row["type"] == "UP" else "red"
        
        rect = patches.Rectangle(
            (x_start, y_bottom),
            new_width,
            height,
            fill=True,
            facecolor=patch_color,
            alpha=0.3,
            zorder=zorder
        )
        ax.add_patch(rect)
    
    plt.show()

if __name__ == "__main__":
    main()
