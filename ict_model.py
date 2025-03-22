import pandas as pd
from datetime import time, datetime

# ----- Step 1: Load Data -----
csv_path = "CME_MINI_NQ1!, 1 (1).csv"  # Replace with your CSV filename
df_full = pd.read_csv(csv_path)

# Assume CSV has a "time" column with ISO timestamps
df_full['timestamp'] = pd.to_datetime(df_full['time'])

# Convert to Eastern Time if needed. If timestamps are not tz-aware, assume UTC.
if df_full['timestamp'].dt.tz is None:
    df_full['timestamp'] = df_full['timestamp'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
else:
    df_full['timestamp'] = df_full['timestamp'].dt.tz_convert('America/New_York')

# Keep only relevant columns and sort by timestamp
df_full = df_full[['timestamp', 'open', 'high', 'low', 'close', 'Volume']]
df_full.sort_values('timestamp', inplace=True)

# Create a 'date' column for the full data (for extra prices)
df_full['date'] = df_full['timestamp'].dt.date

# ----- Step 2: Define Sessions (Using ICT Session Times) -----
# Note: Midnight (00:00) and 08:30 do not belong to any session.
def get_session(dt):
    t = dt.time()
    if time(20, 0) <= t < time(23, 59, 59):
        return "Asia"
    elif time(2, 0) <= t < time(5, 0):
        return "London"
    elif time(9, 30) <= t < time(11, 0):
        return "NY AM"
    elif time(12, 0) <= t < time(13, 0):
        return "NY Lunch"
    elif time(13, 30) <= t < time(16, 0):
        return "NY PM"
    else:
        return None

# Create a copy for session summary and filter by sessions.
df_session = df_full.copy()
df_session['session'] = df_session['timestamp'].apply(get_session)
df_session = df_session[df_session['session'].notna()].copy()

# ----- Step 3: Create a 'date' Column for Grouping (already in df_full) -----
# df_session already has a 'date' column inherited from df_full.

# ----- Step 4: Group by Date & Session, Calculate Session High/Low/Open/Close -----
session_summary = df_session.groupby(['date', 'session']).agg(
    session_high=('high', 'max'),
    session_low=('low', 'min'),
    session_open=('open', 'first'),
    session_close=('close', 'last')
).reset_index()

# ----- Step 5: Extract Midnight and 08:30 Data for Each Day from the Full Data -----
def get_row_at_time(df_day, target_time):
    """
    Return the first row in df_day where the time portion equals target_time.
    If no such row exists, return None.
    """
    df_day = df_day.copy()
    df_day['time_only'] = df_day['timestamp'].dt.time
    result = df_day[df_day['time_only'] == target_time]
    if not result.empty:
        return result.iloc[0]
    else:
        return None

midnight_rows = []
open830_rows = []

for d in sorted(df_full['date'].unique()):
    df_day = df_full[df_full['date'] == d].copy()
    midnight_row = get_row_at_time(df_day, time(0, 0))
    open830_row = get_row_at_time(df_day, time(8, 30))
    if midnight_row is not None:
        midnight_rows.append(midnight_row)
    if open830_row is not None:
        open830_rows.append(open830_row)

midnight_df = pd.DataFrame(midnight_rows)
open830_df = pd.DataFrame(open830_rows)

# ----- Step 6: Rename Columns in Extra Prices DataFrames -----
if not midnight_df.empty:
    midnight_df = midnight_df.rename(columns={
        'timestamp': 'midnight_timestamp',
        'open': 'midnight_open',
        'high': 'midnight_high',
        'low': 'midnight_low',
        'close': 'midnight_close',
        'Volume': 'midnight_Volume'
    })
if not open830_df.empty:
    open830_df = open830_df.rename(columns={
        'timestamp': 'open830_timestamp',
        'open': 'open830',
        'high': 'open830_high',
        'low': 'open830_low',
        'close': 'open830_close',
        'Volume': 'open830_Volume'
    })

# ----- Step 7: Output the Data Separately -----
print("Session Summary:")
print(session_summary.sort_values(['date', 'session']).tail(30))

print("\nMidnight Data (if available):")
if not midnight_df.empty:
    print(midnight_df[['midnight_timestamp', 'midnight_open', 'midnight_high', 'midnight_low', 'midnight_close', 'midnight_Volume']])
else:
    print("No midnight data available.")

print("\n08:30 Data (if available):")
if not open830_df.empty:
    print(open830_df[['open830_timestamp', 'open830', 'open830_high', 'open830_low', 'open830_close', 'open830_Volume']])
else:
    print("No 08:30 data available.")
