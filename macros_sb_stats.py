import pandas as pd
from datetime import time

# Load CSV
df = pd.read_csv('entry_signals_fvg_gp.csv')

# Clean column names
df.columns = [col.strip() for col in df.columns]

# Convert timestamp to datetime (UTC → ET)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')

# Drop invalid
df = df.dropna(subset=['timestamp'])

# Extract time
df['time'] = df['timestamp'].dt.time

# Define main macro & SB time blocks
macros = {
    "New-York AM Macro (08:50 - 09:10)": (time(8, 50), time(9, 10)),
    "New-York AM Macro (09:50 - 10:10)": (time(9, 50), time(10, 10)),
    "New-York AM Macro (10:50 - 11:10)": (time(10, 50), time(11, 10)),
    "New York Lunch Macro (11:50 - 12:10)": (time(11, 50), time(12, 10)),
    "New York Last Hour Macro (15:15 - 15:45)": (time(15, 15), time(15, 45)),
    "Silver Bullet AM (10:00 - 11:00)": (time(10, 0), time(11, 0)),
    "Silver Bullet PM (13:00 - 14:00)": (time(13, 0), time(14, 0)),

    # NEW custom time slots
    "Outside Macro: Pre-AM SB (09:10 - 09:50)": (time(9, 10), time(9, 50)),
    "Outside Macro: Pre-Lunch (11:10 - 11:50)": (time(11, 10), time(11, 50)),
    "Outside Macro: Lunch Gap (12:10 - 13:00)": (time(12, 10), time(13, 0)),
    "Outside Macro: Post-PM SB (14:00 - 15:15)": (time(14, 0), time(15, 15)),
    "Outside Macro: After Hours (15:45 - 17:00)": (time(15, 45), time(17, 0)),
}

# Points for win/loss
point_map = {'WIN': 60, 'LOSS': -20}

# Track used rows
covered_indices = set()
results = []

for name, (start, end) in macros.items():
    mask = df['time'].between(start, end)
    filtered = df[mask]
    covered_indices.update(filtered.index)
    total = len(filtered)
    wins = (filtered['result'] == 'WIN').sum()
    losses = (filtered['result'] == 'LOSS').sum()
    win_rate = (wins / total * 100) if total > 0 else 0
    points = filtered['result'].map(point_map).sum()

    results.append({
        'Macro': name,
        'Trades': total,
        'Wins': wins,
        'Losses': losses,
        'Win Rate (%)': round(win_rate, 2),
        'Total Points': points
    })

# Optional: truly outside all known blocks
outside_mask = ~df.index.isin(covered_indices)
outside_df = df[outside_mask]
if not outside_df.empty:
    total = len(outside_df)
    wins = (outside_df['result'] == 'WIN').sum()
    losses = (outside_df['result'] == 'LOSS').sum()
    win_rate = (wins / total * 100) if total > 0 else 0
    points = outside_df['result'].map(point_map).sum()

    results.append({
        'Macro': 'Outside Macro: Unclassified',
        'Trades': total,
        'Wins': wins,
        'Losses': losses,
        'Win Rate (%)': round(win_rate, 2),
        'Total Points': points
    })

# Output
summary_df = pd.DataFrame(results)
print(summary_df)
summary_df.to_csv('macro_trade_summary.csv', index=False)
