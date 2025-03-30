from ib_insync import *
from datetime import datetime, timedelta
import pandas as pd

# Connect to TWS/IB Gateway
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=1)

# Use local symbol directly
contract = Future(
    localSymbol='NQM5',
    exchange='CME',
    currency='USD'
)

# Verify contract
details = ib.reqContractDetails(contract)
if not details:
    print("❌ Contract could not be resolved with localSymbol.")
    ib.disconnect()
    exit()

resolved = details[0].contract
print(f"✅ Resolved Contract: {resolved.localSymbol}, Expiry: {resolved.lastTradeDateOrContractMonth}")

# Function to fetch one chunk of data
def fetch_chunk(end_dt):
    bars = ib.reqHistoricalData(
        resolved,
        endDateTime=end_dt.strftime('%Y%m%d %H:%M:%S'),
        durationStr='30 D',
        barSizeSetting='5 mins',
        whatToShow='TRADES',
        useRTH=False,
        formatDate=1
    )
    return util.df(bars)

# Get the last 90 days in 3 chunks
now = datetime.now()
chunks = []
for i in range(3):
    end_dt = now - timedelta(days=i * 30)
    print(f"📥 Fetching chunk ending on {end_dt.strftime('%Y-%m-%d')}")
    df = fetch_chunk(end_dt)
    chunks.append(df)

# Combine chunks
full_df = pd.concat(chunks).drop_duplicates(subset='date').sort_values('date')
full_df.set_index('date', inplace=True)
full_df = full_df[['open', 'high', 'low', 'close', 'volume']]

# Save to CSV
full_df.to_csv('nq_5min_90d.csv')
print("✅ Saved 90 days of 5-min data to nq_5min_90d.csv")

ib.disconnect()

# Save to CSV
import pandas as pd

# ib.disconnect()

# Load your IBKR-exported CSV
df = pd.read_csv("nq_5min_90d.csv")  # change to your filename
df['datetime'] = pd.to_datetime(df['date'])

# Localize as Chicago time (CME/GLOBEX is Chicago)
df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')

# Convert to New York time

# Format as: 2025-03-25T05:30:00-04:00
df['time'] = df['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S%z')
df['time'] = df['time'].str.replace(r'(\d{2})(\d{2})$', r'\1:\2', regex=True)

# Optional: Reorder and rename columns to match your strategy format
df_out = df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
df_out.columns = ['time', 'open', 'high', 'low', 'close', 'volume']

# Export to your preferred format
df_out.to_csv("NQM5_5m_90d_nytime.csv", index=False)