from ib_insync import *

# ib = IB()
# ib.connect('127.0.0.1', 7496, clientId=1)

# # Use local symbol directly
# contract = Future(
#     localSymbol='NQM5',
#     exchange='CME',
#     currency='USD'
# )

# # Request contract details to verify it resolves
# details = ib.reqContractDetails(contract)
# if not details:
#     print("❌ Contract could not be resolved with localSymbol.")
#     ib.disconnect()
#     exit()

# resolved = details[0].contract
# print(f"✅ Resolved Contract: {resolved.localSymbol}, Expiry: {resolved.lastTradeDateOrContractMonth}")

# # Request 5-min bars for 30 days (delayed or live depending on your account)
# bars = ib.reqHistoricalData(
#     resolved,
#     endDateTime='',
#     durationStr='30 D',
#     barSizeSetting='5 mins',
#     whatToShow='TRADES',
#     useRTH=False,
#     formatDate=1
# )

# Save to CSV
import pandas as pd

# ib.disconnect()

# Load your IBKR-exported CSV
df = pd.read_csv("NQM5_5m_30d.csv")  # change to your filename
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
df_out.to_csv("NQM5_5m_30d_nytime.csv", index=False)