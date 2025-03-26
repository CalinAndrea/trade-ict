Features:
1. Pivots high/low
2. FVGs mitigated or not
3. Liquidity sweep BSL, SSL mitigated or not

What does 'confluence': reversed originally mean?
This flag was first introduced in stop-hunt-based liquidity detection logic. Here's what it tracked:

"Did price sweep the liquidity AND reverse back across it (in the opposite direction)?"
If price took out a liquidity pool (e.g., buy side at a high), and the close came back below that level → reversed = True
This is considered "smart money" style behavior:
Sweep liquidity (induce traders into bad trades)
Then reverse in the true direction
Can mark an entry zone for ICT-style setups


What does 'confluence' mean now in your current version?
You updated the logic to redefine confluence as:

"Is there an overlap between a liquidity pool and an unmitigated opposite-direction FVG (within tolerance)?"

