from profitspace.space import SpaceMaker
from pathlib import Path
import pandas as pd

# import financial market data
csv_file_path = Path("./DATA/csv/EURUSD-1h.csv")
df_source = pd.read_csv(csv_file_path)

# craete a space maker
maker = SpaceMaker(df_source, max_holds=15)

# get a sample space
space = maker[111]

# create trades
trades = [
    {"otype": "buy", "upper_target": 1.19808, "lower_target": 1.19622},
    {"otype": "sell", "upper_target": 1.19808, "lower_target": 1.19622},
    {"otype": "buy", "upper_target": 1.19800, "lower_target": 1.19580},
    {"otype": "sell", "upper_target": 1.19847, "lower_target": 1.19551},
    {"otype": "buy", "upper_target": 1.19789, "lower_target": 1.19675},
]
otypes = [T["otype"] for T in trades]
upper_targets = [T["upper_target"] for T in trades]
lower_targets = [T["lower_target"] for T in trades]

# solve all at same time
results1 = space.check_trades(otypes, upper_targets, lower_targets)
print(results1)

# solve one by one
results2 = []
for trade in trades:
    results2.append(space.check_trade(**trade))
print(results2)
