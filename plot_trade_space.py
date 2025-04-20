from profitspace.space import SpaceMaker, ProfitSpace
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# import financial market data
csv_file_path = Path("./DATA/csv/EURUSD-1h.csv")
df_source = pd.read_csv(csv_file_path)

# craete a space maker
maker = SpaceMaker(df_source, max_holds=15)

# get a sample space
space = maker[111]
space.plot_price_range(chandle=True)
space.plot_profit_space()
plt.show()

bom = 1
