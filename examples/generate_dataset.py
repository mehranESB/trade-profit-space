from profitspace.dataset import save_dataset, load_dataset
from pathlib import Path
import pandas as pd

# import financial market data
csv_file_path = Path("./DATA/csv/EURUSD-1h.csv")
df_source = pd.read_csv(csv_file_path)

# save .pkl file
pkl_path = Path("./DATA/space/EURUSD-1h.pkl")
save_dataset(df_source, pkl_path)
