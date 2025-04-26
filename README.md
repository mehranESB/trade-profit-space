# trade-profit-space 

## 🌀 ProfitSpace: Mapping Trade Results into 2D Profit Regions

### ✨ Overview

**ProfitSpace** is a Python package for mapping trading outcomes into a 2D **Profit Space**, based on upper and lower price targets instead of time sequences.

It allows you to create a `ProfitSpace` object from market OHLC datasets, and use this transformed view to **compute trade outcomes instantly** — without needing to process future candle movements during trading.

By leveraging Profit Space, you can **train deep learning models** to predict trade options (win, loss, expiration, uncertainty) directly from the current market state, enabling much **faster, more scalable**, and **more effective** financial model training.

ProfitSpace provides a new way to think about market behavior, making the future of price action analyzable at a glance through spatial relationships instead of temporal ones.

### 📈 Visualization

The following animation demonstrates the transformation:

- **Left Side:** Price movement over time, with upper/lower trade targets.

- **Right Side:** Mapped Profit Space showing regions for trade outcomes.

<p align="center">
  <img src="images/two_space_comparison.gif" alt="ProfitSpace Visualization" />
</p>

### 🗺️ Profit Space Regions Explained

n the **Profit Space**, each point represents a trade defined by its **Upper Target** (x-axis) and **Lower Target** (y-axis).
The space is divided into four regions based on how price would interact with these targets (In all cases, the order is executed at the **open price** of the first candle.):

| 🧭 Region         | 🎨 Color | 📈 Trade Setup                                      | 🧾 Outcome                                      |
|------------------|----------|----------------------------------------------------|------------------------------------------------|
| 🔵 Buy Region     | Blue     | Take Profit at **Upper Target** (x), Stop Loss at **Lower Target** (y) | Upper Target is hit first → **Buy Wins**       |
| 🔴 Sell Region    | Red      | Take Profit at **Lower Target** (y), Stop Loss at **Upper Target** (x) | Lower Target is hit first → **Sell Wins**      |
| 🟡 Unknown Region | Yellow   | Both targets are set                               | Both are touched in same candle → **Uncertain** |
| ⚪ Expire Region  | Gray     | Targets set but neither hit                        | Trade expires without target → **No Result**    |

## 🚀 Installation

**📦 Option 1: Install via pip**
```bash
pip install trade-profit-space 
```
**🛠 Option 2: Clone and install manually**
```bash
git clone https://github.com/mehranESB/trade-profit-space.git
cd trade-profit-space
python setup.py install
```

## 📚 Usage Guide

**1️⃣ Creating a Profit Space**

This example shows how to create a `ProfitSpace` object from historical market data.
You can either manually extract the candle data or use a `SpaceMaker` helper class for easier access to multiple samples.
```python 
from profitspace import SpaceMaker, ProfitSpace
from pathlib import Path
import pandas as pd

# Load financial market data from CSV
csv_file_path = Path("./DATA/csv/EURUSD-1h.csv")
df_source = pd.read_csv(csv_file_path)

# Define parameters
idx = 150       # Sample row index for the starting candle
maxhold = 15    # Maximum number of candles to hold the position

# Manually extract Open, High, Low, Close values for the trade horizon
cols = ["Open", "High", "Low", "Close"]
op, hi, lo, cl = df_source.iloc[idx : idx + maxhold, cols].to_numpy()

# Create ProfitSpace manually
profit_space = ProfitSpace(op, hi, lo, cl)

# --- Easier and recommended way ---

# Create a SpaceMaker instance for the dataset
space_maker = SpaceMaker(df_source, max_holds=maxhold)

# Retrieve the ProfitSpace at the specified index
profit_space2 = space_maker[idx]  # Equivalent to profit_space
```

**2️⃣ Visualizing Profit Space and Checking Trades**

This example demonstrates how to plot an interactive visualization of the Profit Space using `matplotlib`,
how to evaluate whether a specific trade setup would win or lose,
and how to identify the region (Buy, Sell, Unknown, Expire) for given target values.
```python
# Visualize the Profit Space interactively alongside the Price Space
profit_space.plot_map_targets()

# --- Check if a trade setup would win or lose ---

# Example: Sell order setup
sl = 1.19808  # Stop Loss (upper target for sell)
tp = 1.19622  # Take Profit (lower target for sell)

# Check whether the trade would win (True) or lose (False)
win_lose = profit_space.check_trade("sell", upper_target=sl, lower_target=tp)
print("Trade Result:", win_lose)

# --- Identify the region for a specific pair of targets ---

ut = 1.19847  # Upper Target
lt = 1.19551  # Lower Target

# Get the region name: {"unknown", "buy", "sell", "expire", "invalid"}
region = profit_space.get_region(upper_target=ut, lower_target=lt)
print("Region:", region)
```

**3️⃣ Using ProfitSpace as a DataLoader for Model Training**

This example demonstrates how to integrate **ProfitSpace** with a **PyTorch** `DataLoader` for training machine learning models. A custom `Dataset` class loads sequences of OHLC data along with their corresponding `ProfitSpace` objects from a preprocessed dataset. The OHLC sequences serve as input features, while the `ProfitSpace` provides a way to evaluate the outcome of trades without needing to simulate future candles during training. This approach allows for efficient, large-scale training of financial models by directly mapping model predictions (target prices) to trade outcomes in the profit space.
```python
from profitspace.dataset import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
import numpy as np

# --- Custom collate function to batch OHLC and keep ProfitSpace objects as list ---

def custom_collate_fn(batch):
    ohlc = [item["ohlc"] for item in batch]
    spaces = [item["space"] for item in batch]
    return {
        "ohlc": torch.tensor(ohlc),
        "space": spaces,  # Keep ProfitSpace objects as is
    }

# --- Custom PyTorch Dataset ---

class MyCustomDataset(Dataset):
    def __init__(self, pkl_path: str, seq_len: int = 128):
        self.data = load_dataset(pkl_path)
        self.market_data = self.data["df_source"]
        self.profit_spaces = self.data["profit_spaces"]
        self.seq_len = seq_len

    def __len__(self):
        # Ensure index + seq_len fits within available data
        return len(self.market_data) - self.seq_len - 1

    def __getitem__(self, index):
        # Extract OHLC sequence (shape: (seq_len, 4))
        ohlc = self.market_data.iloc[index : index + self.seq_len][
            ["Open", "High", "Low", "Close"]
        ].to_numpy(dtype=np.float32)

        # Get corresponding ProfitSpace object for the next candle after the sequence
        space = self.profit_spaces[index + self.seq_len]

        return {
            "ohlc": ohlc,
            "space": space,
        }

# --- Create Dataset and DataLoader ---

# Path to the preprocessed dataset (.pkl file)
pkl_path = Path("./DATA/space/EURUSD-1h.pkl")
dataset = MyCustomDataset(pkl_path)

# Create DataLoader
loader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_fn)

# --- Example usage inside a training loop ---

for batch in loader:
    ohlc = batch["ohlc"]        # Tensor of shape (batch_size, seq_len, 4)
    space = batch["space"]      # List of ProfitSpace objects (batch_size)

    # # Example: Predict trade outcome using a model (pseudo-code)
    # upper_target, lower_target = get_desired_targets(ohlc)
    # confidence = buyer_model(ohlc, upper_target, lower_target)

    # # Evaluate real outcome using ProfitSpace
    # targets = [
    #     S.check_trade("buy", ut, lt)
    #     for S, ut, lt in zip(space, upper_target.cpu().numpy(), lower_target.cpu().numpy())
    # ]

    # # Calculate training loss (e.g., binary cross-entropy)
    # loss = lossFunction(confidence, targets)
```

> **📂 Note:**
For more code examples and usage demos, please refer to the `examples/` folder.

## 🤝 Contributing
Contributions are welcome and appreciated!

Feel free to fork the repository, submit pull requests, or open issues for bugs, feature suggestions, or improvements.

## 📄 License
This project is licensed under the MIT License.
See the `LICENSE.txt` file for details.