from profitspace.dataset import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
import numpy as np


def custom_collate_fn(batch):
    ohlc = [item["ohlc"] for item in batch]
    spaces = [item["space"] for item in batch]
    return {
        "ohlc": torch.tensor(ohlc),
        "space": spaces,
    }  # Keep spaces as list of objects


class MyCustomDataset(Dataset):
    def __init__(self, pkl_path: str, seq_len: int = 128):
        self.data = load_dataset(pkl_path)
        self.market_data = self.data["df_source"]
        self.profit_spaces = self.data["profit_spaces"]
        self.seq_len = seq_len

    def __len__(self):
        # Ensure index + seq_len < len(market_data) and spaces[index + seq_len] is accessible
        return len(self.market_data) - self.seq_len - 1

    def __getitem__(self, index):
        # Extract OHLC numpy array
        ohlc = self.market_data.iloc[index : index + self.seq_len][
            ["Open", "High", "Low", "Close"]
        ].to_numpy(dtype=np.float32)

        # ProfitSpace object (future candle)
        space = self.profit_spaces[index + self.seq_len]

        return {
            "ohlc": ohlc,  # shape: (seq_len, 4)
            "space": space,  # ProfitSpace object
        }


# create dataset object
pkl_path = Path("./DATA/space/EURUSD-1h.pkl")
dataset = MyCustomDataset(pkl_path)

# create torch data loader
loader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_fn)

# retrieve batch of data
for batch in loader:
    ohlc, space = batch["ohlc"], batch["space"]

    # # Example usage as per your commented code
    # upper_target, lower_target = get_desired_targets(ohlc)  # Get upper and lower targets based on the OHLC data

    # # Compute the confidence score from the buyer model
    # confidence = buyer_model(ohlc, upper_target, lower_target)

    # # Use the "space" (ProfitSpace object) to check if a trade would be successful
    # targets = [S.check_trade("buy", ut, lt) for S, ut, lt in zip(space, upper_target.cpu().numpy(), lower_target.cpu().numpy())]

    # # Calculate the loss between the confidence and the targets (this could be any loss function, such as MSE or cross-entropy)
    # loss = lossFunction(confidence, targets)
