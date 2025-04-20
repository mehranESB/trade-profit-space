import pandas as pd
import numpy as np


class SpaceMaker:
    def __init__(self, market_data: pd.DataFrame, max_holds: int = 15):
        """
        Initializes the SpaceMaker instance.

        Parameters:
        - market_data (pd.DataFrame): Market OHLC data used to compute trade results.
        - max_holds (int): Maximum number of bars to hold a position. Must be at least 1.
        """

        # Ensure required OHLC columns are present
        required_columns = {"Open", "High", "Low", "Close"}
        if not required_columns.issubset(market_data.columns):
            raise ValueError(f"market_data must contain columns: {required_columns}")

        # Market data is OHLC data that needs to be used to find trade results
        self.op = market_data["Open"].to_numpy(dtype=np.float32)
        self.hi = market_data["High"].to_numpy(dtype=np.float32)
        self.lo = market_data["Low"].to_numpy(dtype=np.float32)
        self.cl = market_data["Close"].to_numpy(dtype=np.float32)

        # max_holds means number of bars to hold the position
        self.max_holds = max(max_holds, 1)

        # Number of bars in the market data
        self.nBars = len(market_data)

    def __len__(self):
        return self.nBars

    def __getitem__(self, index):
        # Ensure the index is within the valid range
        if not (0 <= index < self.nBars):
            raise IndexError(
                f"Index {index} is out of bounds for market data with {self.nBars} bars."
            )

        # Determine end of slice, limited by data length
        to_index = min(index + self.max_holds, self.nBars)

        # Slice the OHLC data
        op = self.op[index:to_index]
        hi = self.hi[index:to_index]
        lo = self.lo[index:to_index]
        cl = self.cl[index:to_index]

        return ProfitSpace(op, hi, lo, cl)


class ProfitSpace:
    pass
