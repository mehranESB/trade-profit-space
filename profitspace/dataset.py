from .space import SpaceMaker
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm


def save_dataset(market_data: pd.DataFrame, save_path: str, max_holds: int = 15):
    """
    Creates and saves a dataset containing ProfitSpace objects for each bar in the market data.

    Parameters:
    - market_data (pd.DataFrame): DataFrame containing OHLC data with columns ["Open", "High", "Low", "Close"]
    - save_path (str): Path to save the .pkl file
    - max_holds (int): Maximum number of bars to simulate trade holding
    """
    save_path = Path(save_path)

    if save_path.suffix != ".pkl":
        raise ValueError("save_path must end with '.pkl'")

    # Create parent directories if they don't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Create SpaceMaker object from market data
    smaker = SpaceMaker(market_data, max_holds)

    # Generate a ProfitSpace object for each candle/bar
    profit_spaces = [
        smaker[i] for i in tqdm(range(len(market_data)), desc="Creating ProfitSpaces")
    ]

    # Package everything into a dictionary
    packed_data = {
        "df_source": market_data,
        "profit_spaces": profit_spaces,
        "max_holds": max_holds,
    }

    # Save to pickle
    with open(save_path, "wb") as file:
        pickle.dump(packed_data, file)
    print(f"Dataset successfully saved to {save_path}")


def load_dataset(save_path: str):
    """
    Loads a dataset containing market data and ProfitSpace objects from a .pkl file.

    Parameters:
    - save_path (str): Path to the .pkl file to load.

    Returns:
    - dict: A dictionary containing:
        - 'df_source' (pd.DataFrame): The original market data.
        - 'profit_spaces' (list[ProfitSpace]): List of ProfitSpace objects for each bar.
    """
    save_path = Path(save_path)

    if not save_path.exists():
        raise FileNotFoundError(f"File not found: {save_path}")
    if save_path.suffix != ".pkl":
        raise ValueError("save_path must end with '.pkl'")

    with open(save_path, "rb") as file:
        data = pickle.load(file)

    return data
