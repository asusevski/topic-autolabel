from pathlib import Path
from typing import Optional, Union

import pandas as pd


def load_data(
    filepath: Union[str, Path], text_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Load data from a CSV file and validate the text column exists.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"No file found at {filepath}")

    df = pd.read_csv(filepath)

    if text_column and text_column not in df.columns:
        raise ValueError(f"Column {text_column} not found in the CSV file")

    return df
