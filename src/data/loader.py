import pandas as pd

from src.config import DATA_PATH


def load_path(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    return df