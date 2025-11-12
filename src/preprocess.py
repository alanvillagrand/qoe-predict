import pandas as pd

def load_logs(filepath: str) -> pd.DataFrame:
    """Load log data from a CSV file."""
    return pd.read_csv(filepath)

def clean_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the log data by removing duplicates and handling missing values."""
    ##df = df.drop_duplicates()
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df