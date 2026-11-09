import pandas as pd
from sklearn.model_selection import train_test_split

def transform(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    test_df["split"] = "test"
    return pd.concat([train_df, test_df], ignore_index=True)
