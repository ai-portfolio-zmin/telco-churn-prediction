import pandas as pd
from data_loader import load_data
from config import  RAW_DATA_PATH, CLEANED_DATA_PATH, TARGET_COL

def clean_raw_data(df: pd.DataFrame):
    df = df.copy()

    df.columns = df.columns.str.strip()

    if 'customerID' in df.columns:
        df = df.drop('customerID',axis=1)

    if 'TotalCharges' in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Simple strategy: fill missing TotalCharges with median
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if TARGET_COL in df.columns:
        if df[TARGET_COL].dtype =='object':
            df[TARGET_COL] = df[TARGET_COL].map(lambda x:1 if x =='Yes' else 0)

    return df

def make_cleaned_dataset(
        raw_path = RAW_DATA_PATH,
        cleaned_path = CLEANED_DATA_PATH
):
    df = load_data(raw_path)
    cleaned_df = clean_raw_data(df)
    cleaned_df.to_csv(cleaned_path)
    return cleaned_df

if __name__ == '__main__':
    d = make_cleaned_dataset()