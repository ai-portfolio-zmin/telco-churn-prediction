import pandas as pd
from config import RAW_DATA_PATH, TARGET_COL, CLEANED_DATA_PATH
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    return df

def load_cleaned_data(path = CLEANED_DATA_PATH):
    df = pd.read_csv(path)
    return df

if __name__ == '__main__':
    load_data()