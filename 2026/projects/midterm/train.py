import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path="data/water_potability.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    return df


def split_data(df, seed=1):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["potability"])
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=seed, stratify=df_full_train["potability"])
    return [d.reset_index(drop=True) for d in (df_full_train, df_train, df_val, df_test)]


def add_features(df):
    df = df.copy()
    df["ph_dist"] = abs(df["ph"] - 7)
    return df