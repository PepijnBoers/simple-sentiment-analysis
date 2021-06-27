import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

pd.options.display.max_colwidth = 250


def load_data(path):
    df = pd.read_csv(path)
    return df["review"], df["label"]


def load_train_test(generic_path):
    train_df = pd.read_csv(f"{generic_path}_train.csv")
    test_df = pd.read_csv(f"{generic_path}_test.csv")
    return train_df, test_df


def get_train_test_split(data_path):
    """Loads and splits data 70/30 train/validation."""
    X, y = load_data(data_path)
    return train_test_split(X, y, test_size=0.30, random_state=99, stratify=y)


def find_mistakes(y_true, y_pred):
    return np.argwhere(y_true != y_pred)
