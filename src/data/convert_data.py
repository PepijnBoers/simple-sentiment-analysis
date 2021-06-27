import argparse
from os import sep

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


def parse_data(df):
    """ """
    # Extract sentiment label.
    df["label"] = df["unfiltered_review"].split()[0]

    # Extract review title.
    df["title"] = df["unfiltered_review"].split(":", 1)[0].split(" ", 1)[1]

    # Extract review body.
    df["body"] = df["unfiltered_review"].split(":", 1)[1]

    return df


def store_data(df, out_path, separate_test):
    if separate_test:
        train_size = round(len(df) * 0.8)
        df = df.sample(frac=1)

        df.iloc[:train_size].to_csv(f"{out_path}_train.csv", index=False)
        df.iloc[train_size:].to_csv(f"{out_path}_test.csv", index=False)
    else:
        df.to_csv(f"{out_path}.csv", index=False)


def convert(in_path, out_path, separate_test):
    # Load raw data file.
    train_df = pd.read_csv(in_path, delimiter="\n", names=["unfiltered_review"])

    # Store review title, body and label in separate columns.
    train_df = train_df.progress_apply(parse_data, axis=1)

    # Convert string label to integer.
    train_df["label"] = np.where(train_df["label"] == "__label__1", 0, 1)

    # Delete unfiltered (raw) review textfield.
    del train_df["unfiltered_review"]

    # Store columnized data in csv.
    store_data(train_df, out_path, separate_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="input data (raw)")
    parser.add_argument("-o", "--output", dest="output", help="output data to csv")
    parser.add_argument("--separate-test", dest="sep_test", action="store_true")
    args = parser.parse_args()

    convert(args.input, args.output, args.sep_test)

    # Example usage: python convert_data.py -i <path-to-raw-data> -o <path-to-store-csv>
