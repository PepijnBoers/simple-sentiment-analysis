import argparse
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")

from tqdm import tqdm

tqdm.pandas()


def clean_data(data):
    """From TWRDS-DS post by Jodie Zhou"""
    # Removing URLs with a regular expression.
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    data = url_pattern.sub(r"", data)

    # Remove Emails.
    data = re.sub("\S*@\S*\s?", "", data)

    # Remove new line characters.
    data = re.sub("\s+", " ", data)

    # Remove distracting single quotes.
    data = re.sub("'", "", data)

    return data


def preprocess(df, ps, stop_words):
    # Combine review title and body.
    df["review"] = df["title"] + df["body"]

    # clean data.
    df["review"] = clean_data(df["review"])

    # Lowercase text.
    df["review"] = df["review"].lower()

    # Tokenize review body.
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    df["review"] = tokenizer.tokenize(df["review"])

    # Remove stopwords and stem remaining words.
    df["review"] = " ".join(
        [
            ps.stem(w)
            for sentence in df["review"]
            for w in word_tokenize(sentence)
            if not w in stop_words
        ]
    )

    return df


def apply_preprocessing(in_path, out_path):
    # Load stopwords and stemmer.
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()

    # Load raw data file.
    train_df = pd.read_csv(in_path)

    # Apply preprocessing.
    processed_df = train_df.progress_apply(preprocess, args=(ps, stop_words), axis=1)

    # Delete separate title & body columns.
    del processed_df["title"]
    del processed_df["body"]

    # Store preprocessed data in csv.
    processed_df.to_csv(f"{out_path}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="input data (raw)")
    parser.add_argument("-o", "--output", dest="output", help="output data to csv")
    args = parser.parse_args()

    apply_preprocessing(args.input, args.output)
