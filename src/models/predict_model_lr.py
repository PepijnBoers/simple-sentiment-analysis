import argparse
import pickle

import pandas as pd
from src.features.preprocess import preprocess
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def make_prediction(model, df, cv):
    preprocessed_text = preprocess(df, ps, stop_words)
    vectorized_text = cv.transform([preprocessed_text["body"]])
    return model.predict(vectorized_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        help="model to use",
        default="../../models/logistic_regression_model_1.0",
    )
    parser.add_argument(
        "-v",
        "--vectorizer",
        dest="vectorizer",
        help="vectorizer to use",
        default="../../models/count_vectorizer",
    )
    parser.add_argument("-t", "--text", dest="text", help="input text")
    args = parser.parse_args()

    # Initialize stop_words and stemmer.
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()

    # Load model.
    model = load_pickle(args.model)

    # Load Vectorizer.
    vectorizer = load_pickle(args.vectorizer)

    # Create Pandas Series.
    text_series = pd.Series({"title": "", "body": args.text})

    # Make prediction.
    prediction = make_prediction(model, text_series, vectorizer)

    # Print prediction.
    result = "POSITIVE" if prediction == 1 else "NEGATIVE"
    print(f"\nThe sentiment of the text is predicted to be: {result}")
