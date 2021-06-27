import argparse
import pickle
import numpy as np

import pandas as pd
from src.features.preprocess import preprocess
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow import keras

from src.models.train_model_keras import tokenize_and_pad


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def predictions_to_labels(predictions):
    return [1 if x[0] >= 0.5 else 0 for x in predictions]


def make_prediction(model, data, tokenizer, max_length):
    preprocessed_text = tokenize_and_pad(tokenizer, data, max_length)
    predictions = model.predict(preprocessed_text)
    return predictions_to_labels(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        help="model to use",
        default="../../models/keras_20210625-212543_5000_200_lstm.h5",
    )
    parser.add_argument(
        "-v",
        "--tokenizer",
        dest="tokenizer",
        help="tokenizer to use",
        default="../../models/tokenizer.pickle",
    )
    parser.add_argument(
        "--max-length", dest="max_length", help="max text length", default=200
    )
    parser.add_argument("-t", "--text", dest="text", help="input text")
    args = parser.parse_args()

    # Load model.
    model = keras.models.load_model(args.model)

    # Load tokenizer.
    tokenizer = load_pickle(args.tokenizer)

    # Create Pandas Series.
    text_series = pd.Series([args.text], dtype="string")

    # Make prediction.
    prediction = make_prediction(model, text_series, tokenizer, args.max_length)

    # Print prediction.
    result = "POSITIVE" if np.mean(prediction) > 0.5 else "NEGATIVE"
    print(f"\nThe sentiment of the text is predicted to be: {result}")
