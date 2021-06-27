import argparse
from datetime import datetime

import pickle
import nltk
from src.data.utils import load_data
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras import layers

from sklearn.model_selection import train_test_split
from src.visualization.visualize import plot_history

nltk.download("punkt")
nltk.download("stopwords")

from tqdm import tqdm

tqdm.pandas()


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def tokenize_and_pad(tokenizer, data_df, max_length):
    """Vectorizes text corpus and pads review to max length."""
    sequences = tokenizer.texts_to_sequences(data_df)
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    return padded_sequences


def create_lstm_model(max_words):
    # Model architecture.
    model = Sequential()
    model.add(layers.Embedding(max_words, 128))
    model.add(layers.SpatialDropout1D(0.25))
    model.add(layers.LSTM(196, return_sequences=True, dropout=0.5))
    model.add(layers.LSTM(256, return_sequences=True, dropout=0.5))
    model.add(layers.LSTM(128, dropout=0.5))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


def create_cnn_model(max_words):
    # Model architecture.
    model = Sequential()
    model.add(layers.Embedding(max_words, 128))

    # Convolutional block 1.
    model.add(layers.Conv1D(16, 6, activation="relu"))
    model.add(layers.Conv1D(16, 6, activation="relu"))
    model.add(layers.MaxPooling1D(5))

    # COnvolutional block 2.
    model.add(layers.Conv1D(16, 6, activation="relu"))
    model.add(layers.Conv1D(16, 6, activation="relu"))
    model.add(layers.GlobalMaxPooling1D())

    # Dense layers.
    model.add(layers.Dense(10, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


def train_model(model, X, y, X_val, y_val, epochs=10, name="default_name"):
    # Define model name.
    model_name = f"keras_{name}.h5"

    # Define early stopping criteria (NOT USED).
    early_stop = EarlyStopping(
        monitor="val_acc", min_delta=0.001, patience=10, mode="max", verbose=1
    )

    # Define model checkpoint criteria.
    checkpoint = ModelCheckpoint(
        f"../../models/{model_name}",
        monitor="val_acc",
        verbose=1,
        save_best_only=True,
        mode="max",
        save_freq="epoch",
    )

    # Use plotlosses library for live plotting.
    # plot_losses = PlotLossesCallback(outputs=[BokehPlot()])

    # Fit model and save history.
    history = model.fit(
        X,
        y,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[checkpoint],
        shuffle=True,
        verbose=2,
    )

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="input data (raw)")
    parser.add_argument("-m", "--model", dest="model", help="model: lstm or cnn")
    parser.add_argument(
        "--max-words",
        dest="max_words",
        type=int,
        help="maximum number of words in dict",
    )
    parser.add_argument(
        "--max-length", dest="max_length", type=int, help="maximum review length"
    )
    parser.add_argument(
        "-e", "--epochs", dest="epochs", type=int, help="number of epochs"
    )

    args = parser.parse_args()

    # Define name.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}_{args.max_words}_{args.max_length}_{args.model}"

    # Load and split data 70/30 train/validation.
    X, y = load_data(args.input)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.30, random_state=99, stratify=y
    )

    # Tokenize data.
    tokenizer = Tokenizer(num_words=args.max_words)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenize_and_pad(tokenizer, X_train, args.max_length)
    X_val = tokenize_and_pad(tokenizer, X_val, args.max_length)

    # Save tokenizer for prediction.
    with open("../../models/tokenizer.pickle", "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Build model.
    if args.model == "cnn":
        model = create_cnn_model(args.max_words)
    else:
        model = create_lstm_model(args.max_words)

    # Compile model.
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

    # Train model.
    history = train_model(model, X_train, y_train, X_val, y_val, args.epochs, run_name)

    # Store accuracy and loss history in plot.
    plot_history(history, run_name)
