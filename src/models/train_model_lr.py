import argparse
import pickle

from src.data.utils import load_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

tqdm.pandas()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="input data (raw)")
    parser.add_argument(
        "-m",
        "--max-iter",
        type=int,
        dest="max_iter",
        help="maximum number of iterations to converge",
        default=10000,
    )
    parser.add_argument(
        "-c",
        dest="c",
        type=float,
        help="inverse of regularization strength",
        default=1.0,
    )
    args = parser.parse_args()

    # Load and split data 70/30 train/validation.
    X, y = load_data(args.input)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.30, random_state=99, stratify=y
    )

    # Convert text to sparse matrix (all terms).
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    # Fit Logistic Regression.
    log_regression = LogisticRegression(max_iter=args.max_iter, C=args.c)
    log_regression.fit(X_train_vec, y_train)

    # Print accuracy.
    accuracy = accuracy_score(y_val, log_regression.predict(X_val_vec))
    print(
        f"Logistic regression model obtained an accuracy of {accuracy*100:.1f}% with parameter c = {args.c}."
    )

    # Store vectorizer.
    with open(f"../../models/count_vectorizer", "wb") as f:
        pickle.dump(vectorizer, f)
        print(f"Vectorizer saved at: models/count_vectorizer")

    # Store model.
    with open(f"../../models/logistic_regression_model_{args.c}", "wb") as f:
        pickle.dump(log_regression, f)
        print(f"Model saved at: models/logistic_regression_model_{args.c}")

    # Example usage: python train_model_lr.py -i <processed_data.csv> -m 10000 -c 1
