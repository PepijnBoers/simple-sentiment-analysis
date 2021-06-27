import argparse
import pickle
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, InputExample


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def data_to_input_example(text):
    return InputExample(
        guid=None,
        text_a=text,
        text_b=None,
        label=None,
    )


def make_prediction(model, tokenizer, text):
    # Tokenize input data.
    tf_batch = tokenizer(
        text, max_length=128, padding=True, truncation=True, return_tensors="tf"
    )

    # Make prediction.
    tf_outputs = model(tf_batch)
    tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)

    return [tf.argmax(pred) for pred in tf_predictions]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        help="model to use",
        default="../../models/BERT",
    )

    parser.add_argument(
        "--max-length", dest="max_length", help="max text length", default=200
    )
    parser.add_argument("-t", "--text", dest="text", help="input text")
    args = parser.parse_args()

    # Load BERT model.
    model = TFBertForSequenceClassification.from_pretrained(args.model, num_labels=2)

    # Load tokenizer.
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Make prediction.
    prediction = make_prediction(model, tokenizer, [args.text])

    # Print prediction.
    result = "POSITIVE" if prediction[0] == 1 else "NEGATIVE"
    print(f"\nThe sentiment of the text is predicted to be: {result}")
