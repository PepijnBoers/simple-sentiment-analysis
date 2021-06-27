import argparse
from datetime import datetime

import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from transformers import (
    BertTokenizer,
    InputExample,
    InputFeatures,
    TFBertForSequenceClassification,
)
from src.visualization.visualize import plot_history

TF_OUTPUT_TYPES = (
    {"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32},
    tf.int64,
)
TF_OUTPUT_SHAPES = (
    {
        "input_ids": tf.TensorShape([None]),
        "attention_mask": tf.TensorShape([None]),
        "token_type_ids": tf.TensorShape([None]),
    },
    tf.TensorShape([]),
)


def data_to_input_example(df):
    return InputExample(
        guid=None,
        text_a=df["body"],
        text_b=None,
        label=df["label"],
    )


def examples_to_input_features(examples, tokenizer):
    input_features = []

    for example in examples:
        example_dict = tokenizer.encode_plus(
            example.text_a,
            padding="max_length",
            max_length=128,
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            truncation=True,
        )
        input_feature = InputFeatures(
            example_dict["input_ids"],
            example_dict["attention_mask"],
            example_dict["token_type_ids"],
            example.label,
        )
        input_features.append(input_feature)

    def tf_data_generator():
        for feature in input_features:
            yield (
                {
                    "input_ids": feature.input_ids,
                    "attention_mask": feature.attention_mask,
                    "token_type_ids": feature.token_type_ids,
                },
                feature.label,
            )

    return tf.data.Dataset.from_generator(
        tf_data_generator, TF_OUTPUT_TYPES, TF_OUTPUT_SHAPES
    )


def train_model(data_df, run_name, epochs):
    # Split train/validation.
    train_df = data_df.iloc[: round(data_size * 0.8)]
    val_df = data_df.iloc[round(data_size * 0.8) :]

    # Convert train data.
    train_examples = train_df.apply(data_to_input_example, axis=1)
    train_data = examples_to_input_features(list(train_examples), tokenizer)
    train_data = train_data.shuffle(data_size).batch(32)

    # Convert validation data.
    val_examples = val_df.apply(data_to_input_example, axis=1)
    val_data = examples_to_input_features(list(val_examples), tokenizer)
    val_data = val_data.shuffle(data_size).batch(32)

    # Adam optimizer.
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0
    )

    # (Sparse) Categorical cross entropy loss.
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # (Sparse) Categorical accuracy.
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy("acc")]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Define model checkpoint criteria.
    model_name = f"../../models/BERT_model_{run_name}.h5"
    checkpoint = ModelCheckpoint(
        f"../../models/{model_name}",
        monitor="val_acc",
        verbose=1,
        save_best_only=True,
        mode="max",
        save_freq=1,
    )

    history = model.fit(
        train_data,
        validation_data=val_data,
        callbacks=[checkpoint],
        epochs=epochs,
        shuffle=True,
        verbose=1,
    )

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="input data (raw)")
    parser.add_argument(
        "-e", "--epochs", dest="epochs", type=int, help="number of epochs"
    )

    args = parser.parse_args()

    # Define name.
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Load pre-trained BERT model and tokenizer with huggingface's transformers.
    model = TFBertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load data.
    data_df = pd.read_csv(args.input)
    data_size = len(data_df)

    # Fit model.
    history = train_model(data_df, run_name, args.epochs)

    # Save model
    model.save_pretrained("../../models/BERT")

    # Store accuracy and loss history in plot.
    plot_history(history, f"{run_name}_BERT")
