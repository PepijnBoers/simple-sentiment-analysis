import itertools

import numpy as np
from matplotlib import pyplot as plt

PRETTY_NAME_DICT = {"acc": "Accuracy", "loss": "Loss"}
LABELS = {0: "NEGATIVE", 1: "POSITIVE"}


def plot_history(history, name):
    """Simple model history plot."""
    # Create subplot.
    fig, axs = plt.subplots(2, 1, figsize=(15, 6), facecolor="w", edgecolor="k")
    axs = axs.ravel()

    # Plot accuracy and loss.
    for i, kind in enumerate(["acc", "loss"]):
        axs[i].plot(history.history[kind])
        axs[i].plot(history.history[f"val_{kind}"])
        axs[i].set_title(f"{PRETTY_NAME_DICT[kind]} history")
        axs[i].set_ylabel(PRETTY_NAME_DICT[kind])

        if kind == "loss":
            axs[i].set_xlabel("Epoch")

    # Add legend.
    fig.legend(["Train", "Val"], loc="lower right", ncol=2)

    # Save in pdf format.
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{name}.pdf")


def plot_confusion_matrix(cm, title, labels, save=False):
    """Plots confusion matrix and shows precision, recall and f1-score.

    Args:
        cm: sklearn confusion matrix
        title: plot title
        labels: list with labels
        save: save plot as pdf

    modified from: https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    """
    # Extract positives & negatives info.
    tn, fp, fn, tp = cm.ravel()

    # Calculate precision, recall and f1-score.
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    accuracy = np.trace(cm) / float(np.sum(cm))

    # Plot confusion matrix.
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.get_cmap("Blues"))
    plt.colorbar()
    plt.title(title)

    # Plot labels x and y axis.
    if labels is not None:
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)

    # Print normalized percentages.
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm_normalized.max() / 1.5
    for i, j in itertools.product(
        range(cm_normalized.shape[0]), range(cm_normalized.shape[1])
    ):
        plt.text(
            j,
            i,
            f"{cm_normalized[i, j]*100:.2f}%",
            horizontalalignment="center",
            weight="bold",
            size=24,
            color="whitesmoke" if cm_normalized[i, j] > thresh else "slategrey",
        )

    # Add labels and plot
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel(
        f"Predicted label\naccuracy={accuracy:0.4f}; f1-score={f1_score:0.4f}\nprecision={precision:0.4f}; recall={recall:0.4f}"
    )

    # Save plot
    if save:
        plt.savefig(f"./../reports/figures/{title}.png", bbox_inches = 'tight')

    plt.show()

    


def show_mistakes(data, indices, y_pred, nr_examples):
    """Prints reviews that were predicted incorrectly.

    Args:
        data: pandas dataframe containing reviews
        indices: indices of incorrect predictions
        y_pred: predicted labels
        nr_examples: nr of examples to show
    """
    for idx in indices[:nr_examples]:
        print(
            f"(not {LABELS[y_pred[idx[0]]]}):"
            + data.iloc[idx].to_string(index=False)
            + "\n"
        )
