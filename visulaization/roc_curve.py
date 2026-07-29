import os

import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import roc_curve


def save_roc_curve(
    y_true,
    y_scores,
    save_path,
    title="ROC Curve"
):

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    try:

        if len(np.unique(y_true)) > 1:

            fpr, tpr, _ = roc_curve(
                y_true,
                y_scores
            )

            plt.figure(figsize=(6, 5))

            plt.plot(
                fpr,
                tpr,
                linewidth=2,
                label="ROC"
            )

            plt.plot(
                [0, 1],
                [0, 1],
                "--"
            )

            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")

            plt.title(title)

            plt.grid(True)

            plt.legend()

        else:

            plt.figure(figsize=(5, 4))

            plt.text(
                0.5,
                0.5,
                "ROC unavailable",
                ha="center",
                va="center",
                fontsize=14
            )

            plt.axis("off")

        plt.tight_layout()

        plt.savefig(save_path)

        plt.close()

    except Exception:

        plt.figure(figsize=(5, 4))

        plt.text(
            0.5,
            0.5,
            "ROC generation failed",
            ha="center",
            va="center"
        )

        plt.axis("off")

        plt.savefig(save_path)

        plt.close()