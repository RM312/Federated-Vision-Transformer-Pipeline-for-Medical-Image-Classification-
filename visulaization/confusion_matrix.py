import os

import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


def save_confusion_matrix(
    confusion_matrix,
    class_names,
    save_path
):

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=class_names
    )

    disp.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()