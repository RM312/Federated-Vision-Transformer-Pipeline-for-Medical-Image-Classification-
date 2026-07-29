import os

import cv2

import numpy as np

import matplotlib.pyplot as plt


def save_heatmap(

        original_image,

        cam,

        save_path,

        alpha=0.4

):

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    heatmap = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    original = np.array(original_image)

    original = cv2.resize(
        original,
        (224, 224)
    )

    overlay = cv2.addWeighted(
        original,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    plt.figure(figsize=(8, 8))

    plt.imshow(
        cv2.cvtColor(
            overlay,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()