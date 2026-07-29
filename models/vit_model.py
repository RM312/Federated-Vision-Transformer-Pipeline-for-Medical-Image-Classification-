import torch
from transformers import ViTForImageClassification

MODEL_NAME = "google/vit-base-patch16-224-in21k"


def get_vit_model(num_labels=2, image_size=64):
    """
    Create Vision Transformer model.
    """

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        image_size=image_size,
        ignore_mismatched_sizes=True,
    )

    return model