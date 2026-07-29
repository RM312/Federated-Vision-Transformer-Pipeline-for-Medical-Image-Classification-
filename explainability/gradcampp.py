import cv2
import numpy as np
import torch

from .hooks import ViTHook


class GradCAMPlusPlus:

    def __init__(self, model):

        self.model = model

        self.hook = ViTHook(model)

        self.hook.register()

    def generate(self, image_tensor):

        self.model.zero_grad()

        output = self.model(image_tensor)

        prediction = output.logits.argmax(dim=1)

        score = output.logits[:, prediction]

        score.backward(retain_graph=True)

        activations = self.hook.activations

        gradients = self.hook.gradients

        alpha = gradients.pow(2)

        alpha = alpha / (
            2 * alpha +
            (
                activations *
                gradients.pow(3)
            ).sum(dim=(2, 3), keepdim=True)
            + 1e-8
        )

        weights = (
            alpha *
            torch.relu(gradients)
        ).sum(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1)

        cam = torch.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()

        cam = cv2.resize(cam, (224, 224))

        cam = (
            cam - cam.min()
        ) / (
            cam.max() - cam.min() + 1e-8
        )

        return cam