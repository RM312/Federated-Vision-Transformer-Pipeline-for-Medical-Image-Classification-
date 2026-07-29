import torch


class ViTHook:

    def __init__(self, model):

        self.model = model

        self.activations = None
        self.gradients = None

        self.forward_handle = None
        self.backward_handle = None

    def _forward_hook(self, module, inputs, outputs):

        self.activations = outputs

    def _backward_hook(self, module, grad_input, grad_output):

        self.gradients = grad_output[0]

    def register(self):

        target_layer = self.model.vit.encoder.layer[-1].output

        self.forward_handle = target_layer.register_forward_hook(
            self._forward_hook
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def remove(self):

        if self.forward_handle:
            self.forward_handle.remove()

        if self.backward_handle:
            self.backward_handle.remove()