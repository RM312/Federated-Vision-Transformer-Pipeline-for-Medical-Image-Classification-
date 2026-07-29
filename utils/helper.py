import torch

def get_model_parameters(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def set_model_parameters(model, parameters):
    model.load_state_dict(parameters, strict=True)


def move_to_device(model, device):
    model.to(device)
    return model