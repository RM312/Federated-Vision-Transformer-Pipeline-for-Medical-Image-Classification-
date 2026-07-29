import torch


def get_model_parameters(model):
    """
    Return model parameters as dictionary.
    """

    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def set_model_parameters(model, parameters):
    """
    Load model parameters.
    """

    model.load_state_dict(parameters, strict=True)


def save_model(model, path):
    """
    Save model.
    """

    torch.save(model.state_dict(), path)


def load_model(model, path, device="cpu"):
    """
    Load model.
    """

    model.load_state_dict(
        torch.load(path, map_location=device)
    )

    return model