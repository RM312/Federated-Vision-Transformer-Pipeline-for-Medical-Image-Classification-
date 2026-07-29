import torch


def fedavg(client_parameters):
    """
    Federated Averaging
    """

    averaged_parameters = {}

    with torch.no_grad():

        for key in client_parameters[0].keys():

            averaged_parameters[key] = torch.stack(
                [client[key] for client in client_parameters],
                dim=0
            ).mean(dim=0)

    return averaged_parameters