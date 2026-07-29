import torch
import numpy as np

from tqdm.auto import tqdm

from .metrics import compute_metrics


criterion = torch.nn.CrossEntropyLoss()


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
):

    model.train()

    total_loss = 0
    total_samples = 0

    y_true = []
    y_pred = []
    y_scores = []

    loop = tqdm(loader, leave=False)

    for images, labels in loop:

        images = images.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()

        outputs = model(images)

        logits = outputs.logits

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        probabilities = torch.softmax(
            logits,
            dim=1
        )[:, 1]

        predictions = torch.argmax(
            logits,
            dim=1
        )

        y_scores.extend(
            probabilities.detach().cpu().numpy()
        )

        y_pred.extend(
            predictions.detach().cpu().numpy()
        )

        y_true.extend(
            labels.detach().cpu().numpy()
        )

    metrics = compute_metrics(
        y_true,
        y_scores,
        y_pred
    )

    metrics["loss"] = total_loss / total_samples

    metrics["y_true"] = np.array(y_true)
    metrics["y_scores"] = np.array(y_scores)
    metrics["y_pred"] = np.array(y_pred)

    return metrics