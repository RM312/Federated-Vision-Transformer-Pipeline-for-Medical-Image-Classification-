import os
import csv
import numpy as np
import torch

from torch.optim import AdamW

from models.vit_model import get_vit_model
from models.aggregation import fedavg
from models.model_utils import (
    get_model_parameters,
    set_model_parameters,
    save_model
)

from training.train import train_one_epoch
from training.evaluate import evaluate
from training.metrics import compute_metrics


def federated_training(

        global_model,
        client_train_loaders,
        client_val_loaders,
        client_test_loaders,

        device,

        num_rounds,
        local_epochs,
        learning_rate,

        results_dir,
        roc_dir,
        csv_dir

):

    training_metrics_all = {
        "loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "mcc": [],
        "auc": [],
        "confusion_matrix": [],
        "y_true_concat": [],
        "y_scores_concat": [],
        "y_pred_concat": []
    }

    validation_metrics_all = {
        "loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "mcc": [],
        "auc": [],
        "confusion_matrix": [],
        "y_true_concat": [],
        "y_scores_concat": [],
        "y_pred_concat": []
    }

    for rnd in range(num_rounds):

        print(f"\n========== Round {rnd+1}/{num_rounds} ==========")

        global_parameters = get_model_parameters(global_model)

        client_parameters = []

        client_train_true = []
        client_train_scores = []
        client_train_pred = []

        round_train_metrics = []

        ####################################################
        # LOCAL TRAINING
        ####################################################

        for client_id, (train_loader, val_loader) in enumerate(

                zip(
                    client_train_loaders,
                    client_val_loaders
                )

        ):

            client_name = f"Client{client_id+1}"

            print(f"\nTraining {client_name}")

            local_model = get_vit_model().to(device)

            set_model_parameters(
                local_model,
                global_parameters
            )

            optimizer = AdamW(
                local_model.parameters(),
                lr=learning_rate
            )

            epoch_true = []
            epoch_scores = []
            epoch_pred = []

            last_metrics = None

            for epoch in range(local_epochs):

                metrics = train_one_epoch(
                    local_model,
                    train_loader,
                    optimizer,
                    device
                )

                print(
                    f"Epoch {epoch+1} | "
                    f"Loss={metrics['loss']:.4f} "
                    f"Acc={metrics['accuracy']:.4f} "
                    f"Prec={metrics['precision']:.4f} "
                    f"Recall={metrics['recall']:.4f} "
                    f"F1={metrics['f1_score']:.4f}"
                )

                epoch_true.append(metrics["y_true"])
                epoch_scores.append(metrics["y_scores"])
                epoch_pred.append(metrics["y_pred"])

                last_metrics = metrics

            client_true = np.concatenate(epoch_true)
            client_scores = np.concatenate(epoch_scores)
            client_pred = np.concatenate(epoch_pred)

            client_train_true.append(client_true)
            client_train_scores.append(client_scores)
            client_train_pred.append(client_pred)

            round_train_metrics.append(last_metrics)

            client_parameters.append(
                get_model_parameters(local_model)
            )

        ####################################################
        # FEDAVG
        ####################################################

        print("\nRunning Federated Averaging")

        new_global_parameters = fedavg(client_parameters)

        set_model_parameters(
            global_model,
            new_global_parameters
        )
                ####################################################
        # VALIDATION
        ####################################################

        round_val_metrics = []

        client_val_true = []
        client_val_scores = []
        client_val_pred = []

        for client_id, val_loader in enumerate(client_val_loaders):

            client_name = f"Client{client_id+1}"

            metrics = evaluate(
                global_model,
                val_loader,
                device,
            )

            print(
                f"{client_name} Validation | "
                f"Loss={metrics['loss']:.4f} "
                f"Acc={metrics['accuracy']:.4f} "
                f"Prec={metrics['precision']:.4f} "
                f"Recall={metrics['recall']:.4f} "
                f"F1={metrics['f1_score']:.4f}"
            )

            round_val_metrics.append(metrics)

            client_val_true.append(metrics["y_true"])
            client_val_scores.append(metrics["y_scores"])
            client_val_pred.append(metrics["y_pred"])

        ####################################################
        # STORE TRAINING METRICS
        ####################################################

        training_metrics_all["loss"].append(
            np.mean([m["loss"] for m in round_train_metrics])
        )

        training_metrics_all["accuracy"].append(
            np.mean([m["accuracy"] for m in round_train_metrics])
        )

        training_metrics_all["precision"].append(
            np.mean([m["precision"] for m in round_train_metrics])
        )

        training_metrics_all["recall"].append(
            np.mean([m["recall"] for m in round_train_metrics])
        )

        training_metrics_all["f1"].append(
            np.mean([m["f1_score"] for m in round_train_metrics])
        )

        training_metrics_all["mcc"].append(
            np.nanmean([m["mcc"] for m in round_train_metrics])
        )

        training_metrics_all["auc"].append(
            np.nanmean([m["auc"] for m in round_train_metrics])
        )

        ####################################################
        # STORE VALIDATION METRICS
        ####################################################

        validation_metrics_all["loss"].append(
            np.mean([m["loss"] for m in round_val_metrics])
        )

        validation_metrics_all["accuracy"].append(
            np.mean([m["accuracy"] for m in round_val_metrics])
        )

        validation_metrics_all["precision"].append(
            np.mean([m["precision"] for m in round_val_metrics])
        )

        validation_metrics_all["recall"].append(
            np.mean([m["recall"] for m in round_val_metrics])
        )

        validation_metrics_all["f1"].append(
            np.mean([m["f1_score"] for m in round_val_metrics])
        )

        validation_metrics_all["mcc"].append(
            np.nanmean([m["mcc"] for m in round_val_metrics])
        )

        validation_metrics_all["auc"].append(
            np.nanmean([m["auc"] for m in round_val_metrics])
        )

        ####################################################
        # AGGREGATED TRAIN METRICS
        ####################################################

        train_true = np.concatenate(client_train_true)
        train_scores = np.concatenate(client_train_scores)
        train_pred = np.concatenate(client_train_pred)

        training_metrics_all["y_true_concat"].append(train_true)
        training_metrics_all["y_scores_concat"].append(train_scores)
        training_metrics_all["y_pred_concat"].append(train_pred)

        train_metrics = compute_metrics(
            train_true,
            train_scores,
            train_pred
        )

        print(
            f"Round {rnd+1} Train Accuracy : "
            f"{train_metrics['accuracy']:.4f}"
        )

        ####################################################
        # AGGREGATED VALIDATION METRICS
        ####################################################

        val_true = np.concatenate(client_val_true)
        val_scores = np.concatenate(client_val_scores)
        val_pred = np.concatenate(client_val_pred)

        validation_metrics_all["y_true_concat"].append(val_true)
        validation_metrics_all["y_scores_concat"].append(val_scores)
        validation_metrics_all["y_pred_concat"].append(val_pred)

        val_metrics = compute_metrics(
            val_true,
            val_scores,
            val_pred
        )

        print(
            f"Round {rnd+1} Validation Accuracy : "
            f"{val_metrics['accuracy']:.4f}"
        )

    ########################################################
    # SAVE GLOBAL MODEL
    ########################################################

    save_model(
        global_model,
        os.path.join(
            results_dir,
            "vit_monkeypox_federated.pth"
        )
    )

    ########################################################
    # FINAL TEST
    ########################################################

    final_results = []

    for client_id, test_loader in enumerate(client_test_loaders):

        metrics = evaluate(
            global_model,
            test_loader,
            device,
        )

        final_results.append(metrics)

        print(
            f"Client {client_id+1} "
            f"Test Accuracy : {metrics['accuracy']:.4f}"
        )

    ########################################################
    # SAVE CSV
    ########################################################

    csv_file = os.path.join(
        csv_dir,
        "summary_metrics_rounds.csv"
    )

    with open(csv_file, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Round",
            "Train Loss",
            "Train Accuracy",
            "Validation Loss",
            "Validation Accuracy",
        ])

        for round_id in range(num_rounds):

            writer.writerow([
                round_id + 1,
                training_metrics_all["loss"][round_id],
                training_metrics_all["accuracy"][round_id],
                validation_metrics_all["loss"][round_id],
                validation_metrics_all["accuracy"][round_id],
            ])

    print("\nTraining Complete.")

    return (
        global_model,
        training_metrics_all,
        validation_metrics_all,
        final_results
    )