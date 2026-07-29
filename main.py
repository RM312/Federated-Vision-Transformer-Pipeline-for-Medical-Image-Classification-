import os
import torch

from config import (
    DEVICE,
    NUM_GLOBAL_ROUNDS,
    LOCAL_EPOCHS,
    LEARNING_RATE,
    RESULTS_DIR,
    PLOTS_DIR,
    ROC_DIR,
    CM_DIR,
    CSV_DIR,
    XAI_DIR
)

from utils.seed import set_seed

from models.vit_model import get_vit_model

from training.federated import federated_training

from visualization.plots import plot_all_metrics
from visualization.confusion_matrix import save_confusion_matrix
from visualization.roc_curve import save_roc_curve


####################################################
# IMPORT YOUR DATALOADERS HERE
####################################################
#
# Example:
#
# from dataset import (
#     client_train_loaders,
#     client_val_loaders,
#     client_test_loaders
# )
#
####################################################


def main():

    ####################################################
    # Reproducibility
    ####################################################

    set_seed(42)

    print(f"Using Device : {DEVICE}")

    ####################################################
    # Global Model
    ####################################################

    global_model = get_vit_model().to(DEVICE)

    ####################################################
    # Federated Training
    ####################################################

    (
        global_model,
        train_metrics,
        validation_metrics,
        test_results
    ) = federated_training(

        global_model=global_model,

        client_train_loaders=client_train_loaders,
        client_val_loaders=client_val_loaders,
        client_test_loaders=client_test_loaders,

        device=DEVICE,

        num_rounds=NUM_GLOBAL_ROUNDS,

        local_epochs=LOCAL_EPOCHS,

        learning_rate=LEARNING_RATE,

        results_dir=RESULTS_DIR,
        roc_dir=ROC_DIR,
        csv_dir=CSV_DIR
    )

    ####################################################
    # Training Curves
    ####################################################

    plot_all_metrics(
        train_metrics,
        validation_metrics,
        PLOTS_DIR
    )

    ####################################################
    # Final Test Results
    ####################################################

    for idx, metrics in enumerate(test_results):

        print("\n----------------------------")

        print(f"Client {idx+1}")

        print("----------------------------")

        print(f"Accuracy : {metrics['accuracy']:.4f}")

        print(f"Precision : {metrics['precision']:.4f}")

        print(f"Recall : {metrics['recall']:.4f}")

        print(f"F1 Score : {metrics['f1_score']:.4f}")

        print(f"MCC : {metrics['mcc']:.4f}")

        print(f"AUC : {metrics['auc']:.4f}")

        save_confusion_matrix(

            metrics["confusion_matrix"],

            class_names=["Negative", "Positive"],

            save_path=os.path.join(
                CM_DIR,
                f"client_{idx+1}.png"
            )
        )

        save_roc_curve(

            metrics["y_true"],

            metrics["y_scores"],

            save_path=os.path.join(
                ROC_DIR,
                f"client_{idx+1}.png"
            )
        )

    ####################################################
    # Finished
    ####################################################

    print("\n===================================")

    print("Federated Learning Completed")

    print("===================================")


if __name__ == "__main__":

    main()