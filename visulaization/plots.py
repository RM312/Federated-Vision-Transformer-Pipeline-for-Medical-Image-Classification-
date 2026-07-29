import os
import matplotlib.pyplot as plt


def plot_metric(
    train_values,
    val_values,
    metric_name,
    save_dir
):
    """
    Plot training vs validation metric.
    """

    plt.figure(figsize=(8, 5))

    plt.plot(
        train_values,
        marker="o",
        linewidth=2,
        label="Train"
    )

    plt.plot(
        val_values,
        marker="s",
        linewidth=2,
        label="Validation"
    )

    plt.xlabel("Federated Round")
    plt.ylabel(metric_name)

    plt.title(metric_name)

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(
        os.path.join(
            save_dir,
            f"{metric_name.lower().replace(' ','_')}.png"
        )
    )

    plt.close()


def plot_all_metrics(
    train_metrics,
    validation_metrics,
    save_dir,
):

    metric_list = [
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "auc"
    ]

    for metric in metric_list:

        plot_metric(
            train_metrics[metric],
            validation_metrics[metric],
            metric.capitalize(),
            save_dir
        )