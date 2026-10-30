import matplotlib.pyplot as plt

def plot_fig(
    xlabel,
    ylabel,
    title,
    savefig,
    train_data=None,
    val_data=None,
):
    if train_data is None and val_data is None:
        raise ValueError("Provide at least one dataset to plot.")

    plt.figure(figsize=(8, 5))

    if train_data is not None:
        plt.plot(
            range(1, len(train_data) + 1),
            train_data,
            label="Training",
        )

    if val_data is not None:
        plt.plot(
            range(1, len(val_data) + 1),
            val_data,
            label="Validation",
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savefig, dpi=300)
