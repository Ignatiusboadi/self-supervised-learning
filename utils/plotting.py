import matplotlib.pyplot as plt

def plot_fig(
    train_data,
    xlabel,
    ylabel,
    title,
    savefig,
    val_data=None,
):
    plt.figure(figsize=(8, 5))

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
    plt.show()
    plt.close()
