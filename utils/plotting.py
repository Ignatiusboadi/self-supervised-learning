import matplotlib.pyplot as plt

def plot_fig(data, xlabel, ylabel, title, savefig):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(data) + 1), data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savefig)
