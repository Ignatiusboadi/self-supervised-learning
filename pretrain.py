import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from data_pipeline.augmentations import simclr_train_transforms
from data_pipeline.datasets import SimCLRDataset
from utils.optimizers import LARS, SAM
from utils.plotting import plot_fig
from training.train import train_one_epoch
from methods.simclr import SimCLRModel


def main():
    root = "."
    dataset_path = os.path.join(root, "data", "cifar10", "train")
    checkpoint_dir = os.path.join(root, "outputs/checkpoints", "simclr")
    plot_dir = os.path.join(root, "outputs/plots", "simclr")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 2
    batch_size = 512

    base_lr = 0.3
    peak_lr = base_lr * batch_size / 256

    momentum = 0.9
    weight_decay = 1e-6
    eta = 0.001
    warmup_epochs = int(num_epochs * .2)

    base_dataset = ImageFolder(root=dataset_path, transform=None, )
    dataset = SimCLRDataset(base_dataset, simclr_train_transforms(), )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
        pin_memory=(device.type == "cuda"), )
    model = SimCLRModel().to(device)
    optimizer = LARS(model.parameters(), lr=peak_lr, momentum=momentum, weight_decay=weight_decay,
        eta=eta, weight_decay_filter=True, lars_adaptation_filter=True, )
    optimizer = SAM(model.parameters(), torch.optim.SGD, lr=0.01, momentum=0.9, rho=0.05)

    steps_per_epoch = len(loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = min(warmup_epochs * steps_per_epoch, total_steps - 1, )

    train_losses = []
    learning_rates = []
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss, epoch_lrs, global_step = train_one_epoch(model=model, train_loader=loader, optimizer=optimizer,
            device=device, global_step=global_step, total_steps=total_steps, warmup_steps=warmup_steps,
            peak_lr=peak_lr, epoch=epoch, num_epochs=num_epochs, )

        train_losses.append(epoch_loss)
        learning_rates.extend(epoch_lrs)

        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f} LR: {epoch_lrs[-1]:.6f}")

    checkpoint_path = os.path.join(checkpoint_dir, f"simclr_{model.resnet_variant}_checkpoint.pth", )
    loss_fig = os.path.join(plot_dir, "loss.png")
    plot_fig(train_losses, "Training step", "Mean NT-Xent Loss", "SimCLR Training Loss", loss_fig)
    lr_fig = os.path.join(plot_dir, "lr.png")
    plot_fig(learning_rates, "Training step", "Learning rate", "LR: warm-up + cosine decay", lr_fig)
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "epochs": num_epochs, "global_step": global_step, "train_losses": train_losses, }, checkpoint_path, )
    encoder_path = os.path.join(checkpoint_dir, f"simclr_{model.resnet_variant}_encoder.pth",)
    torch.save(model.encoder.state_dict(), encoder_path)

    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Encoder saved to: {encoder_path}")


if __name__ == "__main__":
    main()
