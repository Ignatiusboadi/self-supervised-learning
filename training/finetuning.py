import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm.auto import tqdm
from utils.optimizer_factory import build_optimizer
import torch.distributed as dist
from sklearn.metrics import balanced_accuracy_score, f1_score


def load_pretrained(checkpoint_path, resnet_variant="resnet18", num_classes=10):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")
    if resnet_variant == "resnet18":
        model = models.resnet18(weights=None)
    elif resnet_variant == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(
            f"Unsupported backbone: {resnet_variant}. "
            "Choose 'resnet18' or 'resnet50'."
        )

    model.fc = nn.Identity()

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True, )
    model.load_state_dict(state_dict)
    feature_dim = model.fc.in_features if isinstance(model.fc, nn.Linear) else (
        512 if resnet_variant == "resnet18" else 2048)
    model.fc = nn.Linear(feature_dim, num_classes)
    return model


def configure_trainable_layers(model, strategy, unfreeze_backbone=False):
    """Set requires_grad according to the selected strategy."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    if strategy == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True

    elif strategy == "freeze_then_full":
        if unfreeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = True
        else:
            for parameter in model.fc.parameters():
                parameter.requires_grad = True

    elif strategy == "linear_probe":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    elif strategy == "partial":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    else:
        raise ValueError(f"Unknown fine-tuning strategy: {strategy}")


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

        progress.set_postfix(loss=f"{loss.item():.4f}", accuracy=f"{100.0 * correct / total:.2f}%", )
    return running_loss / total, 100.0 * correct / total



@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = logits.argmax(dim=1)

        all_targets.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    if total_samples == 0:
        raise ValueError("Validation loader is empty.")

    val_loss = total_loss / total_samples
    val_accuracy = (100.0 * sum(p == t for p, t in zip(all_predictions, all_targets)) / total_samples)

    balanced_acc = 100.0 * balanced_accuracy_score(all_targets, all_predictions)
    macro_f1 = 100.0 * f1_score(all_targets, all_predictions, average="macro", zero_division=0,)
    return val_loss, val_accuracy, balanced_acc, macro_f1


def fit_finetuning(
    model,
    train_loader,
    val_loader,
    device,
    output_dir,
    strategy="full",
    num_epochs=30,
    freeze_epochs=10,
    learning_rate=0.01,
    weight_decay=5e-4,
    opt_name="sgd",
    patience=7,
    min_delta=1e-4,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    is_main = rank == 0

    model = model.to(device)

    base_model = model.module if hasattr(model, "module") else model

    criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(
        opt_name,
        parameters=model.parameters(),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_balanced_accuracy": [],
        "val_macro_f1": [],
        "learning_rate": [],
    }

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    epochs_without_improvement = 0

    best_path = output_dir / "best_model.pth"

    for epoch in range(num_epochs):
        unfreeze_backbone = (
            strategy == "freeze_then_full"
            and epoch >= freeze_epochs
        )

        configure_trainable_layers(
            model,
            strategy,
            unfreeze_backbone=unfreeze_backbone,
        )

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        if is_main:
            val_loss, val_accuracy, val_balanced_acc, val_macro_f1 = (
                evaluate_one_epoch(
                    base_model,
                    val_loader,
                    criterion,
                    device,
                )
            )

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
            history["val_balanced_accuracy"].append(
                val_balanced_acc
            )
            history["val_macro_f1"].append(val_macro_f1)
            history["learning_rate"].append(current_lr)

            print(
                f"Epoch {epoch + 1:02d}/{num_epochs} | "
                f"Train loss: {train_loss:.4f} | "
                f"Train acc: {train_accuracy:.2f}% | "
                f"Val loss: {val_loss:.4f} | "
                f"Val acc: {val_accuracy:.2f}% | "
                f"Val balanced acc: {val_balanced_acc:.2f}% | "
                f"Val macro-F1: {val_macro_f1:.2f}% | "
                f"LR: {current_lr:.6f}"
            )

            improved = val_loss < best_val_loss - min_delta

            if improved:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0

                best_state = copy.deepcopy(
                    base_model.state_dict()
                )

                torch.save(best_state, best_path)
            else:
                epochs_without_improvement += 1

            should_stop = (
                patience is not None
                and epochs_without_improvement >= patience
            )

        else:
            should_stop = False

        scheduler.step()

        if distributed:
            stop_tensor = torch.tensor(
                [int(should_stop)],
                dtype=torch.int64,
                device=device,
            )
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())

        if should_stop:
            if is_main:
                print(
                    f"Early stopping at epoch {epoch + 1}. "
                    f"Best epoch: {best_epoch}, "
                    f"best validation loss: {best_val_loss:.4f}"
                )
            break

    if distributed:
        dist.barrier()

    if distributed:
        state = torch.load(
            best_path,
            map_location=device,
            weights_only=True,
        )
        base_model.load_state_dict(state)
    elif best_state is not None:
        base_model.load_state_dict(best_state)
    else:
        raise RuntimeError("No best checkpoint was saved.")

    if is_main:
        torch.save(
            base_model.state_dict(),
            output_dir / "final_model.pth",
        )
        print(f"Best checkpoint restored from epoch {best_epoch}.")

    if distributed:
        dist.barrier()

    return model, history
