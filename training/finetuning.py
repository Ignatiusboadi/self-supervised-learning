
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm.auto import tqdm


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


def fit_finetuning(model, train_loader, device, output_dir, strategy="full", num_epochs=30,
    freeze_epochs=10, learning_rate=0.01, weight_decay=5e-4, ):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=True,)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, )

    history = {"train_loss": [], "train_accuracy": [], "learning_rate": [], }

    best_train_accuracy = -1.0
    best_state = None

    for epoch in range(num_epochs):
        unfreeze_backbone = (strategy == "freeze_then_full" and epoch >= freeze_epochs)

        configure_trainable_layers(model, strategy, unfreeze_backbone=unfreeze_backbone, )

        train_loss, train_accuracy = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer,
            criterion=criterion, device=device, epoch=epoch, )

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["learning_rate"].append(current_lr)

        print(f"Epoch {epoch + 1:02d}/{num_epochs} | Loss: {train_loss:.4f} | Train accuracy: {train_accuracy:.2f}% | "
            f"LR: {current_lr:.6f}")

        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
            best_state = copy.deepcopy(model.state_dict())

            torch.save(best_state, output_dir / "best_model.pth", )

        scheduler.step()

    torch.save(model.state_dict(), output_dir / "final_model.pth", )
    return model, history
