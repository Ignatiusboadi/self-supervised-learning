import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_finetuning_transforms(img_size=None, dataset="cifar10"):
    if dataset == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif dataset == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    elif dataset in ("imagenet", "aptos"):
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_ops = []
    test_ops = []

    if img_size is not None:
        train_ops.append(transforms.Resize((img_size, img_size)))
        test_ops.append(transforms.Resize((img_size, img_size)))

    train_ops.extend([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return transforms.Compose(train_ops), transforms.Compose(test_ops)


def get_stratified_subset_indices(targets, fraction=0.5, seed=42):
    """Sample the same fraction from each class."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")

    if fraction == 1:
        return list(range(len(targets)))

    rng = random.Random(seed)
    targets = list(targets)
    selected_indices = []

    for class_id in sorted(set(targets)):
        class_indices = [
            i for i, target in enumerate(targets)
            if target == class_id
        ]
        rng.shuffle(class_indices)

        n_selected = max(1, int(len(class_indices) * fraction))
        selected_indices.extend(class_indices[:n_selected])

    rng.shuffle(selected_indices)
    return selected_indices


def get_stratified_split_indices(targets, val_fraction=0.1, seed=42):
    """Split indices by class into training and validation sets."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1).")

    rng = random.Random(seed)
    targets = list(targets)
    train_indices, val_indices = [], []

    for class_id in sorted(set(targets)):
        class_indices = [
            i for i, target in enumerate(targets)
            if target == class_id
        ]
        rng.shuffle(class_indices)

        if len(class_indices) < 2:
            train_indices.extend(class_indices)
            continue

        n_val = max(1, int(len(class_indices) * val_fraction))
        n_val = min(n_val, len(class_indices) - 1)

        val_indices.extend(class_indices[:n_val])
        train_indices.extend(class_indices[n_val:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


def get_finetuning_loaders(
    data_dir,
    dataset_name="cifar10",
    img_size=None,
    batch_size=128,
    data_fraction=1.0,
    val_fraction=0.1,
    seed=42,
    num_workers=0,
):
    data_dir = Path(data_dir)
    train_transform, test_transform = get_finetuning_transforms(dataset_name, img_size)

    train_dataset = datasets.ImageFolder(
        root=str(data_dir / "train"),
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        root=str(data_dir / "train"),
        transform=test_transform,
    )

    test_dataset = datasets.ImageFolder(
        root=str(data_dir / "test"),
        transform=test_transform,
    )

    if (
        train_dataset.class_to_idx != val_dataset.class_to_idx
        or train_dataset.class_to_idx != test_dataset.class_to_idx
    ):
        raise ValueError("Training, validation, and test class mappings differ.")

    train_indices, val_indices = get_stratified_split_indices(
        train_dataset.targets,
        val_fraction=val_fraction,
        seed=seed,
    )

    train_targets = [train_dataset.targets[i] for i in train_indices]
    relative_indices = get_stratified_subset_indices(
        train_targets,
        fraction=data_fraction,
        seed=seed,
    )
    train_indices = [train_indices[i] for i in relative_indices]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    loader_generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=loader_generator,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    classes = train_dataset.classes

    return train_loader, val_loader, test_loader, classes
