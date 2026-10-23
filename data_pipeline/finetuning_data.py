
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_finetuning_transforms():
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD), ])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD), ])
    return train_transform, test_transform


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
        class_indices = [i for i, target in enumerate(targets) if target == class_id]
        rng.shuffle(class_indices)
        n_selected = max(1, int(len(class_indices) * fraction))
        selected_indices.extend(class_indices[:n_selected])

    rng.shuffle(selected_indices)
    return selected_indices


def get_finetuning_loaders(data_dir, batch_size=128, data_fraction=1.0, seed=42, num_workers=0, ):
    data_dir = Path(data_dir)
    train_transform, test_transform = get_finetuning_transforms()

    train_dataset = datasets.ImageFolder(root=str(data_dir / "train"), transform=train_transform, )
    test_dataset = datasets.ImageFolder(root=str(data_dir / "test"), transform=test_transform, )

    if train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError("Training and test class mappings differ.")

    if data_fraction < 1.0:
        indices = get_stratified_subset_indices(train_dataset.targets, fraction=data_fraction, seed=seed,)
        train_dataset = Subset(train_dataset, indices)

    loader_generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, generator=loader_generator, )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=pin_memory, )

    classes = (train_dataset.dataset.classes if isinstance(train_dataset, Subset) else train_dataset.classes)
    return train_loader, test_loader, classes
