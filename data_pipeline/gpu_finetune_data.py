import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _read(path, size):
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size is not None:
            im = im.resize((size, size), Image.BILINEAR)
        return np.asarray(im)


def build_split_cache(split_dir, cache_path, size=None, num_threads=32):
    ds = ImageFolder(str(split_dir))
    paths = [p for p, _ in ds.samples]
    with ThreadPoolExecutor(num_threads) as ex:
        arrays = list(tqdm(ex.map(lambda p: _read(p, size), paths), total=len(paths),
                           desc=f"Caching {Path(split_dir).name}"))
    payload = {
        "data": torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous(),
        "labels": torch.tensor(ds.targets, dtype=torch.long),
        "classes": ds.classes,
    }
    tmp = str(cache_path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, cache_path)


def load_split(split_dir, cache_path, size=None):
    cache_path = str(cache_path)
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0 and not os.path.exists(cache_path):
            build_split_cache(split_dir, cache_path, size)
        dist.barrier()
    elif not os.path.exists(cache_path):
        build_split_cache(split_dir, cache_path, size)
    return torch.load(cache_path, map_location="cpu")


def get_stratified_subset_indices(targets, fraction=0.5, seed=42):
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    if fraction == 1:
        return list(range(len(targets)))

    rng = random.Random(seed)
    targets = list(targets)
    selected = []
    for class_id in sorted(set(targets)):
        class_indices = [i for i, t in enumerate(targets) if t == class_id]
        rng.shuffle(class_indices)
        selected.extend(class_indices[:max(1, int(len(class_indices) * fraction))])
    rng.shuffle(selected)
    return selected


class GPUFinetuneTransform(nn.Module):
    """uint8 (B,3,H,W) -> normalized float. Train: + random horizontal flip."""

    def __init__(self, train, mean=CIFAR10_MEAN, std=CIFAR10_STD):
        super().__init__()
        self.train_mode = train
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x_uint8):
        x = x_uint8.float().div_(255.0)
        if self.train_mode:
            flip = (torch.rand(x.shape[0], device=x.device) < 0.5).view(-1, 1, 1, 1)
            x = torch.where(flip, x.flip(3), x)
        return (x - self.mean) / self.std


class GPULabeledLoader:
    def __init__(self, data_uint8, labels, batch_size, device, train, shard=None,
                 drop_last=False, seed=0):
        self.data = data_uint8.to(device, non_blocking=True)
        self.labels = labels.to(device, non_blocking=True)
        self.batch_size = batch_size
        self.device = device
        self.train = train
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.sampler = self
        self.transform = GPUFinetuneTransform(train).to(device)

        shard = train if shard is None else shard
        if shard and dist.is_available() and dist.is_initialized():
            self.rank, self.world = dist.get_rank(), dist.get_world_size()
        else:
            self.rank, self.world = 0, 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _n_local(self):
        n = self.data.shape[0]
        return n // self.world if self.world > 1 else n

    def __len__(self):
        n = self._n_local()
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def __iter__(self):
        n = self.data.shape[0]
        if self.train:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(n, generator=g)
        else:
            perm = torch.arange(n)
        self.epoch += 1

        if self.world > 1:
            perm = perm[: (n // self.world) * self.world][self.rank::self.world]
        idx = perm.to(self.device)

        for b in range(len(self)):
            sel = idx[b * self.batch_size:(b + 1) * self.batch_size]
            yield self.transform(self.data[sel]), self.labels[sel]


def get_stratified_split_indices(targets, val_fraction=0.1, seed=42):
    rng = random.Random(seed)
    targets = list(targets)
    train_idx, val_idx = [], []
    for c in sorted(set(targets)):
        idx = [i for i, t in enumerate(targets) if t == c]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_fraction)) if len(idx) > 1 else 0
        val_idx += idx[:n_val]
        train_idx += idx[n_val:]
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def get_finetuning_loaders_gpu(data_dir, device, batch_size=128, img_size=None, data_fraction=1.0,
                               val_fraction=0.1, seed=42, train_cache_path=".",
                               test_cache_path:str=".", size=None):
    train_cache_path, test_cache_path = Path(train_cache_path), Path(test_cache_path)
    if data_dir is None:
        missing = [p for p in (train_cache_path, test_cache_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Cache file(s) not found: {missing}. "
                                    f"Pass data_dir so they can be built.")
    else:
        data_dir = Path(data_dir)

    for p in (train_cache_path, test_cache_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    train = load_split(data_dir / "train" if data_dir else None, train_cache_path, size)
    test = load_split(data_dir / "test" if data_dir else None, test_cache_path, size)

    if train["classes"] != test["classes"]:
        raise ValueError("Training and test class mappings differ.")

    data, labels = train["data"], train["labels"]
    val_loader = None
    if val_fraction > 0:
        tr, va = get_stratified_split_indices(labels.tolist(), val_fraction, seed)
        tr, va = torch.tensor(tr), torch.tensor(va)
        val_data, val_labels = data[va], labels[va]
        data, labels = data[tr], labels[tr]
    if data_fraction < 1.0:
        idx = torch.tensor(get_stratified_subset_indices(labels.tolist(), data_fraction, seed))
        data, labels = data[idx], labels[idx]

    train_loader = GPULabeledLoader(data, labels, batch_size, device, train=True, seed=seed)
    if val_fraction > 0:
        val_loader = GPULabeledLoader(val_data, val_labels, batch_size, device, train=False, shard=False)

    test_loader = GPULabeledLoader(test["data"], test["labels"], batch_size, device,
                                   train=False, shard=False)
    return train_loader, val_loader, test_loader, train["classes"]
