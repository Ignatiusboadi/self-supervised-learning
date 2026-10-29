import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _read(path, size=None):
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size is not None:
            im = im.resize(
                (size, size),
                Image.Resampling.BILINEAR,
            )
        return np.asarray(im)


def build_split_cache(
    split_dir,
    cache_path,
    size=None,
    num_threads=32,
):
    split_dir = Path(split_dir)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    ds = ImageFolder(str(split_dir))

    if len(ds) == 0:
        raise ValueError(f"No images found in {split_dir}")

    paths = [path for path, _ in ds.samples]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        arrays = list(
            tqdm(
                executor.map(lambda p: _read(p, size), paths),
                total=len(paths),
                desc=f"Caching {split_dir.name}",
            )
        )

    data = (
        torch.from_numpy(np.stack(arrays))
        .permute(0, 3, 1, 2)
        .contiguous()
    )

    payload = {
        "data": data,
        "labels": torch.tensor(ds.targets, dtype=torch.long),
        "classes": ds.classes,
    }

    tmp = cache_path.with_name(cache_path.name + ".tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, cache_path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_split(split_dir, cache_path, size=None):
    cache_path = Path(cache_path)
    distributed = dist.is_available() and dist.is_initialized()

    if distributed:
        if dist.get_rank() == 0 and not cache_path.exists():
            if split_dir is None:
                raise FileNotFoundError(cache_path)
            build_split_cache(split_dir, cache_path, size)

        dist.barrier()

    elif not cache_path.exists():
        if split_dir is None:
            raise FileNotFoundError(cache_path)
        build_split_cache(split_dir, cache_path, size)

    payload = torch.load(
        cache_path,
        map_location="cpu",
        weights_only=True,
    )

    data = payload["data"]

    if (
        not isinstance(data, torch.Tensor)
        or data.ndim != 4
        or data.shape[1] != 3
        or data.dtype != torch.uint8
    ):
        raise ValueError(
            f"Expected uint8 images with shape (N, 3, H, W); "
            f"got shape={getattr(data, 'shape', None)}, "
            f"dtype={getattr(data, 'dtype', None)}"
        )

    if data.shape[0] == 0:
        raise ValueError(f"Empty image cache: {cache_path}")

    return payload


def _uniform(n, lo, hi, device):
    return torch.rand(n, device=device) * (hi - lo) + lo


def random_resized_crop_flip(
    x,
    scale=(0.08, 1.0),
    ratio=(3 / 4, 4 / 3),
    flip_p=0.5,
):
    B, C, H, W = x.shape
    device = x.device

    area = _uniform(B, scale[0], scale[1], device)
    aspect = torch.exp(
        _uniform(
            B,
            math.log(ratio[0]),
            math.log(ratio[1]),
            device,
        )
    )

    crop_w = torch.sqrt(area * aspect).clamp(max=1.0)
    crop_h = torch.sqrt(area / aspect).clamp(max=1.0)

    cx = (
        torch.rand(B, device=device) * (1 - crop_w)
        + crop_w / 2
    ) * 2 - 1

    cy = (
        torch.rand(B, device=device) * (1 - crop_h)
        + crop_h / 2
    ) * 2 - 1

    sign = torch.where(
        torch.rand(B, device=device) < flip_p,
        -torch.ones(B, device=device),
        torch.ones(B, device=device),
    )

    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0] = crop_w * sign
    theta[:, 0, 2] = cx
    theta[:, 1, 1] = crop_h
    theta[:, 1, 2] = cy

    grid = F.affine_grid(
        theta,
        (B, C, H, W),
        align_corners=False,
    )

    return F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )


def _gray(x):
    return (
        0.299 * x[:, 0:1]
        + 0.587 * x[:, 1:2]
        + 0.114 * x[:, 2:3]
    )


def color_jitter(
    x,
    brightness=0.4,
    contrast=0.4,
    saturation=0.4,
    hue=0.1,
    p=0.8,
):
    B = x.shape[0]
    device = x.device

    def view(v):
        return v.view(B, 1, 1, 1)

    y = (
        x * view(_uniform(B, 1 - brightness, 1 + brightness, device))
    ).clamp(0, 1)

    c = view(_uniform(B, 1 - contrast, 1 + contrast, device))
    mean = _gray(y).mean(dim=(2, 3), keepdim=True)
    y = ((y - mean) * c + mean).clamp(0, 1)

    s = view(_uniform(B, 1 - saturation, 1 + saturation, device))
    gray = _gray(y)
    y = ((y - gray) * s + gray).clamp(0, 1)

    angle = _uniform(B, -hue, hue, device) * 2 * math.pi
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)

    rotation = torch.zeros(B, 3, 3, device=device)
    rotation[:, 0, 0] = 1
    rotation[:, 1, 1] = cos_a
    rotation[:, 1, 2] = -sin_a
    rotation[:, 2, 1] = sin_a
    rotation[:, 2, 2] = cos_a

    yiq = torch.tensor(
        [
            [0.299, 0.587, 0.114],
            [0.596, -0.274, -0.322],
            [0.211, -0.523, 0.312],
        ],
        device=device,
        dtype=x.dtype,
    )

    transform = torch.linalg.inv(yiq) @ rotation @ yiq
    y = torch.einsum("bij,bjhw->bihw", transform, y).clamp(0, 1)

    apply = (torch.rand(B, device=device) < p).view(B, 1, 1, 1)
    return torch.where(apply, y, x)


def random_grayscale(x, p=0.2):
    apply = (
        torch.rand(x.shape[0], device=x.device) < p
    ).view(-1, 1, 1, 1)

    return torch.where(apply, _gray(x).expand_as(x), x)


def gaussian_blur3(x, sigma=(0.1, 2.0)):
    B, C, H, W = x.shape

    if min(H, W) < 2:
        return x

    s = _uniform(B, sigma[0], sigma[1], x.device)
    weight = torch.exp(-1.0 / (2 * s.square()))

    kernel = torch.stack(
        [weight, torch.ones_like(weight), weight],
        dim=1,
    )
    kernel = kernel / kernel.sum(dim=1, keepdim=True)
    kernel = kernel.repeat_interleave(C, dim=0)

    y = x.reshape(1, B * C, H, W)

    y = F.conv2d(
        F.pad(y, (1, 1, 0, 0), mode="reflect"),
        kernel.view(B * C, 1, 1, 3),
        groups=B * C,
    )

    y = F.conv2d(
        F.pad(y, (0, 0, 1, 1), mode="reflect"),
        kernel.view(B * C, 1, 3, 1),
        groups=B * C,
    )

    return y.reshape(B, C, H, W)


class GPUSimCLRAugment(nn.Module):
    def __init__(
        self,
        blur=True,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
    ):
        super().__init__()
        self.blur = blur

        self.register_buffer(
            "mean",
            torch.tensor(mean).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor(std).view(1, 3, 1, 1),
        )

    def _view(self, x):
        x = random_resized_crop_flip(x)
        x = color_jitter(x)
        x = random_grayscale(x)

        if self.blur:
            x = gaussian_blur3(x)

        return (x - self.mean) / self.std

    @torch.no_grad()
    def forward(self, batch_uint8):
        x = batch_uint8.float().div_(255.0)
        return self._view(x), self._view(x)


class GPUSimCLRLoader:
    """
    Yields (view1, view2), both on the target device.

    batch_size is PER GPU. In DDP, each rank gets a
    non-overlapping strided shard of the global permutation.
    Incomplete batches are dropped for stable contrastive loss.
    """

    def __init__(
        self,
        data_uint8,
        batch_size,
        device,
        blur=True,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
        seed=0,
    ):
        if data_uint8.ndim != 4 or data_uint8.shape[1] != 3:
            raise ValueError("Expected images shaped (N, 3, H, W).")
        if data_uint8.dtype != torch.uint8:
            raise ValueError("Expected uint8 image data.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        self.data = data_uint8.to(device, non_blocking=True)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.seed = seed
        self.epoch = 0

        self.aug = GPUSimCLRAugment(
            blur=blur,
            mean=mean,
            std=std,
        ).to(device)

        self.sampler = self

        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world = dist.get_world_size()
        else:
            self.rank, self.world = 0, 1

        usable_n = (self.data.shape[0] // self.world) * self.world
        if usable_n < self.world * self.batch_size:
            raise ValueError(
                "Not enough images for one full batch per rank. "
                f"N={self.data.shape[0]}, world={self.world}, "
                f"batch_size={self.batch_size}."
            )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        n_local = self.data.shape[0] // self.world
        return n_local // self.batch_size

    def __iter__(self):
        n = self.data.shape[0]
        usable_n = (n // self.world) * self.world

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        perm = torch.randperm(n, generator=generator)[:usable_n]
        idx = perm[self.rank::self.world].to(self.device)

        for b in range(len(self)):
            sel = idx[
                b * self.batch_size:(b + 1) * self.batch_size
            ]

            assert sel.numel() == self.batch_size
            assert sel.min().item() >= 0
            assert sel.max().item() < n

            batch = self.data[sel]
            yield self.aug(batch)


def get_simclr_pretraining_loader(
    data_dir,
    cache_path,
    device,
    batch_size=128,
    img_size=None,
    blur=True,
    mean=CIFAR10_MEAN,
    std=CIFAR10_STD,
    seed=42,
    num_threads=32,
):
    data_dir = Path(data_dir) if data_dir is not None else None
    cache_path = Path(cache_path) if cache_path is not None else None
    cache_path.parent.mkdir(parents=True, exist_ok=True) if cache_path is not None else None

    distributed = dist.is_available() and dist.is_initialized()

    if distributed:
        if dist.get_rank() == 0 and not cache_path.exists():
            if data_dir is None:
                raise FileNotFoundError(cache_path)

            build_split_cache(
                data_dir,
                cache_path,
                size=img_size,
                num_threads=num_threads,
            )

        dist.barrier()

        payload = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=True,
        )
    else:
        if not cache_path.exists():
            if data_dir is None:
                raise FileNotFoundError(cache_path)

            build_split_cache(
                data_dir,
                cache_path,
                size=img_size,
                num_threads=num_threads,
            )

        payload = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=True,
        )

    loader = GPUSimCLRLoader(
        data_uint8=payload["data"],
        batch_size=batch_size,
        device=device,
        blur=blur,
        mean=mean,
        std=std,
        seed=seed,
    )

    return loader, payload
