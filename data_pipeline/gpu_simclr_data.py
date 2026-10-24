import math
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.247, 0.243, 0.261)


def load_uint8_images(base_dataset, cache_path=None):
    """base_dataset: e.g. ImageFolder(path) with NO transform. Returns (N,3,H,W) uint8."""
    if cache_path is not None and os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu")

    arrays = [np.asarray(img.convert("RGB")) for img, _ in base_dataset]
    data = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()

    if cache_path is not None and (not dist.is_initialized() or dist.get_rank() == 0):
        tmp = cache_path + ".tmp"
        torch.save(data, tmp)
        os.replace(tmp, cache_path)
    return data


def _uniform(n, lo, hi, device):
    return torch.rand(n, device=device) * (hi - lo) + lo


def random_resized_crop_flip(x, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), flip_p=0.5):
    B, C, H, W = x.shape
    dev = x.device
    area = _uniform(B, scale[0], scale[1], dev)
    r = torch.exp(_uniform(B, math.log(ratio[0]), math.log(ratio[1]), dev))
    w = torch.sqrt(area * r).clamp(max=1.0)
    h = torch.sqrt(area / r).clamp(max=1.0)
    cx = (torch.rand(B, device=dev) * (1 - w) + w / 2) * 2 - 1
    cy = (torch.rand(B, device=dev) * (1 - h) + h / 2) * 2 - 1
    sign = torch.where(torch.rand(B, device=dev) < flip_p,
                       -torch.ones(B, device=dev), torch.ones(B, device=dev))

    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = w * sign
    theta[:, 0, 2] = cx
    theta[:, 1, 1] = h
    theta[:, 1, 2] = cy
    grid = F.affine_grid(theta, (B, C, H, W), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                         align_corners=False)


def _gray(x):
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


_YIQ = torch.tensor([[0.299, 0.587, 0.114],
                     [0.596, -0.274, -0.322],
                     [0.211, -0.523, 0.312]])


def color_jitter(x, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8):
    B = x.shape[0]
    dev = x.device
    v = lambda t: t.view(B, 1, 1, 1)
    y = x

    y = (y * v(_uniform(B, 1 - brightness, 1 + brightness, dev))).clamp(0, 1)

    c = v(_uniform(B, 1 - contrast, 1 + contrast, dev))
    mean = _gray(y).mean(dim=(1, 2, 3), keepdim=True)
    y = ((y - mean) * c + mean).clamp(0, 1)

    s = v(_uniform(B, 1 - saturation, 1 + saturation, dev))
    g = _gray(y)
    y = ((y - g) * s + g).clamp(0, 1)

    ang = _uniform(B, -hue, hue, dev) * 2 * math.pi
    cos, sin = torch.cos(ang), torch.sin(ang)
    R = torch.zeros(B, 3, 3, device=dev)
    R[:, 0, 0] = 1
    R[:, 1, 1], R[:, 1, 2] = cos, -sin
    R[:, 2, 1], R[:, 2, 2] = sin, cos
    T = _YIQ.to(dev)
    M = torch.linalg.inv(T) @ R @ T                       # (B,3,3)
    y = torch.einsum("bij,bjhw->bihw", M, y).clamp(0, 1)

    apply = (torch.rand(B, device=dev) < p).view(B, 1, 1, 1)
    return torch.where(apply, y, x)


def random_grayscale(x, p=0.2):
    apply = (torch.rand(x.shape[0], device=x.device) < p).view(-1, 1, 1, 1)
    return torch.where(apply, _gray(x).expand_as(x), x)


def gaussian_blur3(x, sigma=(0.1, 2.0)):
    """3x3 Gaussian blur with a per-sample sigma (same as GaussianBlur(3))."""
    B, C, H, W = x.shape
    s = _uniform(B, sigma[0], sigma[1], x.device)
    w = torch.exp(-1.0 / (2 * s ** 2))
    k = torch.stack([w, torch.ones_like(w), w], dim=1)
    k = (k / k.sum(dim=1, keepdim=True)).repeat_interleave(C, dim=0)  # (B*C, 3)
    y = x.reshape(1, B * C, H, W)
    y = F.conv2d(F.pad(y, (1, 1, 0, 0), mode="reflect"), k.view(B * C, 1, 1, 3), groups=B * C)
    y = F.conv2d(F.pad(y, (0, 0, 1, 1), mode="reflect"), k.view(B * C, 1, 3, 1), groups=B * C)
    return y.reshape(B, C, H, W)


class GPUSimCLRAugment(nn.Module):
    """uint8 (B,3,H,W) on GPU -> two independently augmented, normalized views."""

    def __init__(self, blur=True, mean=MEAN, std=STD):
        super().__init__()
        self.blur = blur
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

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
    """Iterate to get (x_i, x_j) already on `device`. Works with or without DDP.

    batch_size is PER GPU. Call set_epoch(epoch) each epoch (also reachable as
    loader.sampler.set_epoch so existing DistributedSampler code keeps working).
    """

    def __init__(self, data_uint8, batch_size, device, blur=True, seed=0):
        self.data = data_uint8.to(device, non_blocking=True)
        self.batch_size = batch_size
        self.device = device
        self.aug = GPUSimCLRAugment(blur=blur).to(device)
        self.seed = seed
        self.epoch = 0
        self.sampler = self
        if dist.is_available() and dist.is_initialized():
            self.rank, self.world = dist.get_rank(), dist.get_world_size()
        else:
            self.rank, self.world = 0, 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return (self.data.shape[0] // self.world) // self.batch_size

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        n = self.data.shape[0]
        perm = torch.randperm(n, generator=g)[: (n // self.world) * self.world]
        idx = perm[self.rank::self.world].to(self.device)
        for b in range(len(self)):
            sel = idx[b * self.batch_size:(b + 1) * self.batch_size]
            yield self.aug(self.data[sel])

