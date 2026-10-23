import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

@dataclass
class Runtime:
    device: torch.device
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    backend: str | None

    @property
    def is_main(self):
        return self.rank == 0

    def print(self, *args, **kwargs):
        if self.is_main:
            print(*args, **kwargs)


def setup_runtime(requested_device="auto"):
    """
    Initialize single-device or multi-GPU execution.
    """
    requested_device = requested_device.lower()

    cuda_available = torch.cuda.is_available()
    mps_available = (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if requested_device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA requested but unavailable")

    if requested_device == "mps" and not mps_available:
        raise RuntimeError("MPS requested but unavailable")

    if requested_device == "cpu":
        device = torch.device("cpu")

    elif requested_device == "mps":
        device = torch.device("mps")

    elif requested_device == "cuda":
        if world_size > 1 and not torch.cuda.is_available():
            raise RuntimeError("Distributed CUDA requires CUDA")
        device = torch.device(f"cuda:{local_rank}" if world_size > 1 else "cuda")

    elif requested_device == "auto":
        if cuda_available:
            device = torch.device(f"cuda:{local_rank}" if world_size > 1 else "cuda")
        elif mps_available:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    else:
        raise ValueError("device must be one of: auto, cuda, mps, cpu")
    distributed = world_size > 1

    if distributed:
        if device.type != "cuda":
            raise RuntimeError("This implementation supports DDP only with CUDA. Use one process for MPS or CPU.")

        torch.cuda.set_device(device)
        backend = "nccl"

        if not dist.is_initialized():
            dist.init_process_group(backend=backend)

        rank = dist.get_rank()
        world_size = dist.get_world_size()

    else:
        rank = 0
        local_rank = 0
        backend = None

    return Runtime(device=device, distributed=distributed, rank=rank,
                   local_rank=local_rank, world_size=world_size, backend=backend, )


def wrap_model(model, runtime):
    """Move the model to its device and wrap it with DDP when needed."""
    model = model.to(runtime.device)

    if runtime.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[runtime.local_rank], output_device=runtime.local_rank, )
    return model


def cleanup_runtime(runtime):
    if runtime.distributed and dist.is_initialized():
        dist.destroy_process_group()
