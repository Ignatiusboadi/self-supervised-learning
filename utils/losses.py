import torch
import torch.nn.functional as F
import torch.distributed as dist


def _is_distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _gather_with_grad(x):
    if not _is_distributed():
        return x, 0, 1

    from torch.distributed.nn.functional import all_gather
    return torch.cat(all_gather(x), dim=0), dist.get_rank(), dist.get_world_size()


def nt_xent_loss(z_i, z_j, temperature=0.5):
    if z_i.shape != z_j.shape:
        raise ValueError(f"shape mismatch: {z_i.shape} vs {z_j.shape}")

    with torch.autocast(device_type=z_i.device.type, enabled=False):
        z_i = F.normalize(z_i.float(), dim=1)
        z_j = F.normalize(z_j.float(), dim=1)
        B = z_i.shape[0]

        z_i_all, rank, _ = _gather_with_grad(z_i)
        z_j_all, _, _ = _gather_with_grad(z_j)
        G = z_i_all.shape[0]

        anchors = torch.cat([z_i, z_j], dim=0)
        candidates = torch.cat([z_i_all, z_j_all], dim=0)
        logits = anchors @ candidates.T / temperature

        local = rank * B + torch.arange(B, device=z_i.device)
        rows = torch.arange(2 * B, device=z_i.device)
        self_idx = torch.cat([local, G + local])
        pos_idx = torch.cat([G + local, local])

        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[rows, self_idx] = True
        logits = logits.masked_fill(mask, float("-inf"))

        return F.cross_entropy(logits, pos_idx)
