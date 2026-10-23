
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather


def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    SimCLR NT-Xent loss.

    Single GPU: uses the local batch.
    DDP: uses embeddings from all GPUs as contrastive candidates.
    """
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    local_batch_size = z_i.shape[0]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        z_i_all = torch.cat(all_gather(z_i), dim=0)
        z_j_all = torch.cat(all_gather(z_j), dim=0)
    else:
        rank = 0
        world_size = 1
        z_i_all = z_i
        z_j_all = z_j

    global_batch_size = z_i_all.shape[0] * world_size
    candidates = torch.cat([z_i_all, z_j_all], dim=0)
    anchors = torch.cat([z_i, z_j], dim=0)

    logits = anchors @ candidates.T
    logits = logits / temperature

    local_indices = (
        rank * local_batch_size
        + torch.arange(local_batch_size, device=z_i.device)
    )

    positive_indices = torch.cat([
        global_batch_size + local_indices,
        local_indices,
    ])

    self_indices = torch.cat([
        local_indices,
        global_batch_size + local_indices,
    ])

    logits[
        torch.arange(2 * local_batch_size, device=z_i.device),
        self_indices,
    ] = float("-inf")

    return F.cross_entropy(logits, positive_indices)
