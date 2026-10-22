import torch
import torch.nn.functional as F


def nt_xent_loss(z_i, z_j, temperature=0.5):
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)

    similarity = torch.matmul(z, z.T)
    N = z_i.shape[0]

    mask = (~torch.eye(2 * N, dtype=bool)).to(z.device)
    sim = similarity / temperature
    exp_sim = torch.exp(sim) * mask

    positive_sim = torch.exp(F.cosine_similarity(z_i, z_j) / temperature)
    positives = torch.cat([positive_sim, positive_sim], dim=0)

    denominator = exp_sim.sum(dim=1)
    loss = -torch.log(positives / denominator)
    return loss.mean()
