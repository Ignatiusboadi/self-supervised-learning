import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from models.simclr import SimCLR


class TopKSAE(nn.Module):
    def __init__(self, d, h=None, dead_steps=256):
        super().__init__()

        h = h or 4 * d
        self.d, self.h, self.dead_steps = d, h, dead_steps
        self.W_enc = nn.Parameter(torch.empty(h, d))
        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        self.b_enc = nn.Parameter(torch.zeros(h))
        self.b_pre = nn.Parameter(torch.zeros(d))
        self.register_buffer("steps_since_fired", torch.zeros(h))

    @property
    def W_dec(self):
        return self.W_enc.t()

    def pre_acts(self, v):
        return F.linear(v - self.b_pre, self.W_enc, self.b_enc)

    @staticmethod
    def topk_relu(a, k):
        if not 1 <= k <= a.size(-1):
            raise ValueError(f"k must be between 1 and {a.size(-1)}, got {k}")

        vals, idx = a.topk(k, dim=-1)
        z = torch.zeros_like(a)
        z.scatter_(-1, idx, F.relu(vals))
        return z

    def decode(self, z):
        return F.linear(z, self.W_dec) + self.b_pre

    def encode(self, v, k):
        return self.topk_relu(self.pre_acts(v), k)

    def forward(self, v, k):
        a = self.pre_acts(v)
        z = self.topk_relu(a, k)

        return z, self.decode(z), a

    @torch.no_grad()
    def renorm_decoder(self):
        self.W_enc.copy_(F.normalize(self.W_enc, dim=1))

    @torch.no_grad()
    def update_dead(self, z):
        fired = (z > 0).any(0)

        self.steps_since_fired.add_(1)
        self.steps_since_fired[fired] = 0

    def dead_mask(self):
        return self.steps_since_fired > self.dead_steps


class CSRClassifier(nn.Module):
    def __init__(self, backbone, sae, classifier, k=32):
        super().__init__()
        self.backbone = backbone
        self.sae = sae
        self.classifier = classifier
        self.k = k

        for module in (self.backbone, self.sae):
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            features = extract(self.backbone, x)
            z = self.sae.encode(features, self.k)

        return self.classifier(z)


def mse(a, b):
    return (a - b).square().sum(-1).mean()


def recon_loss(sae, v, k, k_aux=512, beta=1 / 32):
    z, v_hat, a = sae(v, k)
    loss = mse(v, v_hat)
    z4 = sae.encode(v, min(4 * k, sae.h))
    loss = loss + mse(v, sae.decode(z4)) / 8
    dead = sae.dead_mask()

    if dead.any():
        e = (v - v_hat).detach()
        a_dead = a.masked_fill(~dead, float("-inf"))
        ka = min(k_aux, int(dead.sum().item()))
        z_aux = sae.topk_relu(a_dead, ka)
        e_hat = F.linear(z_aux, sae.W_dec)
        loss = loss + beta * mse(e, e_hat)

    sae.update_dead(z)
    return loss, z


def ncl_loss(z):
    sim = z @ z.t()
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -log_prob.diagonal().mean()

@torch.no_grad()
def extract(backbone, x):
    features = backbone(x)
    if isinstance(features, (tuple, list)):
        features = features[0]

    if features.ndim == 4:
        features = F.adaptive_avg_pool2d(
            features, 1
        ).flatten(1)

    elif features.ndim > 2:
        features = features.flatten(1)

    if features.ndim != 2:
        raise ValueError(
            f"Expected 2D backbone features, got {features.shape}"
        )

    return features


def train_csr(backbone, train_loader, d, h=None, k=32,
              gamma=1.0, epochs=10, lr=4e-5,
              k_aux=512, beta=1 / 32, device="cuda"):
    h = h or 4 * d
    backbone = backbone.to(device).eval()
    backbone.requires_grad_(False)
    sae = TopKSAE(d, h).to(device)
    with torch.no_grad():
        features = []
        for batch in tqdm(train_loader):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device, non_blocking=True)
            v = extract(backbone, x)
            features.append(v)
        if not features:
            raise ValueError("Training loader is empty.")

        v0 = torch.cat(features, dim=0)
        if v0.size(1) != d:
            raise ValueError(
                f"Expected backbone dimension {d}, "
                f"got {v0.size(1)}")
        sae.b_pre.copy_(v0.mean(0))

    sae.renorm_decoder()
    opt = torch.optim.Adam(
        sae.parameters(),
        lr=lr,
        weight_decay=1e-4,
        eps=6.25e-10
    )
    for ep in range(epochs):
        sae.train()
        total_loss = 0.0
        total_recon = 0.0
        total_ncl = 0.0
        num_batches = 0

        for batch in tqdm(train_loader):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device, non_blocking=True)
            if x.size(0) < 2:
                continue
            with torch.no_grad():
                v = extract(backbone, x)

            if v.size(1) != d:
                raise ValueError(
                    f"Expected feature dimension {d}, "
                    f"got {v.size(1)}"
                )

            l_recon, z = recon_loss(
                sae, v, k, k_aux=k_aux, beta=beta
            )
            l_ncl = ncl_loss(z)
            loss = l_recon + gamma * l_ncl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            sae.renorm_decoder()
            total_loss += loss.item()
            total_recon += l_recon.item()
            total_ncl += l_ncl.item()

            num_batches += 1

        if num_batches == 0:
            raise ValueError(
                "No valid CSR batches. Check batch_size and drop_last."
            )

        print(
            f"[CSR] Epoch {ep + 1}/{epochs} "
            f"Loss: {total_loss / num_batches:.4f} "
            f"Recon: {total_recon / num_batches:.4f} "
            f"NCL: {total_ncl / num_batches:.4f} "
            f"Dead: {sae.dead_mask().float().mean().item():.3f}"
        )

    return sae


@torch.no_grad()
def sparse_features(backbone, sae, x, k):
    v = extract(backbone, x)
    return sae.encode(v, k)


def load_simclr_encoder(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    if not cfg:
        raise ValueError("The checkpoint does not contain a model configuration.")
    cfg = {k: v for k, v in cfg.items() if k != "feature_dim"}
    model = SimCLR(**cfg)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if state is None:
        raise KeyError("Could not find model_state_dict or state_dict in checkpoint.")

    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model.encoder, model.feature_dim


def train_classifier(backbone, sae, clf_loader, val_loader,
                     num_classes, k=32, epochs=30,
                     lr=1e-3, device="cuda"):
    backbone = backbone.to(device).eval()
    backbone.requires_grad_(False)

    sae = sae.to(device).eval()
    sae.requires_grad_(False)

    clf = nn.Linear(sae.h, num_classes).to(device)
    opt = torch.optim.AdamW(
        clf.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    for ep in range(epochs):
        clf.train()
        train_loss = 0.0
        train_correct = 0
        train_n = 0

        for x, y in clf_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                z = sparse_features(backbone, sae, x, k)
            logits = clf(z)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            train_loss += loss.item() * y.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_n += y.size(0)

        clf.eval()
        val_loss = 0.0
        val_correct = 0
        val_n = 0

        with torch.no_grad():
            for x, y in val_loader:

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                z = sparse_features(backbone, sae, x, k)
                logits = clf(z)

                loss = F.cross_entropy(logits, y)

                val_loss += loss.item() * y.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_n += y.size(0)

        print(
            f"[CLF] Epoch {ep + 1}/{epochs} "
            f"Train Loss: {train_loss / train_n:.4f} "
            f"Train Acc: {train_correct / train_n:.4f} "
            f"Val Loss: {val_loss / val_n:.4f} "
            f"Val Acc: {val_correct / val_n:.4f}"
        )

    return clf
