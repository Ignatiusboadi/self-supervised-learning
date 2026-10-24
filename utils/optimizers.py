import torch
from torch.optim import Optimizer


class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling with momentum."""

    def __init__(self, params, lr=0.3, momentum=0.9, weight_decay=1e-6,
                 eta=0.001, weight_decay_filter=True, lars_adaptation_filter=True, eps=1e-8, ):
        self.name = "LARS"
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if momentum < 0:
            raise ValueError("momentum must be non-negative")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if eta < 0:
            raise ValueError("eta must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be positive")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter, eps=eps, )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                excluded = p.ndim <= 1

                if group["weight_decay"] != 0 and not (group["weight_decay_filter"] and excluded):
                    g = g.add(p, alpha=group["weight_decay"])

                if not (group["lars_adaptation_filter"] and excluded):
                    p_norm = torch.norm(p)
                    g_norm = torch.norm(g)

                    trust_ratio = torch.where((p_norm > 0) & (g_norm > 0),
                                              group["eta"] * p_norm / (g_norm + group["eps"]),
                                              torch.ones_like(p_norm), )

                    g = g * trust_ratio

                state = self.state[p]

                if "mu" not in state:
                    state["mu"] = torch.zeros_like(p)

                mu = state["mu"]
                mu.mul_(group["momentum"]).add_(g)

                p.add_(mu, alpha=-group["lr"])

        return loss


class SAM(torch.optim.Optimizer):
    """Sharness Aware Minimization with momentum."""
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        self.name = "SAM"
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None]), p=2)
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def disable_bn_running_stats(model):
    """Call before the 2nd pass so BatchNorm running stats aren't updated twice."""

    def _disable(m):
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.backup_momentum = m.momentum
            m.momentum = 0

    model.apply(_disable)


def enable_bn_running_stats(model):
    """Call after the 2nd pass to restore normal BatchNorm momentum."""

    def _enable(m):
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and hasattr(m, "backup_momentum"):
            m.momentum = m.backup_momentum

    model.apply(_enable)