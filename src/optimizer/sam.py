# implementation of sharpness aware minimizer
import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho: float, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_opt = base_optimizer(params, **kwargs)
        self.param_groups = self.base_opt.param_groups

    @torch.no_grad()
    def first_step(self):
        for group in self.param_groups:
            grad_norm_ = grad_norm(group)
            scale = group["rho"] / (grad_norm_ + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = torch.sign(p.grad) * torch.abs(p.grad) * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_opt.step()  # do the actual "sharpness-aware" update


def grad_norm(param_group):
    param_vector = torch.tensor([param for param in param_group['param'] if param.grad is not None],
                                device=torch.device('cuda'))
    return torch.linalg.norm(param_vector, ord=2, dim=0)
