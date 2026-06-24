import math
import torch
import torch.nn as nn

class FlowPrior(nn.Module):
    def __init__(self, nets, nett, num_flows, D):
        super(FlowPrior, self).__init__()
        self.D = D
        self.t = torch.nn.ModuleList([nett() for _ in range(num_flows)])
        self.s = torch.nn.ModuleList([nets() for _ in range(num_flows)])
        self.num_flows = num_flows

    def coupling(self, x, index, forward=True):
        (xa, xb) = torch.chunk(x, 2, 1)
        s = self.s[index](xa)
        t = self.t[index](xa)
        s = torch.clamp(s, min=-0.1, max=0.1)
        t = torch.clamp(t, min=-0.1, max=0.1)
        if forward:
            yb = (xb - t) * torch.exp(-s)
        else:
            yb = torch.exp(s) * xb + t
        return torch.cat((xa, yb), 1), s

    def permute(self, x):
        return x.flip(1)

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in range(self.num_flows):
            z, s = self.coupling(z, i, forward=True)
            z = self.permute(z)
            log_det_J = log_det_J - s.sum(dim=1)
        return z, log_det_J

    def f_inv(self, z):
        x = z
        for i in reversed(range(self.num_flows)):
            x = self.permute(x)
            x, _ = self.coupling(x, i, forward=False)
        return x

    def sample(self, batch_size):
        z = torch.randn(batch_size, self.D)
        x = self.f_inv(z)
        return x.view(-1, self.D)

    def log_prob(self, x):
        z, log_det_J = self.f(x)
        log_p = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(dim=1) + log_det_J
        return log_p
