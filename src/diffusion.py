# diffusion.py

import torch

class Diffusion:
    def __init__(self, T, device):
        self.T = T
        self.device = device

        self.beta = torch.linspace(1e-4, 0.02, T).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward(self, x0, t):
        noise = torch.randn_like(x0)

        sqrt_alpha_bar = self.alpha_bar[t].view(-1,1,1,1)
        sqrt_one_minus = torch.sqrt(1 - self.alpha_bar[t]).view(-1,1,1,1)

        xt = sqrt_alpha_bar * x0 + sqrt_one_minus * noise

        return xt, noise
