# model.py

import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, dim, T):
        super().__init__()
        self.T = T
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        t = t.float().unsqueeze(-1) / self.T
        return self.net(t)


class UNet(nn.Module):
    def __init__(self, T):
        super().__init__()

        self.time_mlp = TimeEmbedding(128, T)
        self.relu = nn.ReLU()

        self.conv1 = nn.Sequential(
            nn.Conv2d(7, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU()
        )

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv4 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU()
        )

        self.out = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x, masked, mask, t):
        t_emb = self.time_mlp(t).view(-1, 128, 1, 1)

        x = torch.cat([x, masked, mask], dim=1)

        x1 = self.conv1(x)
        x2 = self.pool(x1)
        x2 = self.conv2(x2)

        x3 = self.conv3(x2)
        x3 = x3 + t_emb

        x4 = self.up(x3)
        x4 = torch.cat([x4, x1], dim=1)
        x4 = self.conv4(x4)

        return self.out(x4)
