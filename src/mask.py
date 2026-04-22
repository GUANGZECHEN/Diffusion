# mask.py

import torch
import random

def add_mask(img):
    _, h, w = img.shape

    mask_size = random.choice([4, 6, 8])

    x = torch.randint(0, h - mask_size, (1,))
    y = torch.randint(0, w - mask_size, (1,))

    mask = torch.ones(1, h, w, device=img.device)
    mask[:, x:x+mask_size, y:y+mask_size] = 0

    masked = img * mask + torch.randn_like(img) * (1 - mask)

    return masked, mask
