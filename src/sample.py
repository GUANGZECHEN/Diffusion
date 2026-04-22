# sample.py

import torch

def inpaint_sample(model, diffusion, masked_imgs, masks):
    model.eval()
    x = torch.randn_like(masked_imgs)

    for t in reversed(range(diffusion.T)):
        t_tensor = torch.full((masked_imgs.size(0),), t, device=x.device)

        beta = diffusion.beta[t]
        alpha = diffusion.alpha[t]
        alpha_bar = diffusion.alpha_bar[t]

        pred = model(x, masked_imgs, masks, t_tensor)

        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        x = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_bar)) * pred
        ) + torch.sqrt(beta) * noise

        x = x * (1 - masks) + masked_imgs * masks

    return x
