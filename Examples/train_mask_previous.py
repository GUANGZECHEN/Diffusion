import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 1. Dataset
# =========================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transform
)

loader = DataLoader(dataset, batch_size=64, shuffle=True)

# =========================
# 2. Diffusion schedule
# =========================

T = 200

beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

def forward_diffusion(x0, t):
    noise = torch.randn_like(x0)

    sqrt_alpha_bar = alpha_bar[t].view(-1,1,1,1)
    sqrt_one_minus = torch.sqrt(1 - alpha_bar[t]).view(-1,1,1,1)

    xt = sqrt_alpha_bar * x0 + sqrt_one_minus * noise

    return xt, noise

# =========================
# 3. Masking (IMPROVED)
# =========================

def add_mask(img):
    _, h, w = img.shape

    mask_size = random.choice([4, 6, 8])

    x = torch.randint(0, h - mask_size, (1,))
    y = torch.randint(0, w - mask_size, (1,))

    mask = torch.ones(1, h, w, device=img.device)
    mask[:, x:x+mask_size, y:y+mask_size] = 0

    # 🔥 KEY: fill missing region with noise
    masked_img = img * mask + torch.randn_like(img) * (1 - mask)

    return masked_img, mask

# =========================
# 4. Time embedding
# =========================

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        t = t.float().unsqueeze(-1) / T
        return self.net(t)

# =========================
# 5. Improved Shallow U-Net
# =========================

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_mlp = TimeEmbedding(128)

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

        self.conv_out = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x, masked, mask, t):
        t_emb = self.time_mlp(t).view(-1, 128, 1, 1)

        x = torch.cat([x, masked, mask], dim=1)

        x1 = self.conv1(x)             # (B,64,32,32)

        x2 = self.pool(x1)
        x2 = self.conv2(x2)            # (B,128,16,16)

        x3 = self.conv3(x2)
        x3 = x3 + t_emb               # inject time

        x4 = self.up(x3)
        x4 = torch.cat([x4, x1], dim=1)
        x4 = self.conv4(x4)

        out = self.conv_out(x4)

        return out

model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# =========================
# 6. Training (IMPROVED LOSS)
# =========================

epochs = 25

for epoch in range(epochs):
    for images, _ in loader:
        images = images.to(device)

        masked_imgs = []
        masks = []

        for img in images:
            masked, mask = add_mask(img)
            masked_imgs.append(masked)
            masks.append(mask)

        masked_imgs = torch.stack(masked_imgs)
        masks = torch.stack(masks)

        t = torch.randint(0, T, (images.size(0),), device=device)

        xt, noise = forward_diffusion(images, t)

        pred_noise = model(xt, masked_imgs, masks, t)

        # 🔥 Focus loss on missing region
        loss = ((pred_noise - noise) ** 2 * (1 - masks)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# =========================
# 7. Sampling
# =========================

@torch.no_grad()
def inpaint_sample(model, masked_imgs, masks):
    model.eval()

    x = torch.randn_like(masked_imgs).to(device)

    for t in reversed(range(T)):
        t_tensor = torch.full((masked_imgs.size(0),), t, device=device)

        beta_t = beta[t]
        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t]

        pred_noise = model(x, masked_imgs, masks, t_tensor)

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise
        ) + torch.sqrt(beta_t) * noise

        # enforce known pixels
        x = x * (1 - masks) + masked_imgs * masks

    return x

# =========================
# 8. Visualization
# =========================

images, _ = next(iter(loader))
images = images[:4].to(device)

masked_imgs = []
masks = []

for img in images:
    masked, mask = add_mask(img)
    masked_imgs.append(masked)
    masks.append(mask)

masked_imgs = torch.stack(masked_imgs)
masks = torch.stack(masks)

recon = inpaint_sample(model, masked_imgs, masks)

def show(x):
    return (x * 0.5 + 0.5).clamp(0,1)

plt.figure(figsize=(8,6))

for i in range(4):
    plt.subplot(3,4,i+1)
    plt.imshow(show(images[i]).cpu().permute(1,2,0))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(3,4,i+5)
    plt.imshow(show(masked_imgs[i]).cpu().permute(1,2,0))
    plt.title("Masked")
    plt.axis("off")

    plt.subplot(3,4,i+9)
    plt.imshow(show(recon[i]).cpu().permute(1,2,0))
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.show()
