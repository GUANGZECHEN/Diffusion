import torch
import time
import os
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import UNet
from diffusion import Diffusion

# =========================
# Mask (batch version, faster)
# =========================

def add_mask_batch(images):
    B, C, H, W = images.shape
    mask = torch.ones(B, 1, H, W, device=images.device)

    for i in range(B):
        size = random.choice([4, 6, 8])
        x = torch.randint(0, H - size, (1,))
        y = torch.randint(0, W - size, (1,))
        mask[i, :, x:x+size, y:y+size] = 0

    masked = images * mask + torch.randn_like(images) * (1 - mask)
    return masked, mask

# =========================
# Training
# =========================

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 200

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = UNet(T).to(device)
    diffusion = Diffusion(T, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5  # fast demo

    os.makedirs("checkpoints", exist_ok=True)

    best_loss = float("inf")

    print("Starting training...\n")
    total_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0

        for images, _ in loader:
            images = images.to(device)

            masked_imgs, masks = add_mask_batch(images)

            t = torch.randint(0, T, (images.size(0),), device=device)

            xt, noise = diffusion.forward(images, t)

            pred = model(xt, masked_imgs, masks, t)

            loss = ((pred - noise)**2 * (1 - masks)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

        # 🔥 Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print(f"   Saved best model (loss={best_loss:.4f})")

    total_time = time.time() - total_start

    print("\nTraining complete.")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
