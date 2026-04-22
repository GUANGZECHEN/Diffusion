import sys
import os

sys.path.append(os.path.abspath("../src"))

"""
Example: Diffusion-based image inpainting on CIFAR-10.

This script:
1. Loads a trained model (or randomly initialized if not provided)
2. Applies random masks to images
3. Reconstructs missing regions via diffusion
4. Visualizes results
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import UNet
from diffusion import Diffusion
from mask import add_mask
from sample import inpaint_sample
from utils import show_images

# =========================
# Setup
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
T = 200

# =========================
# Dataset
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

loader = DataLoader(dataset, batch_size=4, shuffle=True)

# =========================
# Load data
# =========================

images, _ = next(iter(loader))
images = images.to(device)

# =========================
# Apply masks
# =========================

masked_imgs = []
masks = []

for img in images:
    m_img, m = add_mask(img)
    masked_imgs.append(m_img)
    masks.append(m)

masked_imgs = torch.stack(masked_imgs).to(device)
masks = torch.stack(masks).to(device)

# =========================
# Model + diffusion
# =========================

model = UNet(T).to(device)
diffusion = Diffusion(T, device)

# =========================
# Load trained weights (IMPORTANT)
# =========================

try:
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    print("Loaded trained model.")
except:
    print("No trained model found. Using random weights (results will be poor).")

# =========================
# Inpainting
# =========================

recon = inpaint_sample(model, diffusion, masked_imgs, masks)

# =========================
# Visualization
# =========================

show_images(images, masked_imgs, recon)
