# Diffusion-Based Image Inpainting

This repository implements a simplified conditional diffusion model for image inpainting on CIFAR-10.

The model learns to reconstruct missing regions by iteratively denoising Gaussian noise, conditioned on visible pixels.

---

## Structure

- `source/`: core implementation (model, diffusion, training, sampling)
- `examples/`: runnable scripts
- `figs/`: generated results

---

## Method

- Forward process: Gaussian noise injection
- Reverse process: neural network predicts noise
- Conditioning: masked image + binary mask
- Inpainting: known pixels enforced during sampling

---

## Results

Example reconstruction:

![result](figs/inpainting_results.png)

---

## Usage

```bash
pip install -r requirements.txt
python examples/run_inpainting.py
