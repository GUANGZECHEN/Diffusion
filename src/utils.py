import matplotlib.pyplot as plt

def show_images(images, masked, recon):
    def show(x):
        return (x * 0.5 + 0.5).clamp(0,1)

    plt.figure(figsize=(8,6))

    for i in range(len(images)):
        plt.subplot(3,len(images),i+1)
        plt.imshow(show(images[i]).detach().cpu().permute(1,2,0))
        plt.axis("off")

        plt.subplot(3,len(images),i+1+len(images))
        plt.imshow(show(masked[i]).detach().cpu().permute(1,2,0))
        plt.axis("off")

        plt.subplot(3,len(images),i+1+2*len(images))
        plt.imshow(show(recon[i]).detach().cpu().permute(1,2,0))
        plt.axis("off")

    plt.tight_layout()
    plt.show()
