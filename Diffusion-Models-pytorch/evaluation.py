import torch
from modules import UNet_conditional
from utils import plot_images, save_images
from ddpm_conditional import Diffusion  # Assuming `Diffusion` is in a file named `diffusion.py`

def evaluate_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 64  # Image size used during training
    num_classes = 200  # Match the number of classes used during training

    # Load the trained model
    model = UNet_conditional(num_classes=num_classes).to(device)
    ckpt_path = "./models/DDPM_conditional/ema_ckpt.pt"  # Path to the saved checkpoint
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    # Initialize the diffusion process
    diffusion = Diffusion(img_size=img_size, device=device)

    # Generate samples
    n = 1  # Number of images to generate
    target_class = 10  # Class for which to generate images
    labels = torch.Tensor([target_class] * n).long().to(device)  # Repeat class 50 `n` times
    sampled_images = diffusion.sample(model, n=n, labels=labels)
    print(f"shape of sampled image = {sampled_images.shape}")

    # Visualize or save the generated images
    # plot_images(sampled_images)  # Plot the images
    save_images(sampled_images, f"./evaluation/evaluation_samples_{target_class}.jpg")  # Save the images to disk

if __name__ == "__main__":

    evaluate_model()
