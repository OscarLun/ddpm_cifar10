import torch
import argparse
import toml
import wandb
import time
import numpy as np
import os

from models.unet import Unet
from diffusion.gaussian_diffusion import GaussianDiffusion
from trainers.trainer import Trainer

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import Subset

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DDPM Training Script')
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use for training (default: cuda)")
    parser.add_argument("--exp_name", type=str, default="ddpm_cifar10",
                        help="Experiment name for logging (default: ddpm_cifar10)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="Mode to run the script in (default: train)")
    args = parser.parse_args()

    # Set the device
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    config = toml.load('configs/cifar10.toml')

    # Load trainer configuration
    trainer_config = config["trainer_params"]

    subset_size = config["subset_params"]["subset_size"]

    # Create a folder for results
    results_folder=trainer_config["results_folder"]
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    if trainer_config["load_from_config"]:
        current_results_folder = trainer_config["load_path"]
    else:
        current_results_folder = f"{results_folder}/{subset_size}_{current_time}"

    os.makedirs(current_results_folder, exist_ok=True)

    # Load full dataset
    full_dataset = CIFAR10(
        root=config["trainer_params"]["folder"],
        train=True,
        download=True,
        transform=Compose([
            ToTensor(),
        ])
    )

    # Handle subset indices for training
    subset_indices_file = "subset_indices.npy"
    subset_indices_file = os.path.join(current_results_folder, subset_indices_file)

    if os.path.exists(subset_indices_file) and config["trainer_params"]["load_path"] is not None:
        # Load existing subset indices
        subset_indices = np.load(subset_indices_file)
        print(f"Loaded subset indices from {subset_indices_file}")
    else:
        # Generate new subset indices and save them
        subset_indices = np.random.choice(len(full_dataset), size=subset_size, replace=False)
        np.save(subset_indices_file, subset_indices)
        print(f"Saved new subset indices to {subset_indices_file}")

    # Initialize U-Net model
    unet_config = config['unet_params']
    model = Unet(
        dim=unet_config['dim'],
        dim_mults=unet_config['dim_mults'],
        channels=unet_config['channels'], 
    )
    model.to(device)

    # Initialize Gaussian Diffusion
    diffusion_config = config["diffusion_params"]
    diffusion = GaussianDiffusion(
        model,
        image_size=diffusion_config["image_size"],
        timesteps=diffusion_config["timesteps"],
        beta_schedule=diffusion_config["beta_schedule"],
        device=device,
    )
    diffusion.to(device)

    # Initialize wandb
    wandb_logger = wandb.init(
        project="denoising-diffusion-cifar",  
        name=f"{args.exp_name}_{subset_size}",
        config={
            "unet_params": unet_config,
            "diffusion_params": diffusion_config,
            "trainer_params": config["trainer_params"],
        },
        mode = "online" if args.mode == "train" else "disabled",
    )

    
    # Subset dataset 
    dataset = Subset(full_dataset, subset_indices)

    # Initialize Trainer
    trainer = Trainer(
        diffusion_model=diffusion,
        dataset=dataset,
        folder=trainer_config["folder"],
        train_batch_size=trainer_config["train_batch_size"],
        train_lr=trainer_config["train_lr"],
        train_num_steps=trainer_config["train_num_steps"],
        save_and_sample_every=trainer_config["save_and_sample_every"],
        num_samples=trainer_config["num_samples"],
        results_folder=current_results_folder,
        wandb_logger=wandb_logger,
        device=device,
        num_fid_samples=trainer_config["num_fid_samples"],
        calculate_fid=False,
        load_milestone=trainer_config["load_milestone"],
        load_path=trainer_config["load_path"],
    )

    if args.mode == "test":
        # Test mode
        trainer.test(save_samples=True)
        print("Testing completed.")

    else:

        # Start training
        if trainer_config["load_from_config"]:
            print("Continue training from time step = ", trainer_config["load_milestone"])
        else:
            print("NEW training is started with subset size = ", subset_size)

        num_params = sum(p.numel() for p in diffusion.parameters())
        print("The number of parameters = ", num_params)
        trainer.train()

    # Finish wandb logging
    wandb.finish()

if __name__ == '__main__':
    main()
