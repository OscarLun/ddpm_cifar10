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
    parser.add_argument("--subset_size", type=int, default=None,
                        help="Subset size for training (default: None, will use config value)")

    args = parser.parse_args()

    # Set the device
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    config = toml.load('configs/cifar10.toml')

    # Load trainer configuration
    trainer_config = config["trainer_params"]

    # Load data folder
    data_folder = config["trainer_params"]["folder"]

    # Check if subset size is provided, otherwise use the config value
    if args.subset_size is not None:
        subset_size = args.subset_size
    else:
        if "subset_size" in config["subset_params"]:
            subset_size = config["subset_params"]["subset_size"]
        else:
            raise ValueError("No subset size provided. Please specify a subset size in the command line or in the config file.")

    # Create a folder for results
    results_folder=trainer_config["results_folder"]
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    if trainer_config["load_from_config"]:
        current_results_folder = trainer_config["load_path"]
    else:
        current_results_folder = f"{results_folder}/{subset_size}_{current_time}"

    os.makedirs(current_results_folder, exist_ok=True)

    # Load CIFAR-10 dataset
    train_data = CIFAR10(
        root=data_folder,
        train=True,
        download=True,
        transform=Compose([
            ToTensor(),
        ])
    )

    test_data = CIFAR10(
        root=data_folder,
        train=False,
        download=True,
        transform=Compose([
            ToTensor(),
        ])
    )

    fid_test_size = trainer_config["num_train_fid_samples"]

    subset_indices_folder = os.path.join(data_folder, "subset_indices")
    subset_indices_file_train = os.path.join(subset_indices_folder, f"subset_indices_{subset_size}.npy")
    subset_indices_file_fid = os.path.join(subset_indices_folder, f"subset_indices_fid_{fid_test_size}.npy")

    # Check if indices file exists
    if os.path.exists(subset_indices_file_train) and os.path.exists(subset_indices_file_fid):
        # Load existing subset indices
        subset_indices_train = np.load(subset_indices_file_train)
        subset_indices_fid = np.load(subset_indices_file_fid)
        print(f"Loaded subset indices from {subset_indices_folder}")
    else:
        raise FileNotFoundError(f"Subset indices files not found in {subset_indices_folder}. Please generate them first.")

    # Check if the subset size is valid
    if subset_size not in config["subset_params"]["subset_sizes"]:
        raise ValueError(f"Subset size {subset_size} is not valid. Please choose from {config['subset_params']['subset_sizes']}.")
    
    # Subset dataset 
    train_dataset = Subset(train_data, subset_indices_train)
    test_dataset = Subset(test_data, subset_indices_fid)
    
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
        sampling_timesteps=diffusion_config["sampling_timesteps"],
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
        resume="allow",
    )

    # Initialize Trainer
    trainer = Trainer(
        diffusion_model=diffusion,
        train_data=train_dataset,
        test_data=test_dataset,
        folder=trainer_config["folder"],
        train_batch_size=trainer_config["train_batch_size"],
        train_lr=trainer_config["train_lr"],
        train_num_steps=trainer_config["train_num_steps"],
        save_and_sample_every=trainer_config["save_and_sample_every"],
        num_samples=trainer_config["num_samples"],
        results_folder=current_results_folder,
        wandb_logger=wandb_logger,
        device=device,
        num_train_fid_samples=trainer_config["num_train_fid_samples"],
        num_test_fid_samples=trainer_config["num_test_fid_samples"],
        calculate_fid=True,
        load_milestone=trainer_config["load_milestone"],
        load_path=trainer_config["load_path"],
        load_from_config=trainer_config["load_from_config"],
        save_best_and_latest_only=True,
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
