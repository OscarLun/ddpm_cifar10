import torch
import argparse
import toml
import wandb
import time
import numpy as np
import os
import random

from models.unet import Unet
from diffusion.gaussian_diffusion import GaussianDiffusion
from trainers.trainer import Trainer
from nearest_neighbor import NearestNeighborEvaluator

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import Subset
from torch.utils.data import DataLoader

def evaluate_diversity(model, device, num_samples, real_data, batch_size=128):
    """
    Evaluate the diversity of generated samples Nearest Neighbor distance.
    """
    nn_eval = NearestNeighborEvaluator(device=device, n_neighbors=5, real_images=real_data)
    nn_eval.fit_database()

    # Generate samples
    fake_images = model.sample(batch_size=batch_size).to(device)

    nn_eval.save_nearest_neighbors(
        generated_images=fake_images,
        n_examples=10,
    )


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DDPM Training Script')
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use for training (default: cuda)")
    parser.add_argument("--exp_name", type=str, default="ddpm_cifar10",
                        help="Experiment name for logging (default: ddpm_cifar10)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "nn_eval", "sample"],
                        help="Mode to run the script in (default: train)")
    parser.add_argument("--subset_size", type=int, default=None,
                        help="Subset size for training (default: None, will use config value)")
    parser.add_argument("--train_steps", type=int, default=None,
                        help="Number of training steps (default: None, will use config value)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to pretrained model (default: None)")
    
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
        
    # Check if training steps are provided, otherwise use the config value
    if args.train_steps is not None:
        train_steps = args.train_steps
        save_and_sample_every = train_steps // 8 # For current implementation
    else:
        save_and_sample_every = config["trainer_params"]["save_and_sample_every"]
        if "train_num_steps" in config["trainer_params"]:
            train_steps = config["trainer_params"]["train_num_steps"]
        else:
            raise ValueError("No training steps provided. Please specify training steps in the command line or in the config file.")
        

    # Check if model path is provided
    if args.model_path is not None:
        load_path = args.model_path
        trainer["load_from_config"] = True
    else:
        load_path = trainer_config["load_path"]

    # Create a folder for results
    results_folder=trainer_config["results_folder"]
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    if trainer_config["load_from_config"]:
        # Get parent folder of the load path
        current_results_folder = os.path.dirname(load_path)
    else:
        current_results_folder = f"{results_folder}/{subset_size}_{current_time}"

    #os.makedirs(current_results_folder, exist_ok=True)

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

    # Subset dataset 
    train_dataset = Subset(train_data, subset_indices_train)

    if args.mode == "test":
        # Use entire test dataset for testing
        test_dataset = test_data
    else:
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
        train_num_steps=train_steps,
        save_and_sample_every=save_and_sample_every,
        num_samples=trainer_config["num_samples"],
        results_folder=current_results_folder,
        wandb_logger=wandb_logger,
        device=device,
        num_train_fid_samples=trainer_config["num_train_fid_samples"],
        num_test_fid_samples=trainer_config["num_test_fid_samples"],
        calculate_fid=True,
        load_path=load_path,
        load_from_config=trainer_config["load_from_config"],
        save_best_and_latest_only=True,
    )

    if args.mode == "test":
        # Test mode
        trainer.test()
        print("Testing completed.")

    elif args.mode == "nn_eval":
        # Nearest Neighbor evaluation
        model = trainer.get_ema_model()
        num_samples = 10
        batch_size = trainer_config["train_batch_size"]
        evaluate_diversity(model, device, num_samples, train_dataset, batch_size)

    elif args.mode == "sample":
        if not trainer_config["load_from_config"]:
            raise ValueError("No path to pretrained model given")

        trainer.sample()

    else:
        
        # Create results folder
        os.makedirs(current_results_folder, exist_ok=True)
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
