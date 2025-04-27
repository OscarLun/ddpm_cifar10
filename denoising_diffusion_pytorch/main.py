import torch
import toml
import wandb
from models.unet import Unet
from diffusion.gaussian_diffusion import GaussianDiffusion
from trainers.trainer import Trainer

def main():
    # Load configuration
    config = toml.load('configs/cifar10.toml')


    # Initialize U-Net model
    unet_config = config['unet_params']
    model = Unet(
        dim=unet_config['dim'],
        dim_mults=unet_config['dim_mults'],
        channels=unet_config['channels']
    )

    # Initialize Gaussian Diffusion
    diffusion_config = config["diffusion_params"]
    diffusion = GaussianDiffusion(
        model,
        image_size=diffusion_config["image_size"],
        timesteps=diffusion_config["timesteps"],
        beta_schedule=diffusion_config["beta_schedule"]
    )

    # Initialize Trainer
    trainer_config = config["trainer_params"]
    trainer = Trainer(
        diffusion_model=diffusion,
        folder=trainer_config["folder"],
        train_batch_size=trainer_config["train_batch_size"],
        train_lr=trainer_config["train_lr"],
        train_num_steps=trainer_config["train_num_steps"],
        save_and_sample_every=trainer_config["save_and_sample_every"],
        num_samples=trainer_config["num_samples"],
        results_folder=trainer_config["results_folder"]
    )

    # Start training
    print("NEW training is started")
    num_params = sum(p.numel() for p in diffusion.parameters())
    print("The number of parameters = ", num_params)
    trainer.train()

if __name__ == '__main__':
    main()