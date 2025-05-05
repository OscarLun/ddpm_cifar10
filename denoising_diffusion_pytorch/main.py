import torch
import argparse
import toml
import wandb
from models.unet import Unet
from diffusion.gaussian_diffusion import GaussianDiffusion
from trainers.trainer import Trainer

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DDPM Training Script')
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use for training (default: cuda)")
    parser.add_argument("--exp_name", type=str, default="ddpm_cifar10",
                        help="Experiment name for logging (default: ddpm_cifar10)")
    args = parser.parse_args()

    # Set the device
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    config = toml.load('configs/cifar10.toml')

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
        name=args.exp_name,
        config={
            "unet_params": unet_config,
            "diffusion_params": diffusion_config,
            "trainer_params": config["trainer_params"],
        },
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
        results_folder=trainer_config["results_folder"],
        wandb_logger=wandb_logger,
        device=device,
	calculate_fid=False
    )

    # Start training
    print("NEW training is started")
    num_params = sum(p.numel() for p in diffusion.parameters())
    print("The number of parameters = ", num_params)
    trainer.train()

if __name__ == '__main__':
    main()
