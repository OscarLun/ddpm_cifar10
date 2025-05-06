import math
import wandb

from pathlib import Path
from functools import partial
from multiprocessing import cpu_count
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torchvision import transforms as T, utils

from tqdm.auto import tqdm
from accelerate import Accelerator
from ema_pytorch import EMA
from utils.helpers import exists, cycle, num_to_groups, has_int_squareroot, divisible_by
from version import __version__

class DatasetNoLabels(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get the image and label from the dataset
        image, _ = self.dataset[idx] # Ignore the label

        return image

class Trainer:
    def __init__(
        self,
        diffusion_model,
        dataset,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False,
        wandb_logger,
        device = torch.device('cpu'),
        load_milestone = 0,
        load_path = None,
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no',
        )

        self.device = device

        self.wandb_logger = wandb_logger

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # default convert_image_to depending on channels

        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        # self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        self.ds = DatasetNoLabels(dataset)

        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        # for logging results in a folder periodically
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # load milestone
        self.load_milestone = load_milestone

        # load path
        self.load_path = load_path

        # load model if milestone is provided
        if self.load_milestone > 0:
            self.load(self.load_milestone, load_path = self.load_path)


        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            from fid_evaluation import FIDEvaluation

            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )

            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

        self.num_fid_samples = num_fid_samples
        self.inception_block_idx = inception_block_idx

    # @property
    # def device(self):
    #     return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        # Save the current checkpoint
        checkpoint_path = self.results_folder / f'model-{milestone}.pt'
        torch.save(data, str(checkpoint_path))

        # Remove the previous checkpoint if it exists
        if milestone > 1:
            previous_checkpoint_path = self.results_folder / f'model-{milestone - self.save_and_sample_every}.pt'
            if previous_checkpoint_path.exists():
                previous_checkpoint_path.unlink()  # Deletes the file
            

    def load(self, milestone, load_path=None):
        if milestone == 0:
            return

        accelerator = self.accelerator
        device = accelerator.device

        try: 
            # Either load from the specified path or the default path
            if load_path is not None:
                data_path = Path(load_path) / f'model-{milestone}.pt'
            else:
                data_path = self.results_folder / f'model-{milestone}.pt'

            # Check if the file exists
            if not data_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {data_path}")
            
            data = torch.load(str(data_path), map_location=device, weights_only=True)
            
            print(f"Loading model from {data_path}")

            model = self.accelerator.unwrap_model(self.model)
            model.load_state_dict(data['model'])

            self.step = data['step']
            self.opt.load_state_dict(data['opt'])
            if self.accelerator.is_main_process:
                self.ema.load_state_dict(data["ema"])

            if 'version' in data:
                print(f"loading from version {data['version']}")

            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])

        except FileNotFoundError as e:
            accelerator.print(f"Checkpoint not found: {e}")

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                log_dict = {}

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):

                    data = next(self.dl)
                    data = data.to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                loss_dict = {
                    'step': self.step,
                    'loss': total_loss
                }

                log_dict.update(loss_dict)
                pbar.set_description(f'loss: {total_loss:.4f}')
        

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            # milestone = self.step // self.save_and_sample_every
                            milestone = self.step
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0)

                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                        # whether to calculate fid

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')

                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                # Log to wandb
                self.wandb_logger.log(log_dict)
                pbar.update(1)

        accelerator.print('training complete')

    def test(self, save_samples = False): 
        from fid_evaluation import FIDEvaluation

        if self.num_fid_samples > len(self.ds):
            self.num_fid_samples = len(self.ds)

        fid_scorer = FIDEvaluation(
        batch_size=self.batch_size,
        dl=self.dl,
        sampler=self.ema.ema_model,
        channels=self.channels,
        accelerator=self.accelerator,
        stats_dir=self.results_folder,
        device=self.device,
        num_fid_samples=self.num_fid_samples,
        inception_block_idx=self.inception_block_idx
        )
        
        fid_score = fid_scorer.fid_score(save_samples=save_samples)
        self.accelerator.print(f'fid_score: {fid_score}')
    