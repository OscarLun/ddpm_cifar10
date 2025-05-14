import torch
import torch.nn.functional as F
import numpy as np
import os
import math
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from pytorch_fid.inception import InceptionV3
from einops import rearrange, repeat
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class DatasetNoLabels(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get the image and label from the dataset
        image, _ = self.dataset[idx] # Ignore the label

        return image

class NearestNeighborEvaluator:
    def __init__(self, 
                 device=torch.device("cpu"), 
                 n_neighbors=5, 
                 inception_block_idx=2048,
                 channels=3,
                 stats_dir="./results",
                 real_images=None,
                 fake_images=None,
                 batch_size=128,
                 ):
        """
        device: torch.device to run computations.
        n_neighbors: Number of nearest neighbors to find.
        """
        self.device = device
        self.n_neighbors = n_neighbors
        self.channels = channels
        self.stats_dir = stats_dir
        self.real_images = real_images
        self.fake_images = fake_images
        self.batch_size = batch_size

        # Prepare dataloader for real images
        self.ds = DatasetNoLabels(real_images)
        self.dl = DataLoader(self.ds, batch_size=batch_size, shuffle=False)

        self.data_size = len(self.ds)

        # Load InceptionV3 pretrained on ImageNet.
        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        self.inception_v3 = InceptionV3([block_idx]).to(device)

        # NearestNeighbors instance (will be fit with real image features)
        self.neighbor_model = None
        # Variable to cache stacked features if needed
        self.database_features = None


    def load_or_precalc_database_features(self, cache_directory="database_stats"):
        cache_dir = os.path.join(self.stats_dir, cache_directory)
        os.makedirs(cache_dir, exist_ok=True)

        cache_filename = f"{self.data_size}_stats.npz"

        cache_path = os.path.join(cache_dir, cache_filename)

        if os.path.exists(cache_path):
            ckpt = np.load(cache_path)
            stacked_features = ckpt["features"]
            print(f"Database features loaded from {cache_path}.")
            ckpt.close()
        else:
            stacked_features_list = []
            print(f"Extracting Inception features for real images...")
            for images in tqdm(self.dl):
                images = images.to(self.device)
                features = self.extract_features(images)
                stacked_features_list.append(torch.tensor(features))

            stacked_features = torch.cat(stacked_features_list, dim=0).cpu().numpy()
            np.savez_compressed(cache_path, features=stacked_features)
            print(f"Database features cached to {cache_path}.")

        self.database_features = stacked_features
        return stacked_features
        
    def extract_features(self, samples):
        """
        Extract feature representations from images using InceptionV3.
        images: torch.Tensor of shape (N, C, H, W).
        Returns numpy array of features.
        """
        if self.channels == 1:
            samples = repeat(samples, "b 1 ... -> b c ...", c=3)

        self.inception_v3.eval()
        features = self.inception_v3(samples)[0]

        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        return features.cpu().numpy()


    def fit_database(self):
        """
        Fit the nearest neighbor model on real images.
        real_images: torch.Tensor (N, C, H, W).
        """
        features = self.load_or_precalc_database_features()
        self.neighbor_model = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')
        self.neighbor_model.fit(features)

    def find_nearest(self, generated_images, save_images=False):
        """
        Find nearest neighbors in the real image database for the generated images.
        generated_images: torch.Tensor (M, C, H, W) in [-1,1].
        Returns: distances and indices arrays.
        """
        if self.neighbor_model is None:
            raise ValueError("Please fit the neighbor model with real images via fit_database() first.")
        features = self.extract_features(generated_images)
        distances, indices = self.neighbor_model.kneighbors(features)
        return distances, indices

    def compute_average_distance(self, generated_images):
        """
        Compute a diversity measure for generated images using average nearest neighbor distance.
        Lower values suggest that generated images are very similar to some real images (possible overfitting).
        """
        distances, _ = self.find_nearest(generated_images)
        return np.mean(distances)
    
    def save_nearest_neighbors(self, generated_images, save_path="./results", n_examples=10):
        real_images = self.ds

        if self.neighbor_model is None:
            raise ValueError("Please fit the neighbor model first.")
        
        distances, indices = self.find_nearest(generated_images)

        save_dir = os.path.join(save_path, "nearest_neighbors")
        os.makedirs(save_dir, exist_ok=True)

        save_filename = f"{self.data_size}_nn.png"
        save_path = os.path.join(save_dir, save_filename)

        def convert(img_tensor):
            return img_tensor  # CIFAR10 already in [0,1]

        gen_imgs = generated_images[:n_examples]
        nn_indices = indices[:n_examples]
        n_neighbors = nn_indices.shape[1]

        fig, axes = plt.subplots(n_examples, n_neighbors + 1, figsize=((n_neighbors+1)*2, n_examples*2))

        if n_examples == 1:
            axes = np.expand_dims(axes, 0)

        for i in range(n_examples):
            gen_img = convert(gen_imgs[i]).cpu().permute(1, 2, 0).numpy()
            axes[i, 0].imshow(gen_img.clip(0, 1))
            axes[i, 0].set_title("Generated")
            axes[i, 0].axis("off")

            for j in range(n_neighbors):
                real_idx = nn_indices[i, j]
                real_img = convert(real_images[real_idx]).cpu().permute(1, 2, 0).numpy()
                axes[i, j + 1].imshow(real_img.clip(0, 1))
                axes[i, j + 1].set_title(f"NN {j+1}")
                axes[i, j + 1].axis("off")

        plt.tight_layout()

        print(f"Nearest Neihbors saved to {save_path}.")
        plt.savefig(save_path)
        plt.close(fig)
