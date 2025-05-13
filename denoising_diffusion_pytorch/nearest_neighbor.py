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

        # Load InceptionV3 pretrained on ImageNet.
        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        self.inception_v3 = InceptionV3([block_idx]).to(device)

        # NearestNeighbors instance (will be fit with real image features)
        self.neighbor_model = None
        # Variable to cache stacked features if needed
        self.database_features = None

    # def _preprocess(self, images):
    #     """
    #     Preprocess images for InceptionV3.
    #     Assumes images is a torch.Tensor of shape (N, C, H, W) in the range [-1, 1].
    #     Converts them to [0, 1], resizes to 299x299, and applies standard InceptionV3 normalization.
    #     """
    #     # Convert from [-1,1] to [0,1]
    #     images = (images + 1) / 2
    #     # Resize images to 299 x 299 (InceptionV3 input size)
    #     images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    #     # Apply the standard InceptionV3 normalization:
    #     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                      std=[0.229, 0.224, 0.225])
    #     # Normalize each image in the batch
    #     images = torch.stack([normalize(img) for img in images])
    #     return images

    def load_or_precalc_database_features(self, cache_filename="database_stats.npz"):
        """
        Loads pre-calculated database features from disk if available. Otherwise,
        extracts Inception features from n_samples real images (using the provided dataloader),
        stacks them, caches to disk, and returns them.

        Parameters:
            dl: DataLoader which yields batches of real images.
            n_samples: Total number of samples to process.
            batch_size: Batch size of the DataLoader.
            cache_filename: Filename (including path) to cache the features.
        Returns:
            stacked_features: numpy array of shape (n_samples, feature_dim)
        """
        if os.path.exists(cache_filename):
            ckpt = np.load(cache_filename)
            stacked_features = ckpt["features"]
            print(f"Database features loaded from {cache_filename}.")
            ckpt.close()
        else:
            n_samples = len(self.dl.dataset) 
            num_batches = int(math.ceil(n_samples / self.batch_size))
            stacked_features_list = []
            print(f"Extracting Inception features for {n_samples} real images...")
            for _ in tqdm(range(num_batches)):
                try:
                    images = next(iter(self.dl))
                except StopIteration:
                    break
                images = images.to(self.device)
                features = self.extract_features(images)
                # Convert to tensor then to numpy for stacking
                features_tensor = torch.tensor(features)
                stacked_features_list.append(features_tensor)
            # Concatenate along first dimension and keep only first n_samples rows.
            stacked_features = torch.cat(stacked_features_list, dim=0)[:n_samples].cpu().numpy()
            np.savez_compressed(cache_filename, features=stacked_features)
            print(f"Database features cached to {cache_filename}.")
        # Cache the features in the instance for later use.
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
    

    def save_nearest_neighbors(self, generated_images, real_images, save_path="nearest_neighbor.png", n_examples=10):
        """
        For the first n_examples in generated_images, finds the 5 nearest real images (from real_images)
        and saves a grid image with each row containing the generated image as the leftmost column
        and its nearest neighbors to the right.
        
        Parameters:
            generated_images: torch.Tensor (M, C, H, W).
            real_images: torch.Tensor (N, C, H, W). The real images used for the nearest neighbor database.
            save_path: Path where the PNG image will be saved.
            n_examples: Number of generated images (rows) to display.
        """
        # Ensure we have fitted the database.
        if self.neighbor_model is None:
            raise ValueError("Please fit the neighbor model with real images via fit_database() first.")
        
        # Get nearest neighbor distances and indices for the generated images.
        distances, indices = self.find_nearest(generated_images)
        
        # Convert images from [-1,1] to [0,1] for display, might need to be adjusted
        def convert(img_tensor):
            return (img_tensor + 1) / 2
        
        # Limit to the first n_examples
        gen_imgs = generated_images[:n_examples]
        nn_indices = indices[:n_examples]  # shape (n_examples, n_neighbors)
        n_neighbors = nn_indices.shape[1]
        
        # Create a matplotlib grid with n_examples rows and (n_neighbors + 1) columns.
        fig, axes = plt.subplots(n_examples, n_neighbors + 1, figsize=((n_neighbors+1)*2, n_examples*2))
        
        # If only one row, axes might not be a 2D array
        if n_examples == 1:
            axes = np.expand_dims(axes, 0)
        
        for i in range(n_examples):
            # Plot the generated image in the leftmost column.
            gen_img = convert(gen_imgs[i]).cpu().permute(1, 2, 0).numpy()
            axes[i, 0].imshow(gen_img.clip(0, 1))
            axes[i, 0].set_title("Generated")
            axes[i, 0].axis("off")
            
            # For each nearest neighbor, plot the corresponding real image.
            for j in range(n_neighbors):
                real_idx = nn_indices[i, j]
                real_img = convert(real_images[real_idx]).cpu().permute(1, 2, 0).numpy()
                axes[i, j + 1].imshow(real_img.clip(0, 1))
                axes[i, j + 1].set_title(f"NN {j+1}")
                axes[i, j + 1].axis("off")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)