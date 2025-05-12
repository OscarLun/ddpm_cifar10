import os
import numpy as np
import toml
from collections import defaultdict
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor



def process_cifar(overwrite_subset=False, overwrite_fid=False):

    # Load configuration
    config = toml.load('configs/cifar10.toml')
    folder = config["trainer_params"]["folder"]

    # Load the CIFAR-10 training dataset
    transform = Compose([ToTensor()])
    train_data = CIFAR10(
        root=folder,
        train=True,
        download=True,
        transform=transform,
    )

    test_data = CIFAR10(
        root=folder,
        train=False,
        download=True,
        transform=transform,
    )

    indices_dir = os.path.join(folder, "subset_indices")

    # Create a subfolder within the data folder for subset indices
    os.makedirs(indices_dir, exist_ok=True)

    # Define subset sizes
    subset_sizes = config["subset_params"]["subset_sizes"]
    fid_test_size = config["subset_params"]["num_train_fid_samples"]

    # Group all training indices by class. CIFAR10 has a 'targets' attribute.
    indices_by_class_train = defaultdict(list)
    for idx, label in enumerate(train_data.targets):
        indices_by_class_train[label].append(idx)

    # Same for test data
    indices_by_class_test = defaultdict(list)
    for idx, label in enumerate(test_data.targets):
        indices_by_class_test[label].append(idx)

    num_classes = len(indices_by_class_train)
    # Generate and save subset indices for each subset size
    for subset_size in subset_sizes:
        # Ensure the subset_size is exactly divisible by the number of classes
        if subset_size % num_classes != 0:
            raise ValueError(f"subset_size {subset_size} is not divisible by the number of classes ({num_classes}). " 
                             "Please choose a subset_size that is divisible by {num_classes}.")
       
        subset_indices_file = os.path.join(indices_dir, f"subset_indices_{subset_size}.npy")

        # Check if the subset indices file already exists
        if os.path.exists(subset_indices_file) and not overwrite_subset:
            print(f"Subset indices for size {subset_size} already exist at {subset_indices_file}.")
        else:
            samples_per_class = subset_size // num_classes
            stratified_indices = []
            for cls in range(num_classes):
                chosen = np.random.choice(
                    indices_by_class_train[cls],
                    size=samples_per_class,
                    replace=False
                )
                stratified_indices.extend(chosen)
            stratified_indices = np.array(stratified_indices)
            np.random.shuffle(stratified_indices)
            np.save(subset_indices_file, stratified_indices)
            print(f"Saved subset indices for size {subset_size} to {subset_indices_file}")


    # Generate and save stratified subset indices for FID test samples, with separate overwrite control
    fid_indices_file = os.path.join(indices_dir, f"subset_indices_fid_{fid_test_size}.npy")
    if os.path.exists(fid_indices_file) and not overwrite_fid:
        print(f"FID test indices already exist at {fid_indices_file}.")
    else:
        if fid_test_size % num_classes != 0:
            raise ValueError(f"fid_test_size {fid_test_size} is not divisible by the number of classes ({num_classes}). "
                             "Please choose a fid_test_size divisible by {num_classes}.")
        samples_per_class_fid = fid_test_size // num_classes
        stratified_indices_fid = []
        for cls in range(num_classes):
            chosen = np.random.choice(
                indices_by_class_test[cls],
                size=samples_per_class_fid,
                replace=False
            )
            stratified_indices_fid.extend(chosen)
        stratified_indices_fid = np.array(stratified_indices_fid)
        np.random.shuffle(stratified_indices_fid)
        np.save(fid_indices_file, stratified_indices_fid)
        print(f"Saved FID test subset indices to {fid_indices_file}")

if __name__ == "__main__":

    np.random.seed(42)  # Set a random seed for reproducibility
    process_cifar()
